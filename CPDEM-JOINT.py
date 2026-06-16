"""
方案B深度优化: 软权重加权参数估计

核心思想: 完全模拟CPD-EM的M步 — 不经过匈牙利硬分配,
直接利用gamma矩阵作为权重, 求解带权重的α/β最小二乘问题.

流程:
  1. 用当前α,β分别补偿两个雷达
  2. 运行CPD-EM, 得到每帧gamma (观测×参考的后验概率)
  3. 遍历所有帧, 收集加权点对 (theta_obs, theta_ref, weight=gamma)
  4. 调用加权Stage2优化器, 基于所有权重点对更新α,β
  5. 检查收敛 (α,β变化<阈值)
  6. 不进行匈牙利分配, 不进行异常剔除

加权优化目标:
  min Σ w_k · ||R(-α)·p_A,k - R(-β)·p_B,k - S_BA||²
"""

import numpy as np, hashlib
from scipy.optimize import minimize
import os, warnings, json, glob, time, argparse

def _stable_hash(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)

warnings.filterwarnings('ignore')

from cpdem_cumgamma import (
    CPDEMAssociation, load_cpd_dataset,
    latlon_to_mercator, mercator_to_latlon,
    latlon_to_polar_batch, wrap_angle,
    EARTH_RADIUS, RANDOM_SEED, EPS,
    evaluate_scene_trajectory_level as baseline_evaluate
)
from cpdem_multimethod_compare import _correct_individual_radar

WEIGHT_THRESH = 1e-6  # gamma 权重阈值

def _detect_field_names(frames):
    """
    自动检测参考点字段和观测点字段。
    优先顺序: 参考点: vessel_points, ref_points, ais_points
             观测点: fixed_points, obs_points, radar_points
    返回 (ref_field, obs_field)
    """
    candidates_ref = ['vessel_points', 'ref_points', 'ais_points']
    candidates_obs = ['fixed_points', 'obs_points', 'radar_points']
    ref_field = None
    obs_field = None
    for f in frames:
        for cand in candidates_ref:
            if cand in f:
                ref_field = cand
                break
        for cand in candidates_obs:
            if cand in f:
                obs_field = cand
                break
        if ref_field and obs_field:
            break
    if ref_field is None:
        raise KeyError(f"None of {candidates_ref} found in frames")
    if obs_field is None:
        raise KeyError(f"None of {candidates_obs} found in frames")
    return ref_field, obs_field


def _numerical_hessian(cost_func, x0, eps=1e-6):
    n = len(x0)
    H = np.zeros((n, n))
    f0 = cost_func(x0)
    for i in range(n):
        xp = x0.copy(); xp[i] += eps
        xm = x0.copy(); xm[i] -= eps
        H[i, i] = (cost_func(xp) - 2*f0 + cost_func(xm)) / (eps**2)
        for j in range(i+1, n):
            xpp = x0.copy(); xpp[i] += eps; xpp[j] += eps
            xpm = x0.copy(); xpm[i] += eps; xpm[j] -= eps
            xmp = x0.copy(); xmp[i] -= eps; xmp[j] += eps
            xmm = x0.copy(); xmm[i] -= eps; xmm[j] -= eps
            H[i, j] = (cost_func(xpp) - cost_func(xpm) - cost_func(xmp) + cost_func(xmm)) / (4 * eps**2)
            H[j, i] = H[i, j]
    return H


def weighted_stage2(weighted_pairs, fixed_station, vessel_station, fix_beta=None):
    """
    加权绝对旋转估计 (替代 estimate_absolute_rotation_stage2 的加权版本)
    
    参数:
        weighted_pairs: list of (obs_lat, obs_lon, ref_lat, ref_lon, weight)
        fixed_station, vessel_station: dict with lat, lon
        fix_beta: 若给定, 固定β为该值, 只优化α (用于单雷达模式)
    
    返回:
        dict: alpha_est_deg, beta_est_deg, sigma_alpha_deg, sigma_beta_deg, ...
    """
    if len(weighted_pairs) < 2:
        return {'alpha_est_deg': np.nan, 'beta_est_deg': np.nan,
                'sigma_alpha_deg': np.nan, 'sigma_beta_deg': np.nan,
                'rmse_m': np.nan, 'n_pairs': len(weighted_pairs)}
    
    def R(theta_deg):
        th = np.radians(theta_deg)
        return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    
    # 站址 Mercator 坐标
    fsx, fsy = latlon_to_mercator(fixed_station['lat'], fixed_station['lon'])
    vsx, vsy = latlon_to_mercator(vessel_station['lat'], vessel_station['lon'])
    S_BA = np.array([vsx - fsx, vsy - fsy])  # vessel - fixed
    
    # 转换所有点对到相对于各自站址的Mercator坐标
    p_A_list, p_B_list, weights = [], [], []
    for obs_lat, obs_lon, ref_lat, ref_lon, w in weighted_pairs:
        if np.isnan(obs_lat) or np.isnan(ref_lat):
            continue
        ox, oy = latlon_to_mercator(obs_lat, obs_lon)
        rx, ry = latlon_to_mercator(ref_lat, ref_lon)
        p_A_list.append([ox - fsx, oy - fsy])
        p_B_list.append([rx - vsx, ry - vsy])
        weights.append(w)
    
    if len(p_A_list) < 2:
        return {'alpha_est_deg': np.nan, 'beta_est_deg': np.nan,
                'rmse_m': np.nan, 'n_pairs': len(p_A_list)}
    
    p_A = np.array(p_A_list)  # (M, 2)
    p_B = np.array(p_B_list)  # (M, 2)
    W = np.array(weights)     # (M,)
    M = len(W)
    
    # 权重归一化: 避免优化器因权重动态范围过大而数值不稳定
    # 归一化不改变最优解 (α,β), 只是缩放代价函数的值
    W_sum = np.sum(W)
    if W_sum > 0:
        W = W / W_sum * M  # 归一化后平均权重为1
    
    if fix_beta is not None:
        # 单参数模式: β固定, 只优化α
        def cost(params):
            alpha = params[0]
            beta = fix_beta
            Ra = R(-alpha)
            Rb = R(-beta)
            e = (Ra @ p_A.T - Rb @ p_B.T - S_BA[:, None]).T
            sq = np.sum(e**2, axis=1)
            return float(np.sum(W * sq))
        
        init_candidates = [[0.0], [90.0], [-90.0], [45.0], [-45.0], [180.0]]
        bounds = [(-180, 180)]
    else:
        # 双参数模式: α, β 联合优化
        def cost(params):
            alpha, beta = params
            Ra = R(-alpha)
            Rb = R(-beta)
            e = (Ra @ p_A.T - Rb @ p_B.T - S_BA[:, None]).T
            sq = np.sum(e**2, axis=1)
            return float(np.sum(W * sq))
        
        init_candidates = [
            [0.0, 0.0], [0.0, 90.0], [0.0, -90.0],
            [45.0, 0.0], [-45.0, 0.0], [0.0, 180.0], [180.0, 0.0],
        ]
        bounds = [(-180, 180), (-180, 180)]
    
    best_params = None
    best_cost = float('inf')
    
    for init in init_candidates:
        try:
            res = minimize(cost, init, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 500})
            if res.fun < best_cost:
                best_cost = res.fun
                best_params = res.x
        except:
            pass
    
    if best_params is None:
        return {'alpha_est_deg': np.nan, 'beta_est_deg': np.nan,
                'rmse_m': np.nan, 'n_pairs': M}
    
    if fix_beta is not None:
        alpha_hat = wrap_angle(best_params[0])
        beta_hat = fix_beta
    else:
        alpha_hat = wrap_angle(best_params[0])
        beta_hat = wrap_angle(best_params[1])
    
    # RMSE = sqrt(加权平均残差)
    total_w = np.sum(W)
    rmse = np.sqrt(best_cost / total_w) if total_w > 0 else np.nan
    
    # ---- 不确定度: 最优解处 Hessian 逆 → Cov(α,β) ----
    n_params = 1 if fix_beta is not None else 2
    try:
        H = _numerical_hessian(cost, best_params, eps=1e-6)
        H_inv = np.linalg.inv(H)
        sigma2_res = best_cost / max(M - n_params, 1)
        cov = 2 * sigma2_res * H_inv
        sigma_alpha = np.sqrt(cov[0, 0]) if n_params >= 1 else np.nan
        sigma_beta = np.sqrt(cov[1, 1]) if n_params >= 2 else 0.0
    except np.linalg.LinAlgError:
        sigma_alpha, sigma_beta = np.nan, np.nan
    
    result = {
        'alpha_est_deg': alpha_hat,
        'beta_est_deg': beta_hat,
        'sigma_alpha_deg': sigma_alpha,
        'sigma_beta_deg': sigma_beta,
        'rel_est_deg': wrap_angle(alpha_hat - beta_hat) if not np.isnan(alpha_hat) else np.nan,
        'rmse_m': rmse,
        'n_pairs': M,
        'total_weight': float(total_w),
    }
    
    # 真值误差
    alpha_true = fixed_station.get('truth_offset_deg')
    beta_true = vessel_station.get('truth_offset_deg')
    if alpha_true is not None and not np.isnan(alpha_hat):
        result['alpha_err_deg'] = wrap_angle(alpha_hat - alpha_true)
    if beta_true is not None and not np.isnan(beta_hat):
        result['beta_err_deg'] = wrap_angle(beta_hat - beta_true)
    
    return result


def method_b_weighted(scene, w=0.1, max_iter=10, max_outer=5,
                      converge_thresh=0.05, drop_ratio=0.2):
    """
    软权重加权迭代方法 — 无匈牙利分配, 无异常剔除
    支持单雷达(ais_points/radar_points)和双雷达(fixed_points/vessel_points)模式
    """
    from scipy.optimize import linear_sum_assignment  # 确保在函数顶部导入
    
    scene_name = scene['scene']
    frames = scene['frames']
    n_targets = len(scene['mmsi_list'])
    station = scene['radar_station']
    conv_lat, conv_lon = station['lat'], station['lon']
    
    # ---- 自动检测字段名 + 单/双雷达模式 ----
    ref_field, obs_field = _detect_field_names(frames)
    is_single_radar = (ref_field == 'ais_points')  # single: AIS参考, radar观测
    
    if is_single_radar:
        truth_A = scene.get('truth_scene_offset_deg', 0)
        truth_B = 0.0  # AIS 无偏移
        fixed_station = station
        vessel_station = station
        print(f"  [Weighted B] single-radar mode, kept refs from AIS")
    else:
        truth_A = scene.get('truth_fixed_offset_deg', 0)
        truth_B = scene.get('truth_vessel_offset_deg', 0)
        fixed_station = scene.get('fixed_station', station)
        vessel_station = scene.get('radar_station', station)
    f_lat, f_lon = fixed_station['lat'], fixed_station['lon']
    v_lat, v_lon = vessel_station['lat'], vessel_station['lon']
    
    # ---- 自动检测字段名 ----
    ref_field, obs_field = _detect_field_names(frames)
    
    # ---- Drop ref ----
    ref_present = sorted({i for f in frames 
                          for i, pt in enumerate(f[ref_field]) 
                          if not np.isnan(pt[0])})
    n_drop = max(1, int(len(ref_present) * drop_ratio))
    n_keep = len(ref_present) - n_drop
    rng = np.random.RandomState(RANDOM_SEED + _stable_hash(scene_name) % 10000)
    kept_ref_original = sorted(rng.choice(ref_present, size=n_keep, replace=False))
    
    print(f"  [Weighted B] kept {n_keep}/{len(ref_present)} refs, max {max_outer} outer iters")
    
    alpha_est, beta_est = 0.0, 0.0
    history = []
    orig_to_new_ref = {orig: i for i, orig in enumerate(kept_ref_original)}
    traj_P_final = 0.0
    traj_R_final = 0.0
    stage2_res = {}
    
    for outer in range(max_outer):
        weighted_pairs = []
        iter_cum_gamma = np.zeros((n_targets, n_keep), dtype=np.float64)
        gammas_by_pair = {}
        
        for frame in frames:
            ref_pts = frame[ref_field]
            obs_pts = frame[obs_field]
            valid_ref = [i for i in range(n_targets) if not np.isnan(ref_pts[i, 0])]
            valid_obs = [i for i in range(n_targets) if not np.isnan(obs_pts[i, 0])]
            if not valid_ref or not valid_obs:
                continue
            kept_ref = [i for i in valid_ref if i in kept_ref_original]
            if not kept_ref:
                continue
            
            obs_ll = obs_pts[valid_obs, :2].astype(np.float32)
            ref_ll = np.array([ref_pts[i, :2] for i in kept_ref], dtype=np.float32)
            
            if outer == 0 or np.isnan(alpha_est):
                # 首次迭代: 不补偿, 直接用原始lat/lon转极坐标
                r_o, th_o = latlon_to_polar_batch(obs_ll[:, 0], obs_ll[:, 1], conv_lat, conv_lon)
                obs_pol = np.column_stack([r_o, th_o])
                r_r, th_r = latlon_to_polar_batch(ref_ll[:, 0], ref_ll[:, 1], conv_lat, conv_lon)
                ref_pol = np.column_stack([r_r, th_r])
            elif is_single_radar:
                # 单雷达: 只补偿雷达观测, AIS参考不补偿
                obs_lat_c, obs_lon_c = _correct_individual_radar(
                    obs_ll[:, 0], obs_ll[:, 1], f_lat, f_lon, alpha_est)
                r_o, th_o = latlon_to_polar_batch(obs_lat_c, obs_lon_c, conv_lat, conv_lon)
                obs_pol = np.column_stack([r_o, th_o])
                # AIS 不补偿
                r_r, th_r = latlon_to_polar_batch(ref_ll[:, 0], ref_ll[:, 1], conv_lat, conv_lon)
                ref_pol = np.column_stack([r_r, th_r])
            else:
                # 双雷达: 双方独立补偿后转极坐标
                obs_lat_c, obs_lon_c = _correct_individual_radar(
                    obs_ll[:, 0], obs_ll[:, 1], f_lat, f_lon, alpha_est)
                ref_lat_c, ref_lon_c = _correct_individual_radar(
                    ref_ll[:, 0], ref_ll[:, 1], v_lat, v_lon, beta_est)
                r_o, th_o = latlon_to_polar_batch(obs_lat_c, obs_lon_c, conv_lat, conv_lon)
                obs_pol = np.column_stack([r_o, th_o])
                r_r, th_r = latlon_to_polar_batch(ref_lat_c, ref_lon_c, conv_lat, conv_lon)
                ref_pol = np.column_stack([r_r, th_r])
            
            # CPD-EM
            cpd = CPDEMAssociation(w=w, max_iter=max_iter)
            cpd.fit(ref_pol, obs_pol)
            if cpd.gamma is None:
                continue
            
            gamma = cpd.gamma      # (N_obs, N_ref)
            gamma_out = cpd.gamma_outlier  # (N_obs,) 或 None
            
            # ---- 定义离群点判断函数 ----
            def is_outlier(idx):
                if gamma_out is None:
                    return False
                # gamma_out[idx] 是观测点为离群点的概率
                return gamma_out[idx] > 0.5  # 概率大于0.5视为离群点
            
            # ---- 累积gamma (用于匈牙利关联，仅输出指标) ----
            for i, oi in enumerate(valid_obs):
                if is_outlier(i):
                    continue
                for j, ri in enumerate(kept_ref):
                    if ri in orig_to_new_ref:
                        iter_cum_gamma[oi, orig_to_new_ref[ri]] += gamma[i, j]
            
            # ---- 收集加权点对: 按(oi,ri)对累积gamma，保留每帧位置 ----
            for i, oi in enumerate(valid_obs):
                if is_outlier(i):
                    continue
                obs_lat = obs_pts[oi, 0]
                obs_lon = obs_pts[oi, 1]
                if np.isnan(obs_lat):
                    continue
                for j, ri in enumerate(kept_ref):
                    w_gamma = gamma[i, j]
                    if w_gamma < WEIGHT_THRESH:
                        continue
                    ref_lat = ref_pts[ri, 0]
                    ref_lon = ref_pts[ri, 1]
                    if np.isnan(ref_lat):
                        continue
                    key = (oi, ri)
                    gammas_by_pair.setdefault(key, {}).setdefault('gamma_sum', 0.0)
                    gammas_by_pair[key]['gamma_sum'] += float(w_gamma)
                    gammas_by_pair[key].setdefault('entries', []).append(
                        (obs_lat, obs_lon, ref_lat, ref_lon))
        
        # ---- 用Γ_cum替换逐帧gamma作为M-step权重 ----
        # Γ_cum(oi,ri) = (1/T) Σ_t γ_t(oi,ri), T=总帧数
        # 同一对在不同帧位置不同, 保留所有帧条目但权重统一为Γ_cum
        total_frames = len(frames)
        weighted_pairs = []
        for key, data in gammas_by_pair.items():
            Gamma_cum = data['gamma_sum'] / total_frames
            if Gamma_cum < 1e-6:
                continue
            for ol, on, rl, rn in data['entries']:
                weighted_pairs.append((ol, on, rl, rn, Gamma_cum))
        
        # 如果加权点对太少，退出并返回默认结果（或跳过该场景）
        if len(weighted_pairs) < 2:
            print(f"  [Weighted B-{outer}] Too few weighted pairs ({len(weighted_pairs)}), stopping")
            # 直接使用当前 alpha_est, beta_est (可能为0) 跳出循环
            break
        
        # ---- 匈牙利硬关联（仅输出指标，不参与参数更新） ----
        row_ind, col_ind = linear_sum_assignment(-iter_cum_gamma)
        iter_assign = {r: kept_ref_original[c] for r, c in zip(row_ind, col_ind)}
        observed_targets = set()
        for f in frames:
            for i in range(n_targets):
                if not np.isnan(f[obs_field][i, 0]):
                    observed_targets.add(i)
        valid_r_true = [r for r in observed_targets if r in kept_ref_original]
        iter_correct = sum(1 for r in valid_r_true if iter_assign.get(r) == r)
        iter_traj_P = iter_correct / len(row_ind) if len(row_ind) else 0
        iter_traj_R = iter_correct / len(valid_r_true) if valid_r_true else 0
        traj_P_final = iter_traj_P
        traj_R_final = iter_traj_R
        
        # ---- 加权Stage2优化 ----
        stage2_res = weighted_stage2(weighted_pairs, fixed_station, vessel_station,
                                      fix_beta=0.0 if is_single_radar else None)
        new_a = stage2_res.get('alpha_est_deg', np.nan)
        new_b = stage2_res.get('beta_est_deg', np.nan)
        rmse = stage2_res.get('rmse_m', np.nan)
        
        if np.isnan(new_a) or np.isnan(new_b):
            print(f"  [Weighted B-{outer}] Stage2 FAILED, stopping")
            break
        
        change_a = abs(wrap_angle(new_a - alpha_est)) if outer > 0 else float('inf')
        change_b = abs(wrap_angle(new_b - beta_est)) if outer > 0 else float('inf')
        
        history.append({
            'iter': outer, 'alpha': new_a, 'beta': new_b,
            'sigma_alpha': stage2_res.get('sigma_alpha_deg', np.nan),
            'sigma_beta': stage2_res.get('sigma_beta_deg', np.nan),
            'rmse': rmse, 'n_pairs': stage2_res.get('n_pairs', 0),
            'total_weight': stage2_res.get('total_weight', 0),
            'traj_P': iter_traj_P,
        })
        alpha_est, beta_est = new_a, new_b
        
        sig_a = stage2_res.get('sigma_alpha_deg', np.nan)
        sig_b = stage2_res.get('sigma_beta_deg', np.nan)
        sig_str = f", σα={sig_a:.4f}°, σβ={sig_b:.4f}°" if (not np.isnan(sig_a) and not np.isnan(sig_b)) else ""
        print(f"  [Weighted B-{outer}] α={alpha_est:.4f}°, β={beta_est:.4f}°{sig_str}, "
              f"Δα={change_a:.4f}°, RMSE={rmse:.1f}m, N={len(weighted_pairs)}, "
              f"TrajP={iter_traj_P*100:.1f}%")
        
        if outer > 0 and max(change_a, change_b) < converge_thresh:
            print(f"  [Weighted B] Converged at iter {outer}: α={alpha_est:.4f}°, β={beta_est:.4f}°")
            break
    
    # 从最后一次 weighted_stage2 获取 σ_α, σ_β
    sigma_a_deg = stage2_res.get('sigma_alpha_deg', np.nan)
    sigma_b_deg = stage2_res.get('sigma_beta_deg', np.nan)
    extra_theta_var = (sigma_a_deg**2 + sigma_b_deg**2) if (not np.isnan(sigma_a_deg) and not np.isnan(sigma_b_deg)) else 0.0
    
    # ===== 用最终 α_est, β_est 跑一次 CPD-EM 输出最终关联精度 =====
    final_cum_gamma = np.zeros((n_targets, n_keep), dtype=np.float64)
    frame_precs, frame_recs, frame_fars = [], [], []
    for frame in frames:
        ref_pts = frame[ref_field]
        obs_pts = frame[obs_field]
        valid_ref = [i for i in range(n_targets) if not np.isnan(ref_pts[i, 0])]
        valid_obs = [i for i in range(n_targets) if not np.isnan(obs_pts[i, 0])]
        if not valid_ref or not valid_obs:
            continue
        kept_ref = [i for i in valid_ref if i in kept_ref_original]
        if not kept_ref:
            continue
        
        obs_ll = obs_pts[valid_obs, :2].astype(np.float32)
        ref_ll = np.array([ref_pts[i, :2] for i in kept_ref], dtype=np.float32)
        
        if outer > 0 and not np.isnan(alpha_est):
            if is_single_radar:
                # 单雷达: 只补偿 obs, ref(AIS) 不补偿
                obs_lat_c, obs_lon_c = _correct_individual_radar(
                    obs_ll[:, 0], obs_ll[:, 1], f_lat, f_lon, alpha_est)
                r_o, th_o = latlon_to_polar_batch(obs_lat_c, obs_lon_c, conv_lat, conv_lon)
                obs_pol = np.column_stack([r_o, th_o])
                r_r, th_r = latlon_to_polar_batch(ref_ll[:, 0], ref_ll[:, 1], conv_lat, conv_lon)
                ref_pol = np.column_stack([r_r, th_r])
            else:
                obs_lat_c, obs_lon_c = _correct_individual_radar(
                    obs_ll[:, 0], obs_ll[:, 1], f_lat, f_lon, alpha_est)
                ref_lat_c, ref_lon_c = _correct_individual_radar(
                    ref_ll[:, 0], ref_ll[:, 1], v_lat, v_lon, beta_est)
                r_o, th_o = latlon_to_polar_batch(obs_lat_c, obs_lon_c, conv_lat, conv_lon)
                obs_pol = np.column_stack([r_o, th_o])
                r_r, th_r = latlon_to_polar_batch(ref_lat_c, ref_lon_c, conv_lat, conv_lon)
                ref_pol = np.column_stack([r_r, th_r])
        else:
            r_o, th_o = latlon_to_polar_batch(obs_ll[:, 0], obs_ll[:, 1], conv_lat, conv_lon)
            obs_pol = np.column_stack([r_o, th_o])
            r_r, th_r = latlon_to_polar_batch(ref_ll[:, 0], ref_ll[:, 1], conv_lat, conv_lon)
            ref_pol = np.column_stack([r_r, th_r])
        
        cpd = CPDEMAssociation(w=w, max_iter=max_iter)
        cpd.fit(ref_pol, obs_pol)
        if cpd.gamma is None:
            continue
        if extra_theta_var > 0:
            cpd.recompute_gamma(extra_theta_var)
        gamma, g_out = cpd.gamma, cpd.gamma_outlier
        
        def is_outlier_final(idx):
            if g_out is None:
                return False
            return g_out[idx] > 0.5
        
        # ---- 帧级关联指标 ----
        assocs = cpd.hungarian_association()
        fp_tp = sum(1 for o_pos, r_pos, _ in assocs
                    if valid_obs[o_pos] == kept_ref[r_pos])
        n_assoc = len(assocs)
        n_possible = len(set(valid_obs) & set(kept_ref_original))
        f_prec = fp_tp / n_assoc if n_assoc > 0 else 0
        f_rec  = fp_tp / n_possible if n_possible > 0 else 0
        frame_precs.append(f_prec)
        frame_recs.append(f_rec)
        frame_fars.append(1 - f_prec)
        
        for i, oi in enumerate(valid_obs):
            if is_outlier_final(i):
                continue
            for j, ri in enumerate(kept_ref):
                if ri in orig_to_new_ref:
                    final_cum_gamma[oi, orig_to_new_ref[ri]] += gamma[i, j]
    
    row_ind, col_ind = linear_sum_assignment(-final_cum_gamma)
    final_assign = {r: kept_ref_original[c] for r, c in zip(row_ind, col_ind)}
    observed_targets = set()
    for f in frames:
        for i in range(n_targets):
            if not np.isnan(f[obs_field][i, 0]):
                observed_targets.add(i)
    valid_r_true = [r for r in observed_targets if r in kept_ref_original]
    final_correct = sum(1 for r in valid_r_true if final_assign.get(r) == r)
    traj_P_final = final_correct / len(row_ind) if len(row_ind) else 0
    traj_R_final = final_correct / len(valid_r_true) if valid_r_true else 0
    
    alpha_err = wrap_angle(alpha_est - truth_A) if not np.isnan(alpha_est) else np.nan
    beta_err = wrap_angle(beta_est - truth_B) if not np.isnan(beta_est) else np.nan
    
    return {
        'method': 'B_weighted',
        'scene': scene_name,
        'alpha_est': alpha_est, 'beta_est': beta_est,
        'sigma_alpha_deg': sigma_a_deg, 'sigma_beta_deg': sigma_b_deg,
        'alpha_err': alpha_err, 'beta_err': beta_err,
        'traj_P': traj_P_final,
        'traj_R': traj_R_final,
        'alpha_true': truth_A,
        'beta_true': truth_B,
        'n_iterations': len(history),
        'iter_history': history,
        'rmse': history[-1]['rmse'] if history else np.nan,
        'assign': final_assign,
        'frame_prec_mean': float(np.mean(frame_precs)) if frame_precs else 0,
        'frame_rec_mean': float(np.mean(frame_recs)) if frame_recs else 0,
        'frame_far_mean': float(np.mean(frame_fars)) if frame_fars else 0,
    }


# ==============================================================================
# 测试主函数
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='方案B: 软权重加权参数估计测试')
    parser.add_argument('--dataset_meta', type=str, required=True)
    parser.add_argument('--dataset_chunk', type=str, required=True)
    parser.add_argument('--n_scenes', type=int, default=200)
    args = parser.parse_args()
    
    scenes = load_cpd_dataset(args.dataset_meta, args.dataset_chunk)
    scenes = scenes[:args.n_scenes]
    
    base_errs_a, base_errs_b = [], []
    bw_errs_a, bw_errs_b = [], []
    base_tp, base_tr = [], []
    bw_tp, bw_tr = [], []
    
    for idx, scene in enumerate(scenes):
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(scenes)}] {scene['scene']}")
        print(f"{'='*60}")
        
        # Baseline
        print("  [Baseline O]")
        r_o = baseline_evaluate(scene, w=0.1, max_iter=10)
        base_errs_a.append(r_o.get('stage2_alpha_err_deg', np.nan))
        base_errs_b.append(r_o.get('stage2_beta_err_deg', np.nan))
        base_tp.append(r_o.get('trajectory_precision', 0))
        base_tr.append(r_o.get('trajectory_recall', 0))
        
        # Weighted B
        print("  [Weighted B]")
        r_bw = method_b_weighted(scene, w=0.1, max_iter=10, max_outer=5)
        bw_errs_a.append(r_bw.get('alpha_err', np.nan))
        bw_errs_b.append(r_bw.get('beta_err', np.nan))
        bw_tp.append(r_bw.get('traj_P', 0))
        bw_tr.append(r_bw.get('traj_R', 0))
    
    # 汇总
    print("\n" + "=" * 80)
    print("WEIGHTED METHOD B - COMPARISON SUMMARY")
    print("=" * 80)
    print(f"\n{'Method':<20} {'alpha MAE(deg)':<18} {'beta MAE(deg)':<18} {'Traj P(%)':<12} {'Traj R(%)':<12}")
    print("-" * 80)
    
    def safe_mae(vals):
        v = [x for x in vals if not (isinstance(x, float) and np.isnan(x))]
        return np.mean(np.abs(v)) if v else 0, np.std(np.abs(v)) if v else 0
    
    def safe_mean(vals):
        v = [x for x in vals if not np.isnan(x)]
        return np.mean(v) * 100 if v else 0
    
    for name, ea, eb, tp, tr in [
        ('O (Baseline)', base_errs_a, base_errs_b, base_tp, base_tr),
        ('B (Weighted)', bw_errs_a, bw_errs_b, bw_tp, bw_tr),
    ]:
        am, as_ = safe_mae(ea)
        bm, bs = safe_mae(eb)
        tpm = safe_mean(tp)
        trm = safe_mean(tr)
        print(f"{name:<20} {am:.4f} +/- {as_:.4f}  {bm:.4f} +/- {bs:.4f}  {tpm:.1f}         {trm:.1f}")
    
    # 保存结果
    summary = {
        'n_scenes': len(scenes),
        'baseline': {
            'alpha_mae': float(safe_mae(base_errs_a)[0]),
            'beta_mae': float(safe_mae(base_errs_b)[0]),
            'traj_P': float(np.mean([x for x in base_tp if not np.isnan(x)])),
            'traj_R': float(np.mean([x for x in base_tr if not np.isnan(x)])),
        },
        'b_weighted': {
            'alpha_mae': float(safe_mae(bw_errs_a)[0]),
            'beta_mae': float(safe_mae(bw_errs_b)[0]),
            'traj_P': float(np.mean([x for x in bw_tp if not np.isnan(x)])),
            'traj_R': float(np.mean([x for x in bw_tr if not np.isnan(x)])),
        },
    }
    os.makedirs('results_weighted_b', exist_ok=True)
    with open('results_weighted_b/summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: results_weighted_b/summary.json")


if __name__ == "__main__":
    main()