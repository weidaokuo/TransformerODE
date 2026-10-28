import os, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_STRAIN = 33.5  

PHYSIC_NAMES = None
ELEMENTS = ["Nb", "W", "Mo", "Zr", "C", "Hf", "Ti", "Ta", "Si", "N", "V", "Sc", "Ru", "B", "Y"]
PROCESS_5 = [1450.0, 1.0, 0.0, 25.0, 0.0]  

def load_data():
    sdf = pd.read_csv(os.path.join(HERE, 'alloy_summary.csv')).set_index('idx')
    curves = {}
    for idx in sdf.index:
        c = pd.read_csv(os.path.join(HERE, 'curves', f'alloy_{idx:03d}.csv'))
        curves[idx] = c[['strain_eng_pct', 'stress_eng_MPa']].values
    return sdf, curves

def get_physic_names():
    global PHYSIC_NAMES
    if PHYSIC_NAMES is None:
        sys_path = os.path.join(os.path.dirname(HERE), 'Args.py')
        import importlib.util
        spec = importlib.util.spec_from_file_location('args_mod', sys_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        PHYSIC_NAMES = list(mod.args.physic_names)
    return PHYSIC_NAMES

def true_pse_of(idx):
    sdf, _ = load_data()
    return float(sdf.loc[idx, 'true_pse'])

GRID_STEP = 0.2  
def fit_normalizers(train_idx, sdf, curves, target_strain_max=20.0, grid_step=GRID_STEP):

    desc_cols = get_physic_names()
    # z-score
    feats = sdf.loc[train_idx, desc_cols].values.astype(float)
    feat_mean = feats.mean(0)
    feat_std = feats.std(0)
    feat_std[feat_std < 1e-12] = 1.0
    grid = np.arange(0, target_strain_max + 1e-9, grid_step)
    stress_all = []
    for idx in train_idx:
        curve = curves[idx]  
        stress_interp = np.interp(grid, curve[:, 0], curve[:, 1])
        stress_all.append(stress_interp)
    stress_all = np.array(stress_all)
    stress_mean = stress_all.mean()
    stress_std = stress_all.std()
    if stress_std < 1e-12:
        stress_std = 1.0
    return {'feat_mean': feat_mean, 'feat_std': feat_std,
            'stress_mean': stress_mean, 'stress_std': stress_std,
            'grid': grid}

def build_cond18(desc_row, feat_mean, feat_std):
    desc = np.array([desc_row[n] for n in get_physic_names()], dtype=float)
    raw = np.zeros(18)
    raw[:5] = PROCESS_5
    raw[5:] = desc
    return ((raw - np.concatenate([np.zeros(5), feat_mean])) /
            np.concatenate([np.ones(5), feat_std])).astype(np.float32)

def normalize_stress(stress, stress_mean, stress_std):
    return (stress - stress_mean) / stress_std

def denormalize_stress(stress_norm, stress_mean, stress_std):
    return stress_norm * stress_std + stress_mean


def extract_props_from_curve(curve):
    eps = curve[:, 0] / 100.0  
    sig = curve[:, 1]
    i_uts = int(np.argmax(sig))
    uts = float(sig[i_uts])
    eps_uts = float(eps[i_uts])

    m_el = eps <= 0.005
    if m_el.sum() >= 2:
        E = np.polyfit(eps[m_el], sig[m_el], 1)[0]
        offset = E * (eps - 0.002)
        diff = sig - offset
        idx = np.where((eps > 0.002) & (diff > 0))[0]
        ys = float(sig[idx[0]]) if len(idx) > 0 else float(sig[1])
    else:
        ys = float(sig[1])
    post = np.where(eps > eps_uts + 1e-6)[0]
    if len(post) > 0:
        drop = np.where(sig[post] <= 0.8 * uts)[0]
        elong = float(eps[post[drop[0]]]) if len(drop) > 0 else float(eps[-1])
    else:
        elong = float(eps[-1])
    return ys, uts, elong

def pse_of_curve(curve):
    ys, uts, elong = extract_props_from_curve(curve)
    return uts * elong

def fast_non_dominated_sort(F):
    N = F.shape[0]
    S = [[] for _ in range(N)]
    n = np.zeros(N)
    fronts = [[]]
    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                S[p].append(q)
            elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                n[p] += 1
        if n[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        Q = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    Q.append(q)
        i += 1
        fronts.append(Q)
    return [f for f in fronts if f]

def crowding_distance(F, front_idx):
    m = F.shape[1]
    d = np.zeros(len(front_idx))
    for obj in range(m):
        idx_sorted = sorted(front_idx, key=lambda i: F[i, obj])
        d[0] = d[-1] = np.inf
        fmin = F[idx_sorted[0], obj]
        fmax = F[idx_sorted[-1], obj]
        denom = fmax - fmin
        if denom > 1e-12:
            for k in range(1, len(idx_sorted) - 1):
                d[k] += (F[idx_sorted[k + 1], obj] - F[idx_sorted[k - 1], obj]) / denom
    return dict(zip(front_idx, d))

def select_nsga3(utilities, top_k=2, seed=42):
    rng = np.random.RandomState(seed)
    F = -utilities  
    fronts = fast_non_dominated_sort(F)
    selected = []
    for front in fronts:
        if len(selected) >= top_k:
            break
        need = top_k - len(selected)
        if len(front) <= need:
            
            selected.extend(front)
        else:
            
            cd = crowding_distance(F, front)
            ranked = sorted(front, key=lambda i: -cd[i])
            selected.extend(ranked[:need])
    return selected[:top_k]

def integral_of_curve(curve):
    eps = curve[:, 0] / 100.0
    sig = curve[:, 1]
    return float(np.trapezoid(sig, eps))

def select_integral_improvement(pred_curves, train_best_curve, top_k=2):

    train_best_integral = integral_of_curve(train_best_curve)
    improvements = {}
    for idx, curve in pred_curves.items():
        imp = integral_of_curve(curve) - train_best_integral
        improvements[idx] = imp
    ranked = sorted(improvements.items(), key=lambda x: -x[1])
    return [i for i, _ in ranked[:top_k]], improvements

def cpei_varlen(pred_curves, fstar_curve, xi=0.0):

    from math import erf, sqrt, pi, exp
    Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))
    phi = lambda z: exp(-z * z / 2) / sqrt(2 * pi)

    f_eps = fstar_curve[:, 0] / 100.0
    f_sig = fstar_curve[:, 1]

    result = {}
    for idx, (curve, cstd) in pred_curves.items():
        eps = curve[:, 0] / 100.0
        sig = curve[:, 1]
        std = cstd[:, 1]

        f_s = np.interp(eps, f_eps, f_sig, left=0.0, right=0.0)
        s = np.maximum(std, 1e-9)
        diff = sig - f_s - xi   
        z = diff / s
        EI = np.maximum(diff * np.array([Phi(zz) for zz in z])
                        + s * np.array([phi(zz) for zz in z]), 0)
        result[idx] = float(np.trapezoid(EI, eps))  
    return result


def load_split():
    init = json.load(open(os.path.join(HERE, 'initial_train_indices.json')))
    cands = json.load(open(os.path.join(HERE, 'candidate_indices.json')))['candidates']
    return init['initial_train'], cands

def make_train_jsonl(train_idx, sdf, curves, normalizers, out_path):
    desc_cols = get_physic_names()
    grid = normalizers['grid']
    rows = []
    for idx in train_idx:
        curve = curves[idx]
        stress_interp = np.interp(grid, curve[:, 0], curve[:, 1])
        stress_norm = normalize_stress(stress_interp, normalizers['stress_mean'],
                                       normalizers['stress_std'])
        cond = build_cond18(sdf.loc[idx], normalizers['feat_mean'], normalizers['feat_std'])
        rows.append({
            'Alloy_Num': int(idx), 'initial_feature': cond.tolist(),
            'strain': grid.tolist(), 'stress': stress_norm.tolist(),
            'strain_pct': grid.tolist(), 'stress_mpa': stress_interp.tolist(),
        })
    with open(out_path, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    return len(rows)

def f_samples_from_raw(z_raw, eps_true, zstats, lam=1.0):
    sig_m = z_raw[:, :, 3] * zstats['sigma']['std'] + zstats['sigma']['mean']   # [n,T]
    th_m = z_raw[:, :, 1] * zstats['theta']['std'] + zstats['theta']['mean']
    f_m = sig_m + lam * th_m * eps_true[None, :]
    return f_m.mean(0), f_m.std(0)

def predict_full_mc(model, idx, latent_rows, zstats, n_samples=50, lam=1.0,
                    max_eng_pct=30.0, seed=None):

    import al_ode_train as OT2
    info = latent_rows[idx]
    eps_true_max = float(np.log(1.0 + max_eng_pct / 100.0))
    eps_true = np.linspace(0, eps_true_max, 150)
    strain_true_pct = eps_true * 100.0
    z_raw = OT2.predict_latent_raw(model, info['cond21'], strain_true_pct,
                                   n_samples=n_samples, seed=seed)
    mu_f, std_f = f_samples_from_raw(z_raw, eps_true, zstats, lam)
    sig_mean = z_raw[:, :, 3].mean(0) * zstats['sigma']['std'] + zstats['sigma']['mean']
    th_mean = z_raw[:, :, 1].mean(0) * zstats['theta']['std'] + zstats['theta']['mean']
    sig_eng = sig_mean * np.exp(-eps_true)
    eps_eng = np.exp(eps_true) - 1.0
    return {'eps_true': eps_true, 'mu_f': mu_f, 'std_f': std_f,
            'sig_true_mean': sig_mean, 'theta_mean': th_mean,
            'eps_eng': eps_eng, 'sig_eng': sig_eng}

def predict_det_full(model, idx, latent_rows, zstats, lam=1.0, max_eng_pct=30.0):
    import al_ode_train as OT2
    info = latent_rows[idx]
    eps_true_max = float(np.log(1.0 + max_eng_pct / 100.0))
    eps_true = np.linspace(0, eps_true_max, 150)
    strain_true_pct = eps_true * 100.0
    z = OT2.predict_det(model, info['cond21'], strain_true_pct)
    sig = z[:, 3] * zstats['sigma']['std'] + zstats['sigma']['mean']
    th = z[:, 1] * zstats['theta']['std'] + zstats['theta']['mean']
    sig_eng = sig * np.exp(-eps_true)
    eps_eng = np.exp(eps_true) - 1.0
    mu_f = sig + lam * th * eps_true
    return {'eps_true': eps_true, 'mu_f': mu_f, 'std_f': np.zeros_like(mu_f),
            'sig_true_mean': sig, 'theta_mean': th,
            'eps_eng': eps_eng, 'sig_eng': sig_eng}


def _integration_end(pred, max_eng_pct=30.0):
    sig = pred['sig_eng']
    i = int(np.argmax(sig))
    n = len(sig)
    if 0 < i < n - 1:                       
        eu = pred['eps_eng'][i] * 100.0
        return float(np.log(1.0 + eu / 100.0)), 'normal'
    if i == n - 1:                         
        return float(np.log(1.0 + max_eng_pct / 100.0)), 'rising'
    return float(np.log(1.0 + max_eng_pct / 100.0)), 'invalid'   

def cpei_mc(preds, fstar_pred, shield_pct=20.0, max_eng_pct=30.0):
    from scipy.stats import norm as _norm
    eps = fstar_pred['eps_true']
    fstar_f = fstar_pred['mu_f']
    result, meta = {}, {}
    for idx, pred in preds.items():
        sig = pred['sig_eng']
        i = int(np.argmax(sig))
        end_t, peak = _integration_end(pred, max_eng_pct)
        meta[idx] = {'eps_uts': float(pred['eps_eng'][i] * 100.0),
                     'uts': float(sig[i]), 'peak': peak,
                     'invalid_peak': peak == 'invalid'}
        cut = min(int(np.searchsorted(eps, end_t + 1e-9)) + 1, len(eps))
        f = pred['mu_f']; s = np.maximum(pred['std_f'], 1e-9)
        diff = f - fstar_f
        z = diff / s
        EI = np.maximum(diff * _norm.cdf(z) + s * _norm.pdf(z), 0)
        result[idx] = float(np.trapezoid(EI[:cut], eps[:cut]))
    return result, meta

def det_score_mc(pred, fstar_pred, max_eng_pct=30.0):
    eps = pred['eps_true']
    sig = pred['sig_eng']
    i = int(np.argmax(sig))
    end_t, peak = _integration_end(pred, max_eng_pct)
    cut = min(int(np.searchsorted(eps, end_t + 1e-9)) + 1, len(eps))
    return float(np.trapezoid(pred['mu_f'][:cut] - fstar_pred['mu_f'][:cut], eps[:cut])), peak


def f_curve(pred, lam=1.0):
    return pred['sig_true_mean'] + lam * pred['theta_mean'] * pred['eps_true']

def f_std(pred, lam=1.0):
    """s_f = sqrt(s_σ² + (λ·ε)²·s_θ²)"""
    return np.sqrt(pred['sig_true_std']**2 + (lam * pred['eps_true'])**2 * pred['theta_std']**2)

def cpei_uts_v2(preds, fstar_pred, lam=1.0, shield_pct=20.0, max_eng_pct=30.0):
    from math import erf, sqrt, pi, exp
    Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))
    phi = lambda z: exp(-z * z / 2) / sqrt(2 * pi)

    eps = fstar_pred['eps_true']
    fstar_f = f_curve(fstar_pred, lam)

    result, meta = {}, {}
    for idx, pred in preds.items():
        eps_uts, uts, has_peak = _uts_peak(pred)
        meta[idx] = {'eps_uts': eps_uts, 'uts': uts, 'has_peak': has_peak}
        if eps_uts <= shield_pct:
            result[idx] = -1e18
            meta[idx]['shielded'] = True
            continue
        meta[idx]['shielded'] = False
        if has_peak:
            eps_uts_true = float(np.log(1.0 + eps_uts / 100.0))
        else:
            eps_uts_true = float(np.log(1.0 + max_eng_pct / 100.0))
        cut = int(np.searchsorted(eps, eps_uts_true + 1e-9)) + 1
        cut = min(cut, len(eps))
        f = f_curve(pred, lam)
        s = np.maximum(f_std(pred, lam), 1e-9)
        diff = f - fstar_f
        z = diff / s
        EI = np.maximum(diff * np.array([Phi(zz) for zz in z])
                        + s * np.array([phi(zz) for zz in z]), 0)
        result[idx] = float(np.trapezoid(EI[:cut], eps[:cut]))
    return result, meta

def _uts_peak(pred):
    sig = pred['sig_eng']
    i = int(np.argmax(sig))
    eps_uts = pred['eps_eng'][i] * 100.0
    has_peak = i < len(sig) - 1
    return eps_uts, float(sig[i]), has_peak
