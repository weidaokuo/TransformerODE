
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import al_ode_train as OT
import al_common as C


def zscore_latent(idx, latent_rows, zstats):
    lc = latent_rows[idx]['latent']
    z = np.zeros_like(lc)
    for d, q in enumerate(OT.QUANT_NAMES):
        z[:, d] = (lc[:, d] - zstats[q]['mean']) / zstats[q]['std']
    return z


def predict_engineering_curve(model, idx, latent_rows, zstats, n_samples=20):

    info = latent_rows[idx]
    cond = info['cond21']
    strain_pct = info['strain_pct']
    if n_samples == 1:
        z = OT.predict_latent(model, cond, strain_pct, n_samples=1)
        sigma_phys = z[:, 3] * zstats['sigma']['std'] + zstats['sigma']['mean']
    else:
        z_mean, z_std = OT.predict_latent(model, cond, strain_pct, n_samples=n_samples)
        sigma_phys = z_mean[:, 3] * zstats['sigma']['std'] + zstats['sigma']['mean']
        sigma_std = z_std[:, 3] * zstats['sigma']['std']
    eps_true = strain_pct / 100.0
    eps_eng = np.exp(eps_true) - 1.0
    sigma_eng = sigma_phys * np.exp(-eps_true)
    curve = np.column_stack([eps_eng * 100, sigma_eng])
    if n_samples == 1:
        return curve
    else:
        sigma_eng_std = sigma_std * np.exp(-eps_true)
        return curve, np.column_stack([eps_eng * 100, sigma_eng_std])



ENG_MAX_PCT = 30.0   
SHIELD_PCT = 20.0    
N_GRID = 150         

def predict_full(model, idx, latent_rows, zstats, n_samples, max_eng_pct=ENG_MAX_PCT):
    info = latent_rows[idx]
    cond = info['cond21']
    eps_true_max = float(np.log(1.0 + max_eng_pct / 100.0))
    eps_true = np.linspace(0, eps_true_max, N_GRID)
    strain_true_pct = eps_true * 100.0
    if n_samples == 1:
        z = OT.predict_latent(model, cond, strain_true_pct, n_samples=1)
        z_mean, z_std = z, np.zeros_like(z)
    else:
        z_mean, z_std = OT.predict_latent(model, cond, strain_true_pct, n_samples=n_samples)
    # QUANT_NAMES=[rho,theta,Lambda,sigma] → theta=1, sigma=3
    sig_mean = z_mean[:, 3] * zstats['sigma']['std'] + zstats['sigma']['mean']
    sig_std = z_std[:, 3] * zstats['sigma']['std']
    th_mean = z_mean[:, 1] * zstats['theta']['std'] + zstats['theta']['mean']
    th_std = z_std[:, 1] * zstats['theta']['std']
    # 工程转换
    eps_eng = np.exp(eps_true) - 1.0
    sig_eng = sig_mean * np.exp(-eps_true)
    sig_eng_std = sig_std * np.exp(-eps_true)
    return {
        'idx': idx, 'eps_true': eps_true,
        'sig_true_mean': sig_mean, 'sig_true_std': sig_std,
        'theta_mean': th_mean, 'theta_std': th_std,
        'eps_eng': eps_eng, 'sig_eng': sig_eng, 'sig_eng_std': sig_eng_std,
    }

def uts_of_pred(pred):
    sig = pred['sig_eng']
    i = int(np.argmax(sig))
    eps_uts = pred['eps_eng'][i] * 100.0
    at_end = i >= len(sig) - 1
    return eps_uts, float(sig[i]), (not at_end)
