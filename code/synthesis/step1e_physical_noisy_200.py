import os, sys, json, pickle
import numpy as np
import pandas as pd

WDK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'data200', 'curves')
TRUE_DIR = os.path.join(HERE, 'data200', 'true_curves')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TRUE_DIR, exist_ok=True)

SEED = 20260830
rng = np.random.RandomState(SEED)

SIGMA_0 = 180.0
K_SOLID = {
    'W':  38.0, 'Mo': 25.0, 'Ta': 21.0, 'Hf': 19.0, 'Zr': 12.0, 'V': 10.0,
    'Ti': 4.0, 'Ru': 17.0, 'Si': 8.0, 'Sc': 5.0, 'Y': 3.0, 'B': 7.0,
    'C':  48.0, 'N': 45.0,
}

OPT_CONTENT = {
    'W': 10.0, 'Mo': 4.0, 'Ta': 12.0, 'Hf': 5.0, 'Zr': 0.8, 'V': 1.5,
    'Ti': 1.5, 'Ru': 1.0, 'Si': 0.15, 'Sc': 0.25, 'Y': 0.15, 'B': 0.05,
    'C': 0.15, 'N': 0.1,
}
ELEMENTS = ["Nb", "W", "Mo", "Zr", "C", "Hf", "Ti", "Ta", "Si", "N", "V", "Sc", "Ru", "B", "Y"]
RANGES = {
    'W': (4, 16), 'Mo': (0, 8), 'Zr': (0, 1.5), 'C': (0, 0.3),
    'Hf': (2, 10), 'Ti': (0, 3), 'Ta': (5, 20), 'Si': (0, 0.3),
    'N': (0, 0.2), 'V': (0, 3), 'Sc': (0, 0.5), 'Ru': (0, 2),
    'B': (0, 0.1), 'Y': (0, 0.3),
}
KEY_ELEMS = ['W', 'Mo', 'Ta', 'Hf']
PROCESS_8 = [0.0, 0.0, 0.0, 1450.0, 1.0, 0.0, 0.0, 0.0]

NOISE_SY = float(sys.argv[1]) if len(sys.argv) > 1 else 0.02
NOISE_SS = NOISE_SY
NOISE_ELONG = NOISE_SY * 1.5
HIDDEN_STD = 0.06
print(f"噪声水平: σ_y/σ_s ±{NOISE_SY*100:.1f}%, elong ±{NOISE_ELONG*100:.1f}%")

def sample_composition():
    while True:
        comp = {}
        for e, (lo, hi) in RANGES.items():
            comp[e] = round(rng.uniform(lo, hi), 3)
        total_alloy = sum(comp.values())
        comp['Nb'] = round(100.0 - total_alloy, 3)
        n_key = sum(1 for e in KEY_ELEMS if comp[e] > 1.0)
        if comp['Nb'] >= 58.0 and n_key >= 2:
            return comp

def gaussian_peaks(w, ta, mo, sig2=12.0):
    """4 个分离高斯峰 (配方) 的激活值: 峰A高W低Ta, B低W高Ta, C高Mo, D中W中Ta"""
    pA = np.exp(-((w-14.0)**2 + (ta-7.0)**2 + (mo-3.0)**2) / sig2)
    pB = np.exp(-((w-5.0)**2 + (ta-18.0)**2 + (mo-3.0)**2) / sig2)
    pC = np.exp(-((w-9.0)**2 + (ta-10.0)**2 + (mo-7.0)**2) / sig2)
    pD = np.exp(-((w-11.0)**2 + (ta-13.0)**2 + (mo-1.0)**2) / sig2)
    return pA, pB, pC, pD

def compute_sigma_y(comp):
    sig_y = SIGMA_0
    for e, c in comp.items():
        if e == 'Nb' or c <= 0:
            continue
        sig_y += K_SOLID.get(e, 0.0) * (c ** (2.0/3.0))
    return sig_y

def compute_sigma_s_and_elong(comp, sigma_y):
    w = comp.get('W', 0); ta = comp.get('Ta', 0)
    delta = sigma_y * 0.35
    sigma_s = sigma_y + delta
    elong = 0.285 - 0.015 * abs(ta - 13.0)
    elong = np.clip(elong, 0.02, 0.40)
    return sigma_s, elong


def voce_engineering(sigma_y, sigma_s, elong, n=200):
    eps_c = rng.uniform(0.05, 0.12)
    eps_true = np.linspace(0, 0.35, n)
    sigma_true = sigma_s - (sigma_s - sigma_y) * np.exp(-eps_true / eps_c)
    eps_eng = np.exp(eps_true) - 1.0
    sigma_eng = sigma_true * np.exp(-eps_true)
    cut_idx = max(int(np.searchsorted(eps_eng, elong)), 5)
    return eps_true[:cut_idx], sigma_true[:cut_idx], eps_eng[:cut_idx], sigma_eng[:cut_idx], eps_c

ALPHA_TAYLOR = 0.3
BURGERS_B = 2.86e-10
SIGMA_I = 0.0
MU_SHEAR = 40e9
AB = ALPHA_TAYLOR * MU_SHEAR * BURGERS_B

def latent_from_voce(eps, sy, ss, ec):
    sigma = ss - (ss - sy) * np.exp(-eps / ec)
    theta = (ss - sy) / ec * np.exp(-eps / ec)
    sqrt_rho = np.maximum((sigma - SIGMA_I) * 1e6 / AB, 1e-6)
    rho = sqrt_rho ** 2
    drho = 2.0 * sqrt_rho * (theta * 1e6) / AB
    Lambda = 1.0 / (BURGERS_B * np.maximum(drho, 1e-6))
    return rho, theta, Lambda, sigma


sys.path.insert(0, WDK_ROOT)
from obtain_feature import ObtainFeatures
from Args import args
pkl = pickle.load(open(os.path.join(WDK_ROOT, 'voce_km_latent', 'results', 'ml_final_two_tier.pkl'), 'rb'))
scaler = pkl['tier2_voce29']['scaler']
PHYSIC_NAMES = list(args.physic_names)

def desc13(comp):
    el = [e for e in ELEMENTS if comp.get(e, 0) > 1e-8]
    cl = [comp[e] for e in el]
    ext = ObtainFeatures(el, cl)
    key = ext.get_key_features()
    return {n: float(key.get(n, 0.0)) for n in PHYSIC_NAMES}

print("生成200条 (物理模型 + 实验噪声)...")
alloys = []
for i in range(200):
    comp = sample_composition()

    sy_true = compute_sigma_y(comp)
    ss_true, elong_base = compute_sigma_s_and_elong(comp, sy_true)

    h_i = rng.normal(0, 1)
    elong_true = np.clip(elong_base + HIDDEN_STD * h_i, 0.02, 0.40)

    sy_exp = sy_true * (1 + rng.normal(0, NOISE_SY))
    ss_exp = ss_true * (1 + rng.normal(0, NOISE_SS))
    elong_exp = elong_base * (1 + rng.normal(0, NOISE_ELONG))
    elong_exp = np.clip(elong_exp, 0.05, 0.35)

    eps_true, sigma_true_exp, eps_eng, sigma_eng, eps_c = voce_engineering(
        sy_exp, ss_exp, elong_exp)


    rho, theta, Lambda, sigma_true_full = latent_from_voce(eps_true, sy_exp, ss_exp, eps_c)


    i_uts = int(np.argmax(sigma_eng))
    uts_exp = float(sigma_eng[i_uts])
    m_el = eps_eng <= 0.003
    if m_el.sum() >= 3:
        E = np.polyfit(eps_eng[m_el], sigma_eng[m_el], 1)[0]
    else:
        E = (sigma_eng[2]-sigma_eng[0])/max(eps_eng[2]-eps_eng[0], 1e-6)
    offset = E * (eps_eng - 0.002)
    diff = sigma_eng - offset
    idx = np.where((eps_eng > 0.002) & (diff > 0))[0]
    ys_exp = float(sigma_eng[idx[0]]) if len(idx) > 0 else float(sigma_eng[1])
    elong_obs = float(eps_eng[-1])

    uts_true_approx = ss_true  
    elong_true_val = elong_true
    true_pse = uts_true_approx * elong_true_val


    desc = desc13(comp)
    X21_raw = np.array([desc[n] for n in PHYSIC_NAMES] + PROCESS_8, dtype=float).reshape(1, -1)
    X21_z = scaler.transform(X21_raw)[0]

    alloys.append({
        'idx': i, 'comp': comp, 'desc': desc, 'cond21': X21_z,
        'sy_exp': sy_exp, 'ss_exp': ss_exp, 'ec': eps_c,
        'sy_true': sy_true, 'ss_true': ss_true, 'elong_true': elong_true,
        'eps_true': eps_true, 'sigma_true': sigma_true_full,
        'rho': rho, 'theta': theta, 'Lambda': Lambda,
        'eps_eng': eps_eng, 'sigma_eng': sigma_eng,
        'ys_exp': ys_exp, 'uts_exp': uts_exp, 'elong_obs': elong_obs,
        'true_pse': true_pse,
    })

# ==================== 保存 ====================
print("保存...")
summary_rows = []
latent_rows = []
for a in alloys:
    pd.DataFrame({'strain_eng_pct': a['eps_eng']*100, 'stress_eng_MPa': a['sigma_eng']}
                 ).to_csv(os.path.join(OUT_DIR, f'alloy_{a["idx"]:03d}.csv'), index=False, float_format='%.4f')
    pd.DataFrame({'strain_true_pct': a['eps_true']*100, 'stress_true_MPa': a['sigma_true']}
                 ).to_csv(os.path.join(TRUE_DIR, f'alloy_{a["idx"]:03d}.csv'), index=False, float_format='%.4f')
    latent_rows.append({
        'Alloy_Num': a['idx'],
        'initial_feature': a['cond21'].tolist(),
        'latent_curve': [[float(x) for x in row] for row in
                         np.column_stack([a['eps_true']*100, a['rho'], a['theta'], a['Lambda'], a['sigma_true']])],
    })
    summary_rows.append({
        'idx': a['idx'], **{e: a['comp'][e] for e in ELEMENTS},
        'ys_MPa': round(a['ys_exp'], 2), 'uts_MPa': round(a['uts_exp'], 2),
        'elong_pct': round(a['elong_obs']*100, 2),
        'pse_MPa_pct': round(a['uts_exp']*a['elong_obs']*100, 2),  # 观测PSE
        'true_pse': round(a['true_pse'], 2),  # 真实PSE (全局最优目标)
        'voce_sy': round(a['sy_exp'], 1), 'voce_ss': round(a['ss_exp'], 1), 'voce_ec': round(a['ec'], 4),
    })
    for n in PHYSIC_NAMES:
        summary_rows[-1][n] = round(a['desc'][n], 6)

with open(os.path.join(HERE, 'data200/latent_curves.jsonl'), 'w') as f:
    for r in latent_rows:
        f.write(json.dumps(r) + '\n')

sdf = pd.DataFrame(summary_rows)
sdf.to_csv(os.path.join(HERE, 'data200', 'alloy_summary_200.csv'), index=False, float_format='%.6f')

# ==================== 检查 ====================
print(f"\n=== 145条 (物理模型+实验噪声) ===")
print(f"YS(观测): {sdf['ys_MPa'].min():.0f} ~ {sdf['ys_MPa'].max():.0f}")
print(f"UTS(观测): {sdf['uts_MPa'].min():.0f} ~ {sdf['uts_MPa'].max():.0f}")
print(f"观测PSE: {sdf['pse_MPa_pct'].min():.0f} ~ {sdf['pse_MPa_pct'].max():.0f}")
print(f"真实PSE: {sdf['true_pse'].min():.0f} ~ {sdf['true_pse'].max():.0f}")

from scipy.stats import spearmanr
rho_obs_true, _ = spearmanr(sdf['pse_MPa_pct'], sdf['true_pse'])
print(f"\n观测PSE vs 真实PSE: Spearman ρ={rho_obs_true:.3f} (噪声导致不一致)")

for e in ['W','Mo','Ta','Hf']:
    ru = np.corrcoef(sdf[e], sdf['uts_MPa'])[0,1]
    print(f"{e}: UTS r={ru:+.3f}")

best_true = sdf.nlargest(1, 'true_pse')
best_obs = sdf.nlargest(1, 'pse_MPa_pct')
print(f"\n真实PSE最优: idx={best_true['idx'].values[0]}, 真实PSE={best_true['true_pse'].values[0]:.0f}")
print(f"观测PSE最优: idx={best_obs['idx'].values[0]}, 观测PSE={best_obs['pse_MPa_pct'].values[0]:.0f}")
