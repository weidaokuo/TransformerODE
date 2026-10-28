
import os
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))          
AL200 = os.path.dirname(HERE)                               
GEN = int(os.environ.get('AREAEI_GEN', '60'))
assert GEN in (30, 40, 50, 60), f'AREAEI_GEN 非法: {GEN}'
EXP = f'{AL200}/experiment{GEN}'
DATA = f'{EXP}/scripts/data{GEN}'                          
RUNS = f'{EXP}/runs{GEN}'                                   
EXPAND_NPZ = f'{HERE}/data/expand500.npz'


QUANT = ['rho', 'theta', 'Lambda', 'sigma']
GRID = np.linspace(0.0, 0.32, 220)
E_CUT = 0.15
CU = int(np.searchsorted(GRID, E_CUT + 1e-9))              # = 103
MAX_ROUND = 25
BATCH = 2
MC_N = 50
N_EXPAND = 500
EXPAND_SP = GRID * 100.0
N_BOOTSTRAP = 1000



def load_summary():
    return pd.read_csv(f'{DATA}/alloy_summary_200.csv')


def load_latent_rows_eng():
    raw = {}
    with open(f'{DATA}/latent_curves.jsonl') as f:
        for line in f:
            r = json.loads(line)
            lc = np.array(r['latent_curve'], dtype=float)
            raw[int(r['Alloy_Num'])] = {'cond21': np.array(r['initial_feature'], dtype=float),
                                        'strain_pct': lc[:, 0], 'latent': lc[:, 1:]}

    def eng_convert(d):
        et = d['strain_pct'] / 100.0
        ee = np.exp(et) - 1.0
        sig_t = d['latent'][:, 3]
        sig_e = sig_t * np.exp(-et)
        q_e = np.gradient(sig_e, ee)
        rho = d['latent'][:, 0]
        lam = d['latent'][:, 2]
        lat = np.column_stack([rho, sig_e, lam, q_e])
        return {'cond21': d['cond21'], 'strain_pct': ee * 100.0, 'latent': lat, 'eng': True}

    return {i: eng_convert(d) for i, d in raw.items()}


def load_init(setname):
    return json.load(open(f'{DATA}/selected_sets.json'))[setname]


def load_expand500():
    z = np.load(EXPAND_NPZ, allow_pickle=True)
    return {'cond21': z['cond21'], 'comps': z['comps'], 'desc13': z['desc13']}


def top_k_sets(sdf):
    sdf = sdf.sort_values('idx')
    pool_all = sorted(sdf.idx.tolist())
    topU = sdf.nlargest(10, 'uts_MPa').idx.tolist()
    return {10: set(topU), 5: set(topU[:5]), 3: set(topU[:3]), 1: {int(topU[0])}}, pool_all, topU


def stats_of(latent_rows, idx_list):
    allv = np.concatenate([latent_rows[i]['latent'] for i in idx_list], 0)
    st = {}
    for d, q in enumerate(QUANT):
        st[q] = {'mean': float(allv[:, d].mean()), 'std': float(allv[:, d].std())}
    return st


def f_from_real(latent_rows, idx):
    d = latent_rows[idx]
    src = d['strain_pct'] / 100.0
    sm = np.interp(GRID, src, d['latent'][:, 1])
    qm = np.interp(GRID, src, d['latent'][:, 3])
    return sm + 1.0 * GRID * qm


def f_from_mc(S_mean, Q_mean):
    return S_mean + 1.0 * GRID * Q_mean


def area_of(f, f0):
    return float(np.trapezoid(f[:CU] - f0[:CU], GRID[:CU]))


def pick_f0(sdf, latent_rows, init):
    sub = sdf[sdf.idx.isin(init)]
    tb = int(sub.loc[sub.uts_MPa.idxmax(), 'idx'])
    f0 = f_from_real(latent_rows, tb)
    return tb, f0


def real_areas(sdf, latent_rows, f0, idx_list):
    return {int(i): area_of(f_from_real(latent_rows, i), f0) for i in idx_list}


def run_dir(setname):
    suffix = '_SMOKE' if os.environ.get('AREAEI_SMOKE') else ''
    return f'{HERE}/runs/gen{GEN}/{setname}_areaei{suffix}'


def round_dir(setname, r):
    return f'{run_dir(setname)}/round_{r:02d}'


def det_model_dir(setname, r):
    if r == 1:
        return f'{RUNS}/{setname}_det/m_iter1'
    return f'{round_dir(setname, r)}/det_model'


def state_path(setname):
    return f'{run_dir(setname)}/state.json'


def load_state(setname):
    p = state_path(setname)
    if os.path.exists(p):
        return json.load(open(p))
    return {'set': setname, 'rounds_done': 0, 'train': [], 'hit_first': {}, 'rounds': []}


def save_state(setname, st):
    os.makedirs(run_dir(setname), exist_ok=True)
    json.dump(st, open(state_path(setname), 'w'), indent=1)
