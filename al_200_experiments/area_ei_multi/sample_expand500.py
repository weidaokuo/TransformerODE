import os
import sys
import pickle
import numpy as np
import pandas as pd
from scipy.stats import qmc

HERE = os.path.dirname(os.path.abspath(__file__))           
AL200 = os.path.dirname(HERE)                               
PKG = os.path.dirname(AL200)                                
CODE = f'{PKG}/code'                                        
sys.path.insert(0, CODE)
EXP60 = f'{AL200}/experiment60/scripts'                     

from obtain_feature import ObtainFeatures          
from Args import args                              


ELEMENTS = ['Nb', 'W', 'Mo', 'Zr', 'C', 'Hf', 'Ti', 'Ta', 'Si', 'N', 'V', 'Sc', 'Ru', 'B', 'Y']
ALLOY_EL = ['W', 'Mo', 'Zr', 'C', 'Hf', 'Ti', 'Ta', 'Si', 'N', 'V', 'Sc', 'Ru', 'B', 'Y']
KEY_ELEMS = ['W', 'Mo', 'Ta', 'Hf']
PROCESS_8 = [0.0, 0.0, 0.0, 1450.0, 1.0, 0.0, 0.0, 0.0]   
PHYSIC_NAMES = list(args.physic_names)                    

EXPAND = 0.20          
N_TARGET = 500         
SEED = 20261009        
NB_MIN = 58.0          
KEY_MIN_N = 2          

OUT = f'{HERE}/data/expand500.npz'


def main():
    sdf = pd.read_csv(f'{EXP60}/data60/alloy_summary_200.csv')
    print(f'[load] 200条 summary: {len(sdf)} 行, Nb 范围 {sdf.Nb.min():.2f}~{sdf.Nb.max():.2f}', flush=True)

    lo, hi = [], []
    for e in ALLOY_EL:
        a, b = float(sdf[e].min()), float(sdf[e].max())
        span = b - a
        lo.append(max(0.0, a - EXPAND * span))     
        hi.append(b + EXPAND * span)
    lo = np.array(lo)
    hi = np.array(hi)
    print(f'[range] ±{EXPAND*100:.0f}% 扩张后各元素范围:', flush=True)
    for j, e in enumerate(ALLOY_EL):
        print(f'   {e:3s} [{lo[j]:7.3f}, {hi[j]:7.3f}]   (200条 {sdf[e].min():.3f}~{sdf[e].max():.3f})', flush=True)


    comps, tried = [], 0
    batch = 64
    while len(comps) < N_TARGET and tried < 20000:
        S = qmc.LatinHypercube(d=len(ALLOY_EL), seed=SEED + tried).random(batch)
        for s in S:
            tried += 1
            v = lo + s * (hi - lo)
            nb = 100.0 - v.sum()
            n_key = sum(1 for j, e in enumerate(ALLOY_EL) if e in KEY_ELEMS and v[j] > 1.0)
            if nb >= NB_MIN and n_key >= KEY_MIN_N:
                comps.append(np.concatenate([[nb], v]))
                if len(comps) >= N_TARGET:
                    break
    comps = np.array(comps)
    print(f'[lhs] 接受 {len(comps)}/{tried} = {100*len(comps)/tried:.0f}%  '
          f'(准则 Nb>={NB_MIN} 且 {KEY_ELEMS} 中>={KEY_MIN_N}个>1%)', flush=True)
    print(f'[lhs] Nb 范围 {comps[:,0].min():.2f}~{comps[:,0].max():.2f}  '
          f'(200条 {sdf.Nb.min():.2f}~{sdf.Nb.max():.2f})', flush=True)

    n_out = 0
    for r in comps[:, 1:]:
        if any(r[j] > sdf[ALLOY_EL[j]].max() + 1e-9 or r[j] < sdf[ALLOY_EL[j]].min() - 1e-9
               for j in range(len(ALLOY_EL))):
            n_out += 1
    print(f'[lhs] 至少一维超出200条范围的点: {n_out}/{len(comps)} = {100*n_out/len(comps):.0f}%', flush=True)

    pkl = pickle.load(open(f'{CODE}/voce_km_latent/results/ml_final_two_tier.pkl', 'rb'))
    scaler = pkl['tier2_voce29']['scaler']
    assert scaler.n_features_in_ == 21, f'scaler 维度异常: {scaler.n_features_in_}'

    cond21 = np.zeros((len(comps), 21), dtype=np.float64)
    desc13 = np.zeros((len(comps), 13), dtype=np.float64)
    for i, comp in enumerate(comps):
        cd = {e: float(comp[j]) for j, e in enumerate(ELEMENTS)}
        el = [e for e in ELEMENTS if cd.get(e, 0) > 1e-8]
        cl = [cd[e] for e in el]
        key = ObtainFeatures(el, cl).get_key_features()
        d = np.array([float(key.get(n, 0.0)) for n in PHYSIC_NAMES], dtype=float)
        desc13[i] = d
        X21_raw = np.concatenate([d, np.array(PROCESS_8, dtype=float)]).reshape(1, -1)
        cond21[i] = scaler.transform(X21_raw)[0]
        if (i + 1) % 100 == 0:
            print(f'[feat] {i+1}/{len(comps)} 完成', flush=True)

    assert np.all(np.isfinite(cond21)), 'cond21 含 NaN/Inf'
    print(f'[feat] cond21 shape={cond21.shape}  有限值检查通过', flush=True)
    print(f'[feat] cond21 各维范围 (min~max):', flush=True)
    for k in range(21):
        print(f'   dim{k:2d}: {cond21[:,k].min():9.3f} ~ {cond21[:,k].max():9.3f}', flush=True)

    rows = [__import__('json').loads(l) for l in open(f'{EXP60}/data60/latent_curves.jsonl')]
    c200 = np.array([np.array(r['initial_feature'], dtype=float) for r in rows])
    print(f'\n[diag] 200条 cond21 范围 vs 500点 cond21 范围 (前6维):', flush=True)
    for k in range(6):
        print(f'   dim{k}: 200条 [{c200[:,k].min():8.3f},{c200[:,k].max():8.3f}]  '
              f'500点 [{cond21[:,k].min():8.3f},{cond21[:,k].max():8.3f}]', flush=True)


    os.makedirs(f'{HERE}/data', exist_ok=True)
    np.savez_compressed(OUT, comps=comps, cond21=cond21, desc13=desc13,
                        elems=np.array(ELEMENTS), expand=EXPAND, seed=SEED,
                        nb_min=NB_MIN, key_min_n=KEY_MIN_N)

    df = pd.DataFrame(comps, columns=ELEMENTS)
    df.insert(0, 'point_id', [f'X{i:03d}' for i in range(len(comps))])
    for k in range(13):
        df[f'desc_{PHYSIC_NAMES[k]}'] = desc13[:, k]
    for k in range(21):
        df[f'cond21_{k}'] = cond21[:, k]
    df.to_csv(f'{HERE}/data/expand500.csv', index=False)
    print(f'\n[save] {OUT}', flush=True)
    print(f'[save] {HERE}/data/expand500.csv', flush=True)


if __name__ == '__main__':
    main()
