import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
D40 = HERE + '/data50'
D = D40
N_BOOT = 1000
MAXR = 25
BATCH = 2                    
OUTP = f'{D}/xgb_ehvi_results.json'


xl = pd.read_excel(f'{D}/ys_uts_200.xlsx')
IDX_LIST = xl.Alloy_Num.tolist()
pos = {i: p for p, i in enumerate(IDX_LIST)}
YS = xl.YS_MPa.values
UTS = xl.UTS_MPa.values
raw = {}
with open(f'{D}/latent_curves.jsonl') as f:
    for line in f:
        r = json.loads(line)
        raw[int(r['Alloy_Num'])] = np.array(r['initial_feature'], dtype=float)
X = np.array([raw[i] for i in IDX_LIST])

topU = [i for _, i in sorted(zip(-UTS, IDX_LIST))[:10]]
TU = {10: set(topU), 5: set(topU[:5]), 3: set(topU[:3]), 1: {int(topU[0])}}
REF = np.array([YS.min() - 1.0, UTS.min() - 1.0])   

# ---------- EHVI (2D, MC over bootstrap samples) ----------
def front_2d(pts):
    order = np.lexsort((-pts[:, 1], pts[:, 0]))     
    f = pts[order]
    keep = np.zeros(len(f), dtype=bool)
    best = -np.inf
    for i in range(len(f) - 1, -1, -1):               
        if f[i, 1] > best:
            keep[i] = True
            best = f[i, 1]
    fp = f[keep]
    a = np.append(fp[:, 0], np.inf)                   
    b = np.append(fp[:, 1], REF[1])                   
    a_prev = np.concatenate([[REF[0]], fp[:, 0]])     
    return fp, a, a_prev, b

def ehvi_mc(samples, a, a_prev, b):
    w = np.clip(np.minimum(samples[:, 0:1], a[None, :-1]) - a_prev[None, :-1], 0, None)
    h = np.clip(samples[:, 1:2] - b[None, :-1], 0, None)
    return (w * h).sum(1)


sets = json.load(open(f'{D}/selected_sets.json'))
out = json.load(open(OUTP)) if os.path.exists(OUTP) else {}
todo = [k for k in sets if k not in out]
print(f'sets: {len(sets)}, todo: {len(todo)}', flush=True)

for name in todo:
    t00 = time.time()
    init_idx = list(sets[name])
    train = [pos[i] for i in init_idx]
    rng = np.random.default_rng(1234 + int(name[2:].lstrip('0123456789') or 0) % 97)
    first = {}
    seen = {k: set() for k in TU}
    rounds = 0
    for it in range(1, MAXR + 1):
        rounds = it
        tr = np.array(train)
        rem = np.array([i for i in range(len(IDX_LIST)) if i not in train])
        if len(rem) == 0:
            break
        Xtr, ytr = X[tr], np.column_stack([YS[tr], UTS[tr]])
        Xrem = X[rem]
        preds = np.empty((len(rem), N_BOOT, 2))
        for b in range(N_BOOT):
            bag = rng.integers(0, len(tr), len(tr))
            m = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.08,
                                 random_state=int(rng.integers(1 << 30)), n_jobs=1)
            m.fit(Xtr[bag], ytr[bag])
            preds[:, b] = m.predict(Xrem)
        _, a, a_prev, fr_b = front_2d(np.column_stack([YS[tr], UTS[tr]]))
        ev = np.array([ehvi_mc(preds[j], a, a_prev, fr_b).mean()
                       for j in range(len(rem))])      
        mu_uts = preds[:, :, 1].mean(1)
        key = ev + 1e-9 * mu_uts
        sel_local = np.argsort(-key)[:BATCH]
        sel = [int(rem[j]) for j in sel_local]
        for k, S in TU.items():
            for s_idx in sel:
                alloy = IDX_LIST[s_idx]
                if alloy in S:
                    seen[k].add(alloy)
                    if k not in first:
                        first[k] = it
        train = sorted(set(train) | set(sel))
        print(f'  [{name}] round {it}: EHVI max={ev[sel_local[0]]:.1f} '
              f'sel={[IDX_LIST[s] for s in sel]} first={first} '
              f'({time.time()-t00:.0f}s)', flush=True)
        if 1 in first:
            break
    out[name] = {'first': {str(k): v for k, v in first.items()},
                 'rounds': rounds, 'top1_alloy': int(topU[0])}
    json.dump(out, open(OUTP, 'w'), indent=1)

print('\n===== 汇总 =====')
t1 = [out[k]['first'].get('1', 99) for k in out]
hit = [x for x in t1 if x < 99]
print(f'{len(out)} sets | top1 命中 {len(hit)}/{len(out)} | '
      f'首中轮 均值 {np.mean(hit):.1f} 中位 {np.median(hit):.0f}')
for k in out:
    print(f'  {k:8s} first={out[k]["first"]} rounds={out[k]["rounds"]}')
