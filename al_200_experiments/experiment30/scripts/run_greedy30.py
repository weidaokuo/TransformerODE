import json, os, sys, time
import numpy as np
import pandas as pd
import xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
D = f'{HERE}/data30'
SEED_BASE = 20260830
MAXR = 25

sdf = pd.read_csv(f'{D}/alloy_summary_200.csv')
topU = sdf.nlargest(10, 'uts_MPa').idx.tolist()
TU = {10: set(topU), 5: set(topU[:5]), 3: set(topU[:3]), 1: {int(topU[0])}}
pool_all = sorted(sdf.idx.tolist())
raw = {}
with open(f'{D}/latent_curves.jsonl') as f:
    for line in f:
        r = json.loads(line)
        raw[int(r['Alloy_Num'])] = np.array(r['initial_feature'], dtype=float)

sets = json.load(open(f'{os.path.dirname(os.path.dirname(D))}/selected_sets_10.json'))
OUTP = f'{D}/greedy_results_repro.json'
out = json.load(open(OUTP)) if os.path.exists(OUTP) else {}

todo = [sys.argv[1]] if len(sys.argv) > 1 and sys.argv[1] != 'all' else sorted(sets)
for name in todo:
    if name not in sets:
        print(f'!! {name} 不在 10-set 名单'); continue
    t0 = time.time()
    train = list(sets[name])
    first, full = {}, {}
    seen = {k: set() for k in TU}
    rounds = []
    for it in range(1, MAXR + 1):
        rem = [i for i in pool_all if i not in train]
        if not rem:
            break
        m = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.08,
                             random_state=SEED_BASE + it, n_jobs=1)
        Xtr = np.array([raw[i] for i in train])
        ytr = sdf[sdf.idx.isin(train)].set_index('idx').loc[train, 'uts_MPa'].values
        m.fit(Xtr, ytr)
        pred = m.predict(np.array([raw[i] for i in rem]))
        sel = [rem[j] for j in np.argsort(-pred)[:2]]
        for k, S in TU.items():
            for s in sel:
                if s in S:
                    seen[k].add(s)
                    if k not in first:
                        first[k] = it
        for k, S in TU.items():
            if len(seen[k]) == len(S) and k not in full:
                full[k] = it
        rounds.append({'iter': it, 'sel': sel})
        train = sorted(set(train) | set(sel))
        if 1 in first:
            break
    out[name] = {'first': {str(k): v for k, v in first.items()},
                 'full': {str(k): v for k, v in full.items()},
                 'rounds': rounds}
    json.dump(out, open(OUTP, 'w'), indent=1)
    print(f'[greedy30/{name}] first={first} rounds={len(rounds)} '
          f'({time.time()-t0:.0f}s)', flush=True)

print(f'\n[greedy30] 完成 {len(out)} set -> {OUTP}')
