import os, sys, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import al_ode_train as OT
import al_ode_common as OC
import al_common as C
from models.transformer_hyper_ODE_latent3 import LatentVariableNeuralODE
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
SETNAME = sys.argv[1]; MODEL = sys.argv[2]; REP = 'eng'
MAX_ROUND = int(sys.argv[3]) if len(sys.argv) > 3 else 25   
QUANT = ['rho', 'theta', 'Lambda', 'sigma']

sdf = pd.read_csv(f'{HERE}/data50/alloy_summary_200.csv')
sdf['pse_noh'] = sdf['uts_MPa'] * sdf['elong_pct']
pool_all = sorted(sdf.idx.tolist())        
topU = sdf.nlargest(10, 'uts_MPa').idx.tolist()
TU = {10: set(topU), 5: set(topU[:5]), 3: set(topU[:3]), 1: {int(topU[0])}}
print(f'[{SETNAME}/{MODEL}/{REP}] UTS top10={topU[:5]}... 门槛{int(sdf.uts_MPa.min())}-{int(sdf.uts_MPa.max())}', flush=True)

raw = {}
with open(f'{HERE}/data50/latent_curves.jsonl') as f:
    for line in f:
        r = json.loads(line); lc = np.array(r['latent_curve'], dtype=float)
        raw[int(r['Alloy_Num'])] = {'cond21': np.array(r['initial_feature'], dtype=float),
                                    'strain_pct': lc[:, 0], 'latent': lc[:, 1:]}

def eng_convert(d):
    et = d['strain_pct'] / 100.0                      
    ee = np.exp(et) - 1.0                              
    sig_t = d['latent'][:, 3]                          # σ_true
    sig_e = sig_t * np.exp(-et)                        # σ_eng
    q_e = np.gradient(sig_e, ee)                       # dσ_eng/dε_eng
    rho = d['latent'][:, 0]
    lam = d['latent'][:, 2]
    lat = np.column_stack([rho, sig_e, lam, q_e])
    return {'cond21': d['cond21'], 'strain_pct': ee * 100.0, 'latent': lat, 'eng': True}

def eng_rows(rows):
    out = {}
    for i, d in rows.items():
        out[i] = eng_convert(d)
    return out

latent_rows = raw if REP == 'true' else eng_rows(raw)

# ---------- init ----------
init = json.load(open(f'{HERE}/data50/selected_sets.json'))[SETNAME]
print(f'init n={len(init)}', flush=True)
OUT = f'{HERE}/data50/runs50/{SETNAME}_{MODEL}'
os.makedirs(OUT, exist_ok=True)
rec_path = f'{OUT}/result.json'
train = list(init)
rec = {'set': SETNAME, 'model': MODEL, 'rep': REP, 'rounds': [],
       'hit_first': {}, 'hit_full': {}}
if os.path.exists(rec_path):
    prev = json.load(open(rec_path))
    for rr in prev['rounds']:
        train += list(rr['sel'])
    train = sorted(set(train)); rec = prev
    print(f'  [resume] {len(rec["rounds"])} 轮', flush=True)

def stats_of(idx_list):
    allv = np.concatenate([latent_rows[i]['latent'] for i in idx_list], 0)
    st = {}
    for d, q in enumerate(QUANT):
        st[q] = {'mean': float(allv[:, d].mean()), 'std': float(allv[:, d].std())}
    return st

def train_mdl(it):
    zs = stats_of(train)
    mdir = f'{OUT}/m_iter{it}'
    if os.path.exists(f'{mdir}/model.pth') and not any(rr['iter'] == it for rr in rec['rounds']):
        return OT.load_ode(mdir)
    return OT.train_ode(train, zs, mdir, n_epochs=1500, latent_src=latent_rows)

GRID = np.linspace(0.0, 0.35 if REP == 'true' else 0.32, 220)

def predict_mc(mdl, idx, n=50):
    d = latent_rows[idx]
    src = d['strain_pct'] / 100.0
    smp = OT.predict_latent_raw(mdl, d['cond21'], d['strain_pct'], n_samples=n)  # [n,T,4] z-score
    si, qi = (3, 1) if REP == 'true' else (1, 3)      # sig/q 列位
    S = np.stack([np.interp(GRID, src, smp[k][:, si]) for k in range(n)])
    Q = np.stack([np.interp(GRID, src, smp[k][:, qi]) for k in range(n)])
    zs = stats_of(train)
    kS, kQ = ('sigma', 'theta') if REP == 'true' else ('theta', 'sigma')
    sm = S.mean(0) * zs[kS]['std'] + zs[kS]['mean']
    ss = S.std(0) * zs[kS]['std']
    qm = Q.mean(0) * zs[kQ]['std'] + zs[kQ]['mean']
    return sm, ss, qm


def sp_axis(d, cu):
    return GRID[:cu]

for it in range(len(rec['rounds']) + 1, MAX_ROUND + 1):
    remaining = [i for i in pool_all if i not in train]
    if not remaining:
        break
    t0 = time.time()
    if MODEL == 'xgb':
        import xgboost as xgb
        X = np.array([raw[i]['cond21'] for i in train])
        y = sdf.set_index('idx').loc[train, 'uts_MPa'].values
        m = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.08,
                             random_state=abs(hash(SETNAME)) % 99991 + it, n_jobs=2)
        m.fit(X, y)
        pred = m.predict(np.array([raw[i]['cond21'] for i in remaining]))
        sel = [remaining[j] for j in np.argsort(-pred)[:2]]
        score_note = ''
    else:
        mdl = train_mdl(it)
        def acq(idx):
            sm, ss, qm = predict_mc(mdl, idx, 50)
            sp = GRID
            sig_eng = sm * np.exp(-sp) if REP == 'true' else sm     
            e_cut = float(np.log(1.15)) if REP == 'true' else 0.15  
            cu = int(np.searchsorted(sp, e_cut + 1e-9))
            f = sm + 1.0 * sp * qm                                   
            return f, ss, cu, e_cut, sm
        tb = max(train, key=lambda i: sdf[sdf.idx == i].iloc[0]['uts_MPa'])
        f0, ss0, cu0, eu0, sm0 = acq(tb)
        vals = {}
        for idx in remaining:
            f, ss, cu, eu, sm = acq(idx)
            if cu < 2 or np.isnan(f[:cu].mean()):
                vals[idx] = -1e18
                continue
            if MODEL == 'det':
                vals[idx] = float(np.trapezoid(f[:cu] - f0[:cu], sp_axis(latent_rows[idx], cu)))
            else:
                z = (f - f0) / np.maximum(ss, 1e-9)
                EI = np.maximum((f - f0) * norm.cdf(z) + ss * norm.pdf(z), 0)
                vals[idx] = float(np.trapezoid(EI[:cu], sp_axis(latent_rows[idx], cu)))
        sel = sorted(vals, key=lambda k: -vals[k])[:2]
# 命中判定
    hit_round = {}
    for k, S in TU.items():
        for s in sel:
            if s in S:
                if k not in rec['hit_first']:
                    rec['hit_first'][k] = it
                hit_round[k] = hit_round.get(k, 0) + 1
    rec['rounds'].append({'iter': it, 'sel': sel,
                          'uts': [float(sdf[sdf.idx == i].iloc[0]['uts_MPa']) for i in sel],
                          'hit': {str(k): v for k, v in hit_round.items()}})
    json.dump(rec, open(rec_path, 'w'), indent=1)
    train = sorted(set(train).union(sel))
    print(f'  [iter{it}] {MODEL}/{REP} 选{sel} UTS={[int(sdf[sdf.idx==i].iloc[0]["uts_MPa"]) for i in sel]} '
          f'hit{hit_round} ({time.time()-t0:.0f}s)', flush=True)

for k, S in TU.items():
    found = sorted({s for rr in rec['rounds'] for s in rr['sel']} & S)
    if len(found) == len(S) and k not in rec['hit_full']:
        missing = set(S) - set(found)
        for rr in rec['rounds']:
            if missing & set(rr['sel']):
                rec['hit_full'][k] = rr['iter']
                break
    rec.setdefault('full_members', {})[str(k)] = [int(i) for i in found]
json.dump(rec, open(rec_path, 'w'), indent=1)
print(f'[done {SETNAME}/{MODEL}/{REP}] first={rec["hit_first"]} full={rec["hit_full"]}', flush=True)
