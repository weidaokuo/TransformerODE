
import os
import sys
import json
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import xgboost as xgb                                  
import common_gen as C                                 
from scipy.stats import norm                           

SEED_BASE = 20261009
FBEST_MODE = 'cand_upper'                              


def xgb_params(setname, rnd, b):
    """与 runner 第116~117行 xgb 基线同超参; random_state 随 bootstrap 序号变化"""
    return dict(n_estimators=200, max_depth=4, learning_rate=0.08,
                random_state=(abs(hash(setname)) % 99991) + rnd * 100000 + b,
                n_jobs=1)


def main():
    setname = sys.argv[1]
    rnd = int(sys.argv[2])
    t_start = time.time()
    sdf = C.load_summary()
    lr = C.load_latent_rows_eng()
    TU, pool_all, topU = C.top_k_sets(sdf)

    rdir = C.round_dir(setname, rnd)
    az = np.load(f'{rdir}/areas.npz')
    a500, cond500 = az['a500'], az['cond500']
    train_idx = az['train_idx'].tolist()
    train_area = az['train_area']
    assert len(train_idx) == len(train_area)
    print(f'[B/gen{C.GEN}/{setname}/r{rnd}] stageA 读入: 真实{len(train_idx)}条 + 扩充{len(a500)}条',
          flush=True)


    X_real = np.array([lr[i]['cond21'] for i in train_idx], dtype=np.float64)
    y_real = np.asarray(train_area, dtype=np.float64)
    X = np.vstack([X_real, cond500])
    y = np.concatenate([y_real, a500])
    is_real = np.concatenate([np.ones(len(y_real), bool), np.zeros(len(a500), bool)])
    finite = np.isfinite(y) & (y > -1e17)
    X, y, is_real = X[finite], y[finite], is_real[finite]
    print(f'[B/gen{C.GEN}/{setname}/r{rnd}] XGB 训练集 n={len(y)} '
          f'(真实{is_real.sum()} + 扩充{(~is_real).sum()})', flush=True)
    print(f'[B/gen{C.GEN}/{setname}/r{rnd}] 面积标签: 正{(y>0).sum()} 负{(y<0).sum()} | '
          f'真实段 min{y[is_real].min():.2f}~max{y[is_real].max():.2f} | '
          f'扩充段 min{y[~is_real].min():.2f}~max{y[~is_real].max():.2f}', flush=True)


    remaining = [i for i in pool_all if i not in set(train_idx)]
    Xc = np.array([lr[i]['cond21'] for i in remaining], dtype=np.float64)
    print(f'[B/gen{C.GEN}/{setname}/r{rnd}] 候选池 n={len(remaining)} (200 - 已训练{len(train_idx)})',
          flush=True)
    if not remaining:
        print(f'[B/gen{C.GEN}/{setname}/r{rnd}] 候选池耗尽, 结束', flush=True)
        return

    # ---------- Bootstrap 1000 次 → μ, σ ----------
    nb = int(os.environ.get('AREAEI_NB', C.N_BOOTSTRAP))
    if nb != C.N_BOOTSTRAP:
        print(f'[B/gen{C.GEN}/{setname}/r{rnd}] ⚠ SMOKE: bootstrap={nb} (正式应为 {C.N_BOOTSTRAP}), '
              f'结果不可用于论文', flush=True)
    rng = np.random.default_rng(SEED_BASE + rnd)
    preds = np.zeros((nb, len(remaining)), dtype=np.float32)
    t0 = time.time()
    n = len(y)
    for b in range(nb):
        idx_b = rng.integers(0, n, n)
        m = xgb.XGBRegressor(**xgb_params(setname, rnd, b))
        m.fit(X[idx_b], y[idx_b])
        preds[b] = m.predict(Xc)
        if (b + 1) % 200 == 0:
            el = time.time() - t0
            print(f'   [B/gen{C.GEN}/{setname}/r{rnd}] bootstrap {b+1}/{nb} '
                  f'({el:.0f}s, 剩 {el/(b+1)*(nb-b-1):.0f}s)', flush=True)
    mu = preds.mean(0).astype(np.float64)
    sg = preds.std(0).astype(np.float64)
    print(f'[B/gen{C.GEN}/{setname}/r{rnd}] bootstrap 完成 ({time.time()-t0:.0f}s): '
          f'μ {mu.min():.2f}~{mu.max():.2f} | σ {sg.min():.3f}~{sg.max():.3f}', flush=True)

    # ---------- 标准 EI (ξ=0), f_best = 候选μ上界 ----------
    f_best = float(mu.max())
    z = (mu - f_best) / np.maximum(sg, 1e-12)
    EI = np.maximum((mu - f_best) * norm.cdf(z) + sg * norm.pdf(z), 0.0)
    order = np.argsort(-EI)
    sel = [remaining[j] for j in order[:C.BATCH]]
    print(f'[B/gen{C.GEN}/{setname}/r{rnd}] f_best={f_best:.3f} | EI top2={sel} '
          f'UTS={[int(sdf[sdf.idx==i].iloc[0].uts_MPa) for i in sel]}', flush=True)

    st = C.load_state(setname)
    hit_round = {}
    for k, S in TU.items():
        for s in sel:
            if s in S:
                if str(k) not in st['hit_first']:
                    st['hit_first'][str(k)] = rnd
                hit_round[str(k)] = hit_round.get(str(k), 0) + 1

    st['train'] = sorted(set(train_idx).union(sel))
    st['rounds'].append({
        'iter': rnd, 'sel': sel,
        'uts': [float(sdf[sdf.idx == i].iloc[0]['uts_MPa']) for i in sel],
        'area_mu': [float(mu[j]) for j in order[:C.BATCH]],
        'area_sigma': [float(sg[j]) for j in order[:C.BATCH]],
        'ei': [float(EI[j]) for j in order[:C.BATCH]],
        'hit': hit_round,
        'n_train_xgb': int(len(y)), 'n_expand_pos': int((a500 > 0).sum()),
        'f_best': f_best, 'fbest_mode': FBEST_MODE,
        'sec': round(time.time() - t_start, 1)})
    st['rounds_done'] = rnd
    for k, S in TU.items():
        found = sorted({s for rr in st['rounds'] for s in rr['sel']} & S)
        if len(found) == len(S) and str(k) not in st.get('hit_full', {}):
            st.setdefault('hit_full', {})[str(k)] = rnd
        st.setdefault('full_members', {})[str(k)] = [int(i) for i in found]
    C.save_state(setname, st)


    np.savez_compressed(f'{rdir}/ei_detail.npz',
                        remaining=np.array(remaining, dtype=int), mu=mu, sigma=sg, EI=EI,
                        preds_sample=preds[:50], f_best=f_best,
                        sel=np.array(sel, dtype=int),
                        uts_true=np.array([float(sdf[sdf.idx == i].iloc[0]['uts_MPa'])
                                           for i in remaining]))
    json.dump({'set': setname, 'gen': C.GEN, 'round': rnd, 'sel': sel, 'hit': hit_round,
               'hit_first': st['hit_first'], 'f_best': f_best, 'fbest_mode': FBEST_MODE,
               'mu_range': [float(mu.min()), float(mu.max())],
               'sigma_range': [float(sg.min()), float(sg.max())],
               'n_candidates': len(remaining), 'n_train_xgb': int(len(y)),
               'sec': round(time.time() - t_start, 1)},
              open(f'{rdir}/stageB.json', 'w'), indent=1)
    print(f'[B/gen{C.GEN}/{setname}/r{rnd}] 选{sel} hit{hit_round} | 落盘 state + ei_detail '
          f'({time.time()-t_start:.0f}s)', flush=True)
    if str(1) in st['hit_first']:
        print(f'[B/gen{C.GEN}/{setname}/r{rnd}] ★ 命中 top1 (idx{topU[0]}) 于第 '
              f'{st["hit_first"]["1"]} 轮', flush=True)


if __name__ == '__main__':
    main()
