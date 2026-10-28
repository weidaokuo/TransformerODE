
import os
import sys
import json
import time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import common_gen as C                                   
sys.path.insert(0, f'{C.EXP}/scripts')                      

import al_ode_train as OT                               

GRID = C.GRID
CU = C.CU
MC_N = C.MC_N


def predict_mc_expand(mdl, cond21, zstats, n=MC_N):
    sp_pct = C.EXPAND_SP                              
    src = sp_pct / 100.0                              
    smp = OT.predict_latent_raw(mdl, cond21, sp_pct, n_samples=n)  
    si, qi = 1, 3                                      
    S = np.stack([np.interp(GRID, src, smp[k][:, si]) for k in range(n)])
    Q = np.stack([np.interp(GRID, src, smp[k][:, qi]) for k in range(n)])
    kS, kQ = 'theta', 'sigma'                         
    sm = S.mean(0) * zstats[kS]['std'] + zstats[kS]['mean']
    qm = Q.mean(0) * zstats[kQ]['std'] + zstats[kQ]['mean']
    return sm, qm


def main():
    setname = sys.argv[1]
    rnd = int(sys.argv[2])
    t_start = time.time()
    print(f'[A/gen{C.GEN}/{setname}/r{rnd}] start', flush=True)

    sdf = C.load_summary()
    lr = C.load_latent_rows_eng()
    ex = C.load_expand500()
    cond500 = ex['cond21']
    assert len(cond500) == C.N_EXPAND, f'扩充点数异常: {len(cond500)}'
    if os.environ.get('AREAEI_SMOKE'):
        nsm = int(os.environ.get('AREAEI_N_EXPAND', 30))
        cond500 = cond500[:nsm]
        print(f'[A/gen{C.GEN}/{setname}/r{rnd}] ⚠ SMOKE: 只预测 {nsm}/{C.N_EXPAND} 个扩充点'flush=True)

    init = C.load_init(setname)
    st = C.load_state(setname)
    train = sorted(st['train']) if st['rounds_done'] > 0 else sorted(init)
    if rnd == 1:
        train = sorted(init)
    print(f'[A/gen{C.GEN}/{setname}/r{rnd}] 训练集 n={len(train)} (init={len(init)})', flush=True)

    tb, f0 = C.pick_f0(sdf, lr, init)
    print(f'[A/gen{C.GEN}/{setname}/r{rnd}] f0 = idx{tb} 真实曲线 (UTS '
          f'{float(sdf[sdf.idx==tb].iloc[0].uts_MPa):.0f} MPa), 全程固定', flush=True)

    zs = C.stats_of(lr, train)
    mdir = C.det_model_dir(setname, rnd)
    if rnd == 1:
        assert os.path.exists(f'{mdir}/model.pth'), f'现成 det 缺失: {mdir}'
        print(f'[A/gen{C.GEN}/{setname}/r{rnd}] 加载现成 det: {mdir}', flush=True)
        t0 = time.time()
        mdl = OT.load_ode(mdir)
        print(f'[A/gen{C.GEN}/{setname}/r{rnd}] 加载完成 ({time.time()-t0:.0f}s)', flush=True)
    else:
        os.makedirs(mdir, exist_ok=True)
        if os.path.exists(f'{mdir}/model.pth'):
            print(f'[A/gen{C.GEN}/{setname}/r{rnd}] det 已存在, 跳过重训: {mdir}', flush=True)
            mdl = OT.load_ode(mdir)
        else:
            print(f'[A/gen{C.GEN}/{setname}/r{rnd}] 重训 det (n_epochs=1500, 与det原方法一致)...',
                  flush=True)
            t0 = time.time()
            OT.train_ode(train, zs, mdir, n_epochs=1500, latent_src=lr)
            print(f'[A/gen{C.GEN}/{setname}/r{rnd}] det 训练完成 ({time.time()-t0:.0f}s)',
                  flush=True)
            mdl = OT.load_ode(mdir)

    print(f'[A/gen{C.GEN}/{setname}/r{rnd}] MC{MC_N} 预测 {len(cond500)} 个扩充点...', flush=True)
    t0 = time.time()
    a500 = np.zeros(len(cond500), dtype=np.float64)
    for i in range(len(cond500)):
        sm, qm = predict_mc_expand(mdl, cond500[i], zs)
        f = C.f_from_mc(sm, qm)
        if np.isnan(f[:CU].mean()):
            a500[i] = -1e18
        else:
            a500[i] = C.area_of(f, f0)
        if (i + 1) % 100 == 0:
            print(f'   [A/gen{C.GEN}/{setname}/r{rnd}] {i+1}/{len(cond500)} '
                  f'({time.time()-t0:.0f}s, 剩 {(time.time()-t0)/(i+1)*(len(cond500)-i-1):.0f}s)',
                  flush=True)
    print(f'[A/gen{C.GEN}/{setname}/r{rnd}] 面积完成 ({time.time()-t0:.0f}s): '
          f'正 {(a500>0).sum()} 负 {(a500<0).sum()}  '
          f'min {a500[a500>-1e17].min():.2f} max {a500.max():.2f} mean {a500[a500>-1e17].mean():.2f}',
          flush=True)


    a_real = C.real_areas(sdf, lr, f0, train)

    rdir = C.round_dir(setname, rnd)
    os.makedirs(rdir, exist_ok=True)
    np.savez_compressed(f'{rdir}/areas.npz',
                        a500=a500, cond500=cond500, f0=f0,
                        train_idx=np.array(train, dtype=int),
                        train_area=np.array([a_real[i] for i in train], dtype=np.float64),
                        tb=np.array([tb]))
    json.dump({'set': setname, 'gen': C.GEN, 'round': rnd, 'n_train': len(train), 'tb': tb,
               'n_expand': len(a500),
               'a500_pos': int((a500 > 0).sum()), 'a500_neg': int((a500 < 0).sum()),
               'a500_max': float(a500.max()),
               'a500_mean_finite': float(a500[a500 > -1e17].mean()),
               'a_real_max': float(max(a_real.values())),
               'a_real_min': float(min(a_real.values())),
               'det_dir': mdir, 'det_retrained': rnd > 1,
               'sec_det': None, 'sec_total': round(time.time() - t_start, 1)},
              open(f'{rdir}/stageA.json', 'w'), indent=1)
    print(f'[A/gen{C.GEN}/{setname}/r{rnd}] 落盘 {rdir}/areas.npz + stageA.json '
          f'(总耗时 {time.time()-t_start:.0f}s)', flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)                         
    main()
