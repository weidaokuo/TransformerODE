
import os
import sys
import json
import time
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import common_gen as C                                   

PY_MODEL = os.environ.get('PY_MODEL', sys.executable)      


def run_stage(script, setname, rnd, extra_env=None):
    logdir = f'{HERE}/logs/gen{C.GEN}'
    os.makedirs(logdir, exist_ok=True)
    tag = 'A' if 'stageA' in script else 'B'
    suffix = '_SMOKE' if os.environ.get('AREAEI_SMOKE') else ''
    logpath = f'{logdir}/{setname}_areaei{suffix}_r{rnd:02d}_{tag}.log'
    env = dict(os.environ)
    env['OMP_NUM_THREADS'] = '2'                      
    env['MKL_NUM_THREADS'] = '2'
    env['AREAEI_GEN'] = str(C.GEN)                    
    if extra_env:
        env.update(extra_env)
    t0 = time.time()
    with open(logpath, 'w') as lf:
        p = subprocess.run([PY_MODEL, '-u', f'{HERE}/{script}', setname, str(rnd)],
                           stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=HERE)
    dt = time.time() - t0
    if p.returncode != 0:
        tail = ''.join(open(logpath, errors='ignore').readlines()[-25:])
        raise RuntimeError(f'stage{tag} {setname} r{rnd} failed (rc={p.returncode}, {dt:.0f}s)\n'
                           f'Log: {logpath}\n--- Tail ---\n{tail}')
    print(f'  [driver/gen{C.GEN}/{setname}] stage{tag} r{rnd} OK ({dt:.0f}s) → {logpath}',
          flush=True)
    return dt


def main():
    setname = sys.argv[1]
    smoke = '--smoke' in sys.argv
    if smoke:
        os.environ['AREAEI_SMOKE'] = '1'

    st = C.load_state(setname)
    start = st['rounds_done'] + 1
    if str(1) in st.get('hit_first', {}) and not smoke:
        print(f'[driver/gen{C.GEN}/{setname}] top1 already hit at round {st["hit_first"]["1"]} '
              f'(rounds_done={st["rounds_done"]}) → skipping, no rerun', flush=True)
        return
    max_round = start if smoke else int(os.environ.get('AREAEI_MAXR', C.MAX_ROUND))
    if start > 1:
        print(f'[driver/gen{C.GEN}/{setname}] Resume from checkpoint: {st["rounds_done"]} rounds done, '
              f'starting from round {start}', flush=True)
    else:
        print(f'[driver/gen{C.GEN}/{setname}] Fresh start (init n={len(C.load_init(setname))})',
              flush=True)
    print(f'[driver/gen{C.GEN}/{setname}] max_round={max_round} smoke={smoke} '
          f'bootstrap={20 if smoke else C.N_BOOTSTRAP} fbest=cand_upper (scheme B locked)', flush=True)

    t_all = time.time()
    for rnd in range(start, max_round + 1):
        print(f'\n===== [driver/gen{C.GEN}/{setname}] ROUND {rnd}/{max_round} =====', flush=True)
        env = {'AREAEI_SMOKE': '1', 'AREAEI_NB': '20'} if smoke else None
        tA = run_stage('stageA_gen.py', setname, rnd, env)
        tB = run_stage('stageB_gen.py', setname, rnd, env)

        st = C.load_state(setname)                      
        last = st['rounds'][-1]
        print(f'  [driver/gen{C.GEN}/{setname}] r{rnd} selected {last["sel"]} hit{last["hit"]} '
              f'this round {tA+tB:.0f}s (A {tA:.0f} + B {tB:.0f})', flush=True)

        if str(1) in st['hit_first'] and not smoke:
            print(f'\n★ [driver/gen{C.GEN}/{setname}] top1 hit at round {st["hit_first"]["1"]} '
                  f'→ stopping (stop on top1 hit, max {C.MAX_ROUND} rounds)', flush=True)
            break
        if not st['rounds'] or len(last['sel']) == 0:
            print(f'[driver/gen{C.GEN}/{setname}] No selection point, stopping', flush=True)
            break

    st = C.load_state(setname)
    print(f'\n[driver/gen{C.GEN}/{setname}] Completed {st["rounds_done"]} rounds, total time '
          f'{(time.time()-t_all)/3600:.2f} h', flush=True)
    print(f'[driver/gen{C.GEN}/{setname}] hit_first = {st["hit_first"]}', flush=True)
    print(f'[driver/gen{C.GEN}/{setname}] hit_full  = {st.get("hit_full", {})}', flush=True)
    json.dump({'set': setname, 'gen': C.GEN, 'rounds_done': st['rounds_done'],
               'hit_first': st['hit_first'], 'hit_full': st.get('hit_full', {}),
               'smoke': smoke, 'hours': round((time.time() - t_all) / 3600, 3)},
              open(f'{C.run_dir(setname)}/driver_summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
