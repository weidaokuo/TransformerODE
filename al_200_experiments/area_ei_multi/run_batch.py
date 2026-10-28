#!/usr/bin/env python3
import os
import sys
import json
import time
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PY_MODEL = os.environ.get('PY_MODEL', sys.executable)       
MAX_PAR = 3          
BATCH_STATE = f'{HERE}/batch_state.json'
HEARTBEAT = f'{HERE}/logs/batch/heartbeat.txt'

TASKS = {
 30: ['bi007', 'hg007', 'ht002', 'ht009', 'lx005', 'mi023', 'r005', 'r020', 'r033', 'st009'],
 40: ['bi004', 'cv007', 'ht002', 'lo002', 'lo007', 'lo019', 'lo023', 'lx006', 'mi008', 'r032'],
 50: ['bi002', 'cl002', 'cl013', 'lo008', 'lo009', 'mi006', 'mi007', 'r049', 'st012', 'st014'],
 60: ['bi013', 'cl013', 'ht007', 'lo000', 'lo004', 'lx004', 'lx008', 'mi009', 'mi016', 'st000'],
}


def load_state():
    if os.path.exists(BATCH_STATE):
        return json.load(open(BATCH_STATE))
    return {'done': {}, 'running': {}, 'failed': {}}


def save_state(st):
    json.dump(st, open(BATCH_STATE, 'w'), indent=1)


def task_key(g, s):
    return f'gen{g}_{s}'


def is_finished(g, s):
    """set 完成判据 (唯一权威: 该 set 自己的 state.json):
       命中 top1 或 rounds_done>=25 (预算尽)"""
    p = f'{HERE}/runs/gen{g}/{s}_areaei/state.json'
    if not os.path.exists(p):
        return False, None
    try:
        st = json.load(open(p))
    except Exception:
        return False, None
    if '1' in st.get('hit_first', {}):
        return True, f'top1@{st["hit_first"]["1"]}'
    if st.get('rounds_done', 0) >= 25:
        return True, '25轮未中(预算尽)'
    return False, f'进行中 rounds_done={st.get("rounds_done", 0)}'


def launch(g, s):
    """拉起一个 detached driver; 返回 Popen 对象 (不是 pid!) —
    巡检必须用 proc.poll(), 禁止 os.waitpid (ECHILD 教训见文件头)"""
    logdir = f'{HERE}/logs/batch'
    os.makedirs(logdir, exist_ok=True)
    logf = open(f'{logdir}/{task_key(g, s)}.log', 'a')
    env = dict(os.environ)
    env.update({'AREAEI_GEN': str(g), 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                'AREAEI_FBEST': 'cand_upper'})
    p = subprocess.Popen([PY_MODEL, '-u', f'{HERE}/driver_gen.py', s],
                         stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=HERE,
                         start_new_session=True)          # detached, daemon重启不连带杀
    return p


def beat(msg=''):
    try:
        os.makedirs(os.path.dirname(HEARTBEAT), exist_ok=True)
        open(HEARTBEAT, 'w').write(f'{time.strftime("%F %T")} pid={os.getpid()} {msg}\n')
    except Exception:
        pass


def main():
    global MAX_PAR
    dry = '--dry' in sys.argv
    if '--par' in sys.argv:
        MAX_PAR = int(sys.argv[sys.argv.index('--par') + 1])
    gens = [int(x) for x in sys.argv[sys.argv.index('--gens') + 1].split(',')] \
        if '--gens' in sys.argv else [30, 40, 50, 60]
    only_sets = sys.argv[sys.argv.index('--sets') + 1].split(',') if '--sets' in sys.argv else None

    queue = []
    for g in gens:
        for s in TASKS[g]:
            if only_sets and s not in only_sets:
                continue
            fin, why = is_finished(g, s)
            if fin:
                print(f'[skip] {task_key(g,s)} 已完成: {why}')
            else:
                queue.append((g, s))
                print(f'[queue] {task_key(g,s)}' + (f' ({why})' if why else ' (全新)'))
    print(f'\n队列 {len(queue)} 个任务, 并行 {MAX_PAR}, dry={dry}')
    if dry:
        return

    st = load_state()
    st['running'] = {}          # 清掉上个(可能已崩的) master 遗留的 running 僵尸记录
    save_state(st)
    running = {}                # key -> {'proc': Popen, 'gen', 'set', 'retries', 't0'}
    qi = 0
    beat('start')
    while qi < len(queue) or running:
        # 补足并行位
        while qi < len(queue) and len(running) < MAX_PAR:
            g, s = queue[qi]
            key = task_key(g, s)
            proc = launch(g, s)
            running[key] = {'proc': proc, 'gen': g, 'set': s, 'retries': 0, 't0': time.time()}
            st['running'][key] = {'pid': proc.pid, 't0': time.strftime('%H:%M:%S')}
            save_state(st)
            print(f'[{time.strftime("%H:%M:%S")}] 启动 {key} pid={proc.pid} '
                  f'(并行 {len(running)}/{MAX_PAR}, 队列剩 {len(queue)-qi-1})', flush=True)
            qi += 1
        time.sleep(30)
        beat(f'poll running={len(running)} qi={qi}/{len(queue)}')
        # 巡检 — 整体 try/except: 任何异常只打印, 绝不打死 master (05:15 ECHILD 教训)
        try:
            for key in list(running):
                info = running[key]
                g, s = info['gen'], info['set']
                fin, why = is_finished(g, s)
                alive = info['proc'].poll() is None      # Popen.poll(): 安全可重复
                if fin:
                    print(f'[{time.strftime("%H:%M:%S")}] ✓ {key} 完成: {why} '
                          f'({(time.time()-info["t0"])/60:.0f}min)', flush=True)
                    st['done'][key] = why
                    st['running'].pop(key, None)
                    save_state(st)
                    del running[key]
                elif not alive:
                    # 进程退出但未达完成判据 → 失败
                    rc = info['proc'].returncode
                    if info['retries'] < 1:
                        print(f'[{time.strftime("%H:%M:%S")}] ✗ {key} 进程退出(rc={rc})未完成, '
                              f'重试1次(从checkpoint续)', flush=True)
                        proc = launch(g, s)
                        info.update(proc=proc, retries=1, t0=time.time())
                        st['running'][key] = {'pid': proc.pid, 't0': time.strftime('%H:%M:%S'),
                                              'retry': 1}
                        save_state(st)
                    else:
                        print(f'[{time.strftime("%H:%M:%S")}] ✗✗ {key} 二次失败(rc={rc}), '
                              f'记FAILED跳过', flush=True)
                        st['failed'][key] = f'rc={rc} @{time.strftime("%H:%M:%S")}'
                        st['running'].pop(key, None)
                        save_state(st)
                        del running[key]
        except Exception as e:
            print(f'[{time.strftime("%H:%M:%S")}] ⚠ 巡检异常(不致命, 下轮重试): {e!r}',
                  flush=True)
        # 进度快照
        if running:
            prog = []
            for key, info in running.items():
                _, why = is_finished(info['gen'], info['set'])
                prog.append(f'{key}:{(why or "?")}({(time.time()-info["t0"])/60:.0f}m)')
            print(f'[{time.strftime("%H:%M:%S")}] 并行中: ' + ' | '.join(prog), flush=True)

    print(f'\n===== 批量完成 =====')
    print(f'done={len(st["done"])} failed={len(st["failed"])}')
    for k, v in st['failed'].items():
        print(f'  FAILED: {k} {v}')
    save_state(st)
    beat('ALL DONE')


if __name__ == '__main__':
    main()
