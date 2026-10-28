#!/usr/bin/env python3
import csv
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
AL200 = os.path.dirname(HERE)                               
EXTRACT = AL200                                             

TASKS = {
 30: ['bi007', 'hg007', 'ht002', 'ht009', 'lx005', 'mi023', 'r005', 'r020', 'r033', 'st009'],
 40: ['bi004', 'cv007', 'ht002', 'lo002', 'lo007', 'lo019', 'lo023', 'lx006', 'mi008', 'r032'],
 50: ['bi002', 'cl002', 'cl013', 'lo008', 'lo009', 'mi006', 'mi007', 'r049', 'st012', 'st014'],
 60: ['bi013', 'cl013', 'ht007', 'lo000', 'lo004', 'lx004', 'lx008', 'mi009', 'mi016', 'st000'],
}


four = {}
with open(f'{EXTRACT}/sets_and_iterations.csv') as f:
    for r in csv.DictReader(f):
        four[(int(r['gen_init']), r['set'])] = {
            'greedy': int(r['greedy_rounds']), 'ehvi': int(r['ehvi_rounds']),
            'cpei': int(r['cpei_rounds']), 'det': int(r['det_rounds'])}

rows = []
for g in (30, 40, 50, 60):
    for s in TASKS[g]:
        p = f'{HERE}/runs/gen{g}/{s}_areaei/state.json'
        stt = json.load(open(p))
        hf = {int(k): v for k, v in stt['hit_first'].items()}
        rd = stt['rounds_done']
        rows.append({
            'gen_init': g, 'set': s,
            'greedy': four[(g, s)]['greedy'], 'det': four[(g, s)]['det'],
            'ehvi': four[(g, s)]['ehvi'], 'cpei': four[(g, s)]['cpei'],
            'areaei_top1': hf.get(1), 'areaei_top3': hf.get(3),
            'areaei_top5': hf.get(5), 'areaei_top10': hf.get(10),
            'areaei_rounds_done': rd})

outp = f'{HERE}/five_way_comparison.csv'
with open(outp, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print('==== Five methods top1 first-hit round — per-set details (40 sets) ====')
print(f'{"gen":>4} {"set":<7} {"greedy":>6} {"det":>4} {"ehvi":>4} {"cpei":>4} {"areaEI":>6}')
for r in rows:
    print(f'{r["gen_init"]:>4} {r["set"]:<7} {r["greedy"]:>6} {r["det"]:>4} '
          f'{r["ehvi"]:>4} {r["cpei"]:>4} {r["areaei_top1"]:>6}')

print('\n==== Mean by generation ± sample std (ddof=1, n=10) ====')
print(f'{"Gen (init)":<10} {"greedy":>12} {"det":>12} {"ehvi":>12} {"cpei":>12} {"areaEI":>12}')
grand = {m: [] for m in ('greedy', 'det', 'ehvi', 'cpei', 'areaei_top1')}
for g in (30, 40, 50, 60):
    sub = [r for r in rows if r['gen_init'] == g]
    cells = []
    for m in ('greedy', 'det', 'ehvi', 'cpei', 'areaei_top1'):
        v = [r[m] for r in sub]
        grand[m] += v
        cells.append(f'{st.mean(v):.2f}±{st.stdev(v):.2f}')
    print(f'{g:<10} ' + ' '.join(f'{c:>12}' for c in cells))
cells = [f'{st.mean(grand[m]):.2f}±{st.stdev(grand[m]):.2f}'
         for m in ('greedy', 'det', 'ehvi', 'cpei', 'areaei_top1')]
print(f'{"All 40 sets":<9} ' + ' '.join(f'{c:>12}' for c in cells))

print('\n==== Additional statistics ====')
for thr in (1, 3, 5, 10):
    cnt = {m: sum(1 for r in rows if r[m] <= thr)
           for m in ('greedy', 'det', 'ehvi', 'cpei', 'areaei_top1')}
    print(f'Top1 hit within {thr} rounds (n/40): ' + '  '.join(f'{m}={cnt[m]}' for m in cnt))
med = {m: st.median([r[m] for r in rows])
       for m in ('greedy', 'det', 'ehvi', 'cpei', 'areaei_top1')}
print('Median: ' + '  '.join(f'{m}={med[m]:.0f}' for m in med))
mx = {m: max(r[m] for r in rows) for m in ('greedy', 'det', 'ehvi', 'cpei', 'areaei_top1')}
print('Worst set round: ' + '  '.join(f'{m}={mx[m]}' for m in mx))

print('\n==== areaEI all-tier first-hit mean (top1/3/5/10) ====')
for g in (30, 40, 50, 60):
    sub = [r for r in rows if r['gen_init'] == g]
    line = ' '.join(f'top{k}={st.mean([r[f"areaei_top{k}"] for r in sub]):.2f}'
                    for k in (1, 3, 5, 10))
    print(f'gen{g}: {line}')
line = ' '.join(f'top{k}={st.mean([r[f"areaei_top{k}"] for r in rows]):.2f}'
                for k in (1, 3, 5, 10))
print(f'All 40 sets: {line}')
print(f'\nCSV saved: {outp}')