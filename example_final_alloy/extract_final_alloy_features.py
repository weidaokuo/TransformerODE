import sys, os, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))          # example_final_alloy/
PKG = os.path.dirname(HERE)                                 # TransformerODE/ 包根
sys.path.insert(0, PKG)
from obtain_feature import ObtainFeatures                    # noqa: E402  (pymatgen/matminer)
from Args import args                                        # noqa: E402
import warnings
warnings.filterwarnings('ignore')

CSV_NAME = sys.argv[1] if len(sys.argv) > 1 else 'test.csv'
CSV_PATH = os.path.join(HERE, CSV_NAME)
FS_PATH = os.path.join(PKG, 'iterations', 'feature_stats_51_iter2.txt')
REAL6 = os.path.join(PKG, 'iterations', 'real_test.csv')
OUT = os.path.join(HERE, 'final_alloy_cond18.json')

EL = args.element_list          
PN = args.physic_names         


def load_feature_stats(path):
    stats = {}
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith('Feature') or '==' in ln or ln.startswith('-'):
                continue
            p = ln.rsplit(None, 2)
            if len(p) == 3:
                try:
                    stats[p[0]] = (float(p[1]), float(p[2]))
                except ValueError:
                    pass
    names = list(stats.keys())
    means = np.array([stats[n][0] for n in names])
    stds = np.array([stats[n][1] for n in names])
    return names, means, stds


def build_cond(comp, process_5, fs_path):
    names, means, stds = load_feature_stats(fs_path)
    cl = [comp.get(e, 0.0) for e in EL]
    ext = ObtainFeatures(EL, cl)
    key = ext.get_key_features()
    raw = np.zeros(18)
    raw[:5] = process_5
    for i, n in enumerate(PN):
        raw[5 + i] = key.get(n, 0.0)
    return ((raw - means) / stds).astype(np.float32).tolist(), key


def main():
    df = pd.read_csv(CSV_PATH)
    r = df.iloc[0]
    label = str(r['composition_label'])

    comp = {e: float(r[f'{e}_wt_pct']) for e in ['Nb', 'Ta', 'Hf', 'W', 'C'] if f'{e}_wt_pct' in r}
    process_5 = [float(r['process_temp_C']), float(r['process_time_h']),
                 float(r['process_cool_rate_C_s']), float(r['process_strain_rate_s-1']),
                 float(r['process_prestrain_pct'])]
    elong = float(r['elong_pct']) if 'elong_pct' in r and pd.notna(r['elong_pct']) else None
    if elong is None and os.path.exists(REAL6):
        elong = float(pd.read_csv(REAL6)['strain_pct'].max())

    cond_18, key = build_cond(comp, process_5, FS_PATH)
    rec = {'label': label, 'comp': comp, 'process_5': process_5, 'elong': elong,
           'cond_18': cond_18,
           'desc13': {n: float(key.get(n, 0.0)) for n in PN},
           'feature_stats': os.path.basename(FS_PATH)}
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)

    print(f'[extract] alloy: {label}')
    print(f'[extract] composition wt%: {comp}')
    print(f'[extract] processing: {process_5}  (temp, time, cool, strain_rate, prestrain)')
    print(f'[extract] elong = {elong}%')
    print(f'[extract] 13 Descriptors (Top 5)): {[round(key.get(n,0.0),4) for n in PN[:5]]}')
    print(f'[extract] cond_18(top 6): {[round(x,4) for x in cond_18[:6]]}')
    print(f'[extract] Standardization parameters: {os.path.basename(FS_PATH)}')
    print(f'[extract] save → {OUT}')


if __name__ == '__main__':
    main()
