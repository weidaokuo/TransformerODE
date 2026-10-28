import sys, os, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))          
PKG = os.path.dirname(HERE)                                 
ITER = os.path.join(PKG, 'iterations')
sys.path.insert(0, PKG)
sys.path.insert(0, ITER)
import iteration_common as IC                               

LABEL = 'Nb-20Ta-15Hf-10W-0.07C'
CKPT_DIR = os.path.join(ITER, 'checkpoints_51_iter2')        
SMS_PATH = os.path.join(ITER, 'stress_mean_std_51_iter2.json')
COND_JSON = os.path.join(HERE, 'final_alloy_cond18.json')
REF_FALLBACK = os.path.join(ITER, 'ref_features_51iter2.json')
REAL6 = os.path.join(ITER, 'alloy_real_exact.csv')
OUT_CSV = os.path.join(HERE, f'prediction_{LABEL}.csv')
OUT_PNG = os.path.join(HERE, f'prediction_{LABEL}.png')


def load_cond():
    if os.path.exists(COND_JSON):
        rec = json.load(open(COND_JSON))
        return rec['cond_18'], rec.get('elong'), rec.get('label', LABEL), 'final_alloy_cond18.json'
    refs = {r['tag']: r for r in json.load(open(REF_FALLBACK))}
    r6 = refs['(6)']
    return r6['cond_18'], r6['elong'], r6['label'], 'ref_features_51iter2.json (6)'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cond18, elong, label, src = load_cond()
    elong = float(elong) if elong else 33.3
    print(f'[predict] alloy: {label}  (Source of conditions: {src})')
    print(f'[predict] Predict the upper limit of strain elong = {elong}%')

    model, ckpt, ckpt_path = IC.load_model(CKPT_DIR)
    print(f'[predict] model: {os.path.basename(ckpt_path)} (epoch {ckpt["epoch"]}, loss={ckpt.get("loss")})')
    sms = json.load(open(SMS_PATH))
    gm, gs = float(sms['global_mean_stress']), float(sms['global_std_stress'])
    print(f'[predict] inverse standardization: global_mean={gm:.4f}, global_std={gs:.4f}')

    # MC50 sampling prediction (with the same caliber as the iteration line)
    grid, mu_s, s_s, mu_t, s_t = IC.infer(model, cond18, elong, step=0.1, stress_stats=(gm, gs))
    lo, hi = mu_s - 2 * s_s, mu_s + 2 * s_s
    print(f'[predict] MC50 complete: {len(grid)} strain point')
    print(f'[predict] peak stress = {mu_s.max():.1f} ± {s_s[np.argmax(mu_s)]:.1f} MPa '
          f'(95%区间 [{lo[np.argmax(mu_s)]:.0f}, {hi[np.argmax(mu_s)]:.0f}])')

    #---- Output CSV (with uncertainty intervals) ----
    pd.DataFrame({
        'strain_pct': grid, 'mu_sigma_MPa': mu_s, 'sigma_MPa': s_s,
        'ci95_lower_MPa': lo, 'ci95_upper_MPa': hi,
        'mu_theta_MPa': mu_t, 'sigma_theta_MPa': s_t,
    }).to_csv(OUT_CSV, index=False, float_format='%.6f')
    print(f'[predict] forecasted statement → {OUT_CSV}')

    # ---- matplotlib: prediction + ±2σ band + observed overlay ----
    fig, ax = plt.subplots(figsize=(9, 6.5))
    has_real = os.path.exists(REAL6)
    if has_real:
        real = pd.read_csv(REAL6)
        ax.plot(real['strain_pct'], real['stress_MPa'], 'b-', lw=2.2,
                label='Experimental (real_6)')
    ax.fill_between(grid, lo, hi, color='red', alpha=0.18, label='Pred mean ±2σ (95% CI)')
    ax.plot(grid, mu_s, 'r--', lw=2.2, label='Pred mean (51-curve model, MC50)')
    ax.set_xlabel('Strain (%)', fontsize=12)
    ax.set_ylabel('Stress (MPa)', fontsize=12)
    ax.set_title(f'Final alloy {label}\n'
                 f'prediction with uncertainty (51-curve model, MC50: 10 noise × 5 expert)',
                 fontsize=12)
    pk = int(np.argmax(mu_s))
    ax.annotate(f'peak {mu_s[pk]:.0f}±{s_s[pk]:.0f} MPa',
                xy=(grid[pk], mu_s[pk]), xytext=(grid[pk] * 0.55, mu_s[pk] * 1.06),
                arrowprops=dict(arrowstyle='->', color='k', lw=0.9), fontsize=10)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    print(f'[predict] picture → {OUT_PNG}')
    if not has_real:
        print('[predict] (real_6.csv)')


if __name__ == '__main__':
    main()
