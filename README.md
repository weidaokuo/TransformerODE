# Uncertainty-aware inverse design of Nb-based refractory alloys by learning deformation trajectories
## Overview
Designing high-strength Nb-based refractory alloys with adequate room-temperature ductility remains challenging due to the complexity of composition–processing–property relationships and the limited availability of experimental data. Conventional machine-learning approaches predominantly rely on scalar mechanical properties, such as yield strength and ultimate tensile strength, which cannot fully capture the evolution of mechanical responses during deformation.
This work presents an uncertainty-aware inverse design framework that learns complete stress–strain trajectories and their strain-dependent predictive uncertainty. A conditional Transformer–ODE model represents deformation as continuous state evolution in strain space, enabling the prediction of complete stress–strain curves rather than isolated mechanical properties. The resulting trajectory-level uncertainty is incorporated into a curve-pointwise expected improvement (CPEI) acquisition strategy to guide candidate selection by balancing predicted mechanical performance and uncertainty.
The following sections describe the software environment, installation requirements, and instructions for running the framework. Together, these resources provide the foundation for reproducing the proposed deformation-trajectory learning and uncertainty-aware inverse design workflow.
## Environment Setup
The runtime environment is Python 3.11.15 (macOS). Training runs on CPU only; no GPU / CUDA / MPS is required. To avoid libomp conflicts when xgboost and torch are loaded in the same process, the project uses two separate virtual environments: the feature environment handles descriptor computation and synthetic data generation, while the model environment handles torch model training and xgboost feature selection.
### Environment A (model)
Environment A (model) is used for ODE model training and inference, active learning iterations, and XGBoost-based feature screening.
| libraries | version |
|---|---|
| torch | 2.13.0 |
| torchdiffeq | 0.2.5 |
| numpy | 2.4.6 |
| pandas | 3.0.5 |
| scipy | 1.17.1 |
| xgboost | 3.2.0 |
| scikit-learn | 1.9.0 |
| matplotlib | 3.11.1 |
| openpyxl | 3.1.5 |
| joblib | 1.5.3 |

```bash
python3.11 -m venv .venv-model
.venv-model/bin/pip install torch==2.13.0 torchdiffeq==0.2.5 numpy==2.4.6 \
    pandas==3.0.5 scipy==1.17.1 xgboost==3.2.0 scikit-learn==1.9.0 \
    matplotlib==3.11.1 openpyxl==3.1.5 joblib==1.5.3
```
### Environment B
Environment B (feature) is used for matminer-based descriptor computation and synthetic data generation.
| libraries | version |
|---|---|
| matminer | 0.10.1 |
| pymatgen | 2026.5.4 |
| numpy | 2.4.6 |
| pandas | 2.3.3 |
| scipy | 1.17.1 |
| scikit-learn | 1.9.0 |
| tqdm | 4.70.0 |
| joblib | 1.5.3 |

```bash
python3.11 -m venv .venv-feature
.venv-feature/bin/pip install matminer==0.10.1 pymatgen==2026.5.4 numpy==2.4.6 \
    pandas==2.3.3 scipy==1.17.1 scikit-learn==1.9.0 tqdm==4.70.0 joblib==1.5.3
```
## Data and Model Validation
All commands assume execution from the project root (TransformerODE/). Define $M=.venv-model/bin/python and $F=.venv-feature/bin/python (create the environments as described above; use your own paths if different).
### Data 
1、Synthesis data
```bash
$F code/synthesis/step1e_physical_noisy_200.py
```
Input: code/voce_km_latent/results/ml_final_two_tier.pkl, code/obtain_feature.py + code/Args.py (matminer 13-dimensional descriptors).
Output: code/synthesis/data200/{latent_curves.jsonl, alloy_summary_200.csv, curves/, true_curves/}.
2、experiment data
```bash
$M code/train_last.py data/real_47/train_data_47.jsonl \
    data/real_47/stress_mean_std_47.json models/exp_47_repro
```
lr: 7e-4, weight_decay: 1e-5, cosine annealing scheduler, gradient clipping norm=0.5, 10000 epochs, checkpoint every 1000 epochs (early stopping can be used to shorten runtime).
### Model Validation
1、Active learning iteration on synthetic data
```bash
cd al_200_experiments/experiment60/scripts
$M al200_runner_v60.py bi013 cpei        # use: <setname> <cpei|det> [max_rounds=25]
$M al200_runner_v60.py bi013 det
```

2、Experimental data iteration
End-to-end alloy validation：
```bash
cd iterations
$F build_train53_last.py       
$M train_full_49_iter1.py       
$M ../code/train_last.py train_data_53.jsonl stress_mean_std_53.json checkpoints_53_repro                              
# End-to-end alloy example: composition → features → predicted curve
cd ../example_final_alloy
$F extract_final_alloy_features.py            # Environment B
$M predict_final_alloy.py                     # Environment A
```













