# Uncertainty-aware inverse design of Nb-based refractory alloys by learning deformation trajectories
## Overview
Designing high-strength Nb-based refractory alloys with adequate room-temperature ductility remains challenging due to the complexity of composition–processing–property relationships and the limited availability of experimental data. Conventional machine-learning approaches predominantly rely on scalar mechanical properties, such as yield strength and ultimate tensile strength, which cannot fully capture the evolution of mechanical responses during deformation.
This work presents an uncertainty-aware inverse design framework that learns complete stress–strain trajectories and their strain-dependent predictive uncertainty. A conditional Transformer–ODE model represents deformation as continuous state evolution in strain space, enabling the prediction of complete stress–strain curves rather than isolated mechanical properties. The resulting trajectory-level uncertainty is incorporated into a curve-pointwise expected improvement (CPEI) acquisition strategy to guide candidate selection by balancing predicted mechanical performance and uncertainty.
The following sections describe the software environment, installation requirements, and instructions for running the framework. Together, these resources provide the foundation for reproducing the proposed deformation-trajectory learning and uncertainty-aware inverse design workflow.
## Environment Setup
This project is based on Python 3.11. The following lists the libraries and version numbers required to run this project.
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
| tqdm | 4.70.0 |
| matminer | 0.10.1 |
| pymatgen | 2026.5.4 |







