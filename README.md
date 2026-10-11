# Predicting Crude Oil Preheat Train Fouling with Machine Learning

A physics-informed comparative study of four regressors (Linear Regression, Random Forest, SVR, MLP) predicting crude-oil preheat train fouling rate. Full write-up: [`paper.pdf`](paper.pdf).

> **Important:** the data is **synthetic**, generated from an *Ebert-Panchal-style* deposition-minus-removal formulation with illustrative parameters. It is **not** plant data and not the exact Ebert-Panchal model. The study is about comparing ML methods under controlled, physically structured conditions.

## Key results (N = 3000, seed 42, 80/20 split)

| Model | Test R² | RMSE | 5-fold CV R² | Hot-regime holdout R² |
|---|---|---|---|---|
| Linear Regression | 0.7227 | 0.01282 | 0.7239 ± 0.0037 | -0.516 |
| Random Forest | 0.9778 | 0.00363 | 0.9748 ± 0.0035 | -0.028 |
| MLP (64,32) | 0.9883 | 0.00264 | 0.9860 ± 0.0025 | **0.750** |
| SVR (RBF) | **0.9893** | **0.00252** | **0.9870 ± 0.0020** | 0.552 |

**Takeaways**
- SVR and MLP win in interpolation; tree-based Random Forest cannot extrapolate: it collapses on the unseen high-temperature regime (train T_f < 566 K, test 566-620 K).
- Film temperature dominates permutation importance (1.67 vs. 0.25 for fouling propensity).
- Physics-consistency test (500 background points): the models respect the expected trend for temperature, fouling propensity and shear stress, but only about 57% of the time for Reynolds number. This is reported honestly in the paper.
- Derived features (Re, wall shear stress) give a small edge over primitive variables (v, D, ρ): R² 0.9893 vs 0.9876 for SVR.

Hyperparameters were reasonable defaults, **not** grid-searched.

## Reproduce

```bash
pip install -r requirements.txt
python fouling_ml.py
```

Regenerates the dataset (`synthetic_fouling_data.csv`), all metrics and the figures in `figures/`.

## Contents

- `fouling_ml.py`: data generation, models, CV, holdout test, permutation importance, physics test, figures
- `synthetic_fouling_data.csv`: the dataset used in the paper
- `figures/`: result plots
- `paper.pdf`: the full paper (about 13,000 words)

## Limitations / next steps
- Synthetic data only; validating on real or public fouling data is the obvious next step.
- No time-dependent (asymptotic) fouling or crude-composition effects.
- No hyperparameter tuning.

## Author
AJ, Chemical and Polymer Engineering, Lagos State University.
