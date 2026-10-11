"""
Synthetic data generation + baseline ML models for predicting crude-oil
preheat-train fouling rate, physically grounded in the Ebert-Panchal (1995/1997)
threshold fouling model:

    dRf/dt = alpha * fouling_propensity * Re^beta * exp(-Ea / (R * Tf))  -  gamma * tau_w

  deposition term (Arrhenius reaction kinetics, weakened by higher Re / shorter
  residence time) minus removal term (proportional to wall shear stress).
  Below a "threshold" shear stress for given thermal conditions, net fouling
  rate approaches zero or goes negative -> no net fouling.

Parameter magnitudes (alpha, Ea, gamma) are literature-informed illustrative
values consistent with published ranges for crude oil preheat trains, NOT an
exact reproduction of one specific paper's fitted regression constants -- this
is stated explicitly in the paper's methodology/limitations.

v2 revision: Reynolds number and wall shear stress are no longer sampled
independently. In a real preheat-train tube, both derive from the same
underlying flow (velocity, fluid density/viscosity, tube diameter), so
independent sampling permitted physically inconsistent combinations (e.g.
high Re with low shear). Here Re and tau_w are computed from shared primitive
variables -- velocity, tube diameter, crude density, and a temperature-
dependent viscosity -- so they now co-vary the way real operating conditions
do. Viscosity uses an Andrade/Arrhenius-type liquid viscosity-temperature
law, calibrated to the order-of-magnitude drop (~10x) reported for real crude
oil across preheat-train-relevant temperatures.
"""

import os
os.makedirs("figures", exist_ok=True)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

np.random.seed(42)
N = 3000
R_GAS = 8.314  # J/(mol K)

# ---- 1. Sample PRIMITIVE physical variables, then derive Re and tau_w ----
Tf = np.random.uniform(400, 620, N)                 # film temperature, K (~127-347 C)
v = np.random.uniform(0.6, 2.5, N)                  # tube-side velocity, m/s (standard shell&tube design range)
D = np.random.uniform(0.016, 0.025, N)              # tube ID, m (~5/8"-1", standard tube sizes)
rho = np.random.uniform(800, 900, N)                # crude density, kg/m3 (light-medium crude range)
fouling_propensity = np.random.uniform(0.5, 1.5, N) # crude-blend fouling propensity (1.0 = reference crude)

# Temperature-dependent viscosity (Andrade/Arrhenius form), calibrated to
# ~4 cP at 400 K down to ~0.4 cP at 620 K -- consistent with the order-of-
# magnitude viscosity drop reported for crude oil over this temperature span
mu0, Ev = 6.076e-6, 21_589  # Pa.s, J/mol
mu = mu0 * np.exp(Ev / (R_GAS * Tf))

Re = rho * v * D / mu                                # Reynolds number -- now derived, not sampled
f_darcy = 0.316 * Re ** -0.25                         # Blasius friction factor (turbulent, smooth tube)
tau_w = f_darcy * rho * v ** 2 / 8                    # wall shear stress, Pa -- now derived, not sampled

# ---- 2. Ebert-Panchal-style parameters (illustrative magnitudes) ----
alpha = 3.0e7
beta = -0.75          # Panchal: -0.66, Polley: -0.8 -> use a mid-range value
Ea = 60_000            # J/mol, typical crude fouling reaction range
gamma = 4.0e-4

deposition = alpha * fouling_propensity * (Re ** beta) * np.exp(-Ea / (R_GAS * Tf))
removal = gamma * tau_w
fouling_rate = np.maximum(deposition - removal, 0.0)  # clean physical floor: no net fouling once removal >= deposition

noise = np.random.normal(0, 0.08 * np.abs(fouling_rate) + 2e-6, N)
fouling_rate_obs = np.maximum(fouling_rate + noise, 0.0)  # observed fouling rate can't be negative either

df = pd.DataFrame({
    "reynolds_number": Re,
    "film_temperature_K": Tf,
    "wall_shear_stress_Pa": tau_w,
    "fouling_propensity_index": fouling_propensity,
    "velocity_m_s": v,
    "tube_diameter_m": D,
    "density_kg_m3": rho,
    "fouling_rate_m2K_per_kWh": fouling_rate_obs,
})
df.to_csv("synthetic_fouling_data.csv", index=False)

print("=== Dataset summary ===")
pd.set_option("display.max_columns", None, "display.width", 160)
print(df.describe().round(6))
print(f"\nZero-fouling observations (fouling_rate == 0): {(df['fouling_rate_m2K_per_kWh'] == 0).sum()} / {N} "
      f"({(df['fouling_rate_m2K_per_kWh'] == 0).mean()*100:.1f}%)")
print()

# ---- 3. Train/test baseline models ----
X = df[["reynolds_number", "film_temperature_K", "wall_shear_stress_Pa", "fouling_propensity_index"]]
y = df["fouling_rate_m2K_per_kWh"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Fouling rate is right-skewed with a long tail -> scale the target too for the
# gradient-/margin-based models (MLP, SVR). Tree models and OLS don't need it.
y_scaler = StandardScaler()
y_train_s = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()

models = {
    "Linear Regression": (LinearRegression(), True, False),
    "Random Forest": (RandomForestRegressor(n_estimators=300, random_state=42), False, False),
    "MLP (ANN)": (MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=5000,
                                early_stopping=False, random_state=42), True, True),
    "SVR": (SVR(kernel="rbf", C=10, epsilon=0.01), True, True),
}

print("=== Model performance (test set) ===")
results = {}
for name, (model, needs_x_scaling, needs_y_scaling) in models.items():
    Xtr, Xte = (X_train_s, X_test_s) if needs_x_scaling else (X_train, X_test)
    ytr = y_train_s if needs_y_scaling else y_train
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    if needs_y_scaling:
        pred = y_scaler.inverse_transform(pred.reshape(-1, 1)).ravel()
    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    results[name] = {"R2": r2, "RMSE": rmse, "MAE": mae}
    print(f"{name:20s}  R2={r2:.4f}  RMSE={rmse:.6f}  MAE={mae:.6f}")

print()
print("=== Random Forest feature importances ===")
rf = models["Random Forest"][0]
for feat, imp in sorted(zip(X.columns, rf.feature_importances_), key=lambda x: -x[1]):
    print(f"{feat:28s} {imp:.4f}")

# ---- 4. Results figure: predicted vs actual (best model) + feature importance ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

best_name = max(results, key=lambda k: results[k]["R2"])
best_model, needs_x_scaling, needs_y_scaling = models[best_name]
Xte = X_test_s if needs_x_scaling else X_test
pred_best = best_model.predict(Xte)
if needs_y_scaling:
    pred_best = y_scaler.inverse_transform(pred_best.reshape(-1, 1)).ravel()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].scatter(y_test, pred_best, alpha=0.35, s=14, color="#2E5EAA")
lims = [min(y_test.min(), pred_best.min()), max(y_test.max(), pred_best.max())]
axes[0].plot(lims, lims, "r--", linewidth=1.2, label="Ideal (y = x)")
axes[0].set_xlabel("Actual fouling rate (m$^2$K/kWh)")
axes[0].set_ylabel("Predicted fouling rate (m$^2$K/kWh)")
axes[0].set_title(f"{best_name}: predicted vs. actual (R$^2$={results[best_name]['R2']:.3f})")
axes[0].legend()

feats, imps = zip(*sorted(zip(X.columns, rf.feature_importances_), key=lambda x: x[1]))
axes[1].barh(feats, imps, color="#2E5EAA")
axes[1].set_xlabel("Relative importance")
axes[1].set_title("Random Forest feature importance")

plt.tight_layout()
plt.savefig("figures/fouling_results.png", dpi=160)
print("\nSaved figure to fouling_results.png")

# ---- 5. Permutation importance (cross-check on RF's built-in importance) ----
print()
print("=== Permutation importance (Random Forest, test set) ===")
perm = permutation_importance(rf, X_test, y_test, n_repeats=20, random_state=42)
for feat, imp, sd in sorted(zip(X.columns, perm.importances_mean, perm.importances_std), key=lambda x: -x[1]):
    print(f"{feat:28s} {imp:.4f} +/- {sd:.4f}")

# ---- 6. Physics-consistency check ----
# NOTE (revised): an earlier version of this check swept one variable at a time
# with all others held at a single fixed point (the median). That is a check
# against ONE arbitrary background condition, not the model's behaviour in
# general -- and it's exactly what produced a plot showing "expected: increase"
# next to a curve that dips before it rises. The check below instead sweeps
# each variable across many different, randomly chosen background conditions
# and reports what fraction actually obey the expected direction. See section
# 12 for the combined quantified test + figure.
checks = {
    "film_temperature_K": "increase",
    "fouling_propensity_index": "increase",
    "reynolds_number": "decrease",
    "wall_shear_stress_Pa": "decrease",
}

# ---- 7. 5-fold cross-validation (scaling done fold-safe via Pipeline/TTR) ----
print()
print("=== 5-fold cross-validation (mean +/- SD across folds) ===")
cv_pipelines = {
    "Linear Regression": Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
    "MLP (ANN)": TransformedTargetRegressor(
        regressor=Pipeline([("scale", StandardScaler()),
                             ("model", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=5000, random_state=42))]),
        transformer=StandardScaler()),
    "SVR": TransformedTargetRegressor(
        regressor=Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="rbf", C=10, epsilon=0.01))]),
        transformer=StandardScaler()),
}
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_summary = {}
for name, pipe in cv_pipelines.items():
    scores = cross_validate(pipe, X, y, cv=kf,
                             scoring={"r2": "r2",
                                      "rmse": lambda est, Xv, yv: -np.sqrt(mean_squared_error(yv, est.predict(Xv))),
                                      "mae": "neg_mean_absolute_error"})
    r2m, r2s = scores["test_r2"].mean(), scores["test_r2"].std()
    rmsem, rmses = -scores["test_rmse"].mean(), scores["test_rmse"].std()
    maem, maes = -scores["test_mae"].mean(), scores["test_mae"].std()
    cv_summary[name] = (r2m, r2s, rmsem, rmses, maem, maes)
    print(f"{name:20s}  R2={r2m:.4f}+/-{r2s:.4f}  RMSE={rmsem:.6f}+/-{rmses:.6f}  MAE={maem:.6f}+/-{maes:.6f}")

# ---- 8. Generalization test: train on cooler conditions, test on an unseen  ----
#         hotter regime the model never saw during training
print()
print("=== Generalization test: train on Tf < 75th pct, test on hottest quartile (unseen) ===")
cutoff = df["film_temperature_K"].quantile(0.75)
train_mask = df["film_temperature_K"] < cutoff
Xg_train, yg_train = X[train_mask], y[train_mask]
Xg_test, yg_test = X[~train_mask], y[~train_mask]
print(f"Train range: {df.loc[train_mask,'film_temperature_K'].min():.0f}-{df.loc[train_mask,'film_temperature_K'].max():.0f} K "
      f"({train_mask.sum()} pts)  |  Held-out test range: {df.loc[~train_mask,'film_temperature_K'].min():.0f}-"
      f"{df.loc[~train_mask,'film_temperature_K'].max():.0f} K ({(~train_mask).sum()} pts)")
for name, pipe in cv_pipelines.items():
    pipe.fit(Xg_train, yg_train)
    pred_g = pipe.predict(Xg_test)
    r2g = r2_score(yg_test, pred_g)
    rmseg = np.sqrt(mean_squared_error(yg_test, pred_g))
    maeg = mean_absolute_error(yg_test, pred_g)
    print(f"{name:20s}  R2={r2g:.4f}  RMSE={rmseg:.6f}  MAE={maeg:.6f}   (vs. random-split R2={results[name]['R2']:.4f})")

# ---- 9. Residual plot for the best model (random-split test set) ----
residuals = y_test.values - pred_best
fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.5))
axes2[0].scatter(pred_best, residuals, alpha=0.35, s=14, color="#B23A48")
axes2[0].axhline(0, color="black", linewidth=1, linestyle="--")
axes2[0].set_xlabel("Predicted fouling rate")
axes2[0].set_ylabel("Residual (actual - predicted)")
axes2[0].set_title(f"{best_name}: residuals vs. predicted")

axes2[1].hist(residuals, bins=40, color="#B23A48", alpha=0.8)
axes2[1].set_xlabel("Residual")
axes2[1].set_ylabel("Count")
axes2[1].set_title("Residual distribution")

plt.tight_layout()
plt.savefig("figures/fouling_residuals.png", dpi=160)
print("\nSaved residual figure to fouling_residuals.png")

# ---- 10. Feature-set comparison: derived (Re, tau_w) vs. primitive (v, D, rho) ----
# Same train/test rows (via .loc on the same index) as Config A, different columns,
# so this isolates the effect of feature engineering rather than the data split.
print()
print("=== Feature-set comparison: Config A (derived) vs. Config B (primitive) ===")

def fit_score_all_models(Xtr_raw, Xte_raw, ytr_raw, yte_raw):
    sc = StandardScaler().fit(Xtr_raw)
    Xtr_s, Xte_s = sc.transform(Xtr_raw), sc.transform(Xte_raw)
    ysc = StandardScaler().fit(ytr_raw.values.reshape(-1, 1))
    ytr_s = ysc.transform(ytr_raw.values.reshape(-1, 1)).ravel()

    out = {}
    lr = LinearRegression().fit(Xtr_s, ytr_raw)
    out["Linear Regression"] = r2_score(yte_raw, lr.predict(Xte_s))

    rf_c = RandomForestRegressor(n_estimators=300, random_state=42).fit(Xtr_raw, ytr_raw)
    out["Random Forest"] = r2_score(yte_raw, rf_c.predict(Xte_raw))

    mlp_c = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=5000, random_state=42).fit(Xtr_s, ytr_s)
    out["MLP (ANN)"] = r2_score(yte_raw, ysc.inverse_transform(mlp_c.predict(Xte_s).reshape(-1, 1)).ravel())

    svr_c = SVR(kernel="rbf", C=10, epsilon=0.01).fit(Xtr_s, ytr_s)
    out["SVR"] = r2_score(yte_raw, ysc.inverse_transform(svr_c.predict(Xte_s).reshape(-1, 1)).ravel())
    return out

X_primitive = df[["film_temperature_K", "velocity_m_s", "tube_diameter_m", "density_kg_m3", "fouling_propensity_index"]]
Xp_train, Xp_test = X_primitive.loc[X_train.index], X_primitive.loc[X_test.index]

results_A = fit_score_all_models(X_train, X_test, y_train, y_test)
results_B = fit_score_all_models(Xp_train, Xp_test, y_train, y_test)

print(f"{'Model':20s} {'A: derived (Re, tau_w)':>24s} {'B: primitive (v, D, rho)':>26s}")
for name in ["Linear Regression", "Random Forest", "MLP (ANN)", "SVR"]:
    print(f"{name:20s} {results_A[name]:24.4f} {results_B[name]:26.4f}")

# ---- 11. Figure: model comparison (test R2, CV mean+/-SD, generalization R2) ----
model_order = ["Linear Regression", "Random Forest", "MLP (ANN)", "SVR"]
test_r2 = [results[n]["R2"] for n in model_order]
cv_r2 = [cv_summary[n][0] for n in model_order]
cv_sd = [cv_summary[n][1] for n in model_order]

fig3, ax3 = plt.subplots(figsize=(7, 4.5))
xpos = np.arange(len(model_order))
ax3.bar(xpos - 0.2, test_r2, width=0.4, label="Random-split test R2", color="#2E5EAA")
ax3.bar(xpos + 0.2, cv_r2, width=0.4, yerr=cv_sd, capsize=4, label="5-fold CV R2 (mean +/- SD)", color="#6FA8DC")
ax3.set_xticks(xpos)
ax3.set_xticklabels(model_order, rotation=15)
ax3.set_ylabel("R2")
ax3.set_title("Model comparison: test split vs. cross-validation")
ax3.legend()
plt.tight_layout()
plt.savefig("figures/fouling_model_comparison.png", dpi=160)
print("\nSaved figure to fouling_model_comparison.png")

# ---- 12. Robust physics-consistency test: many background conditions, not one ----
# For each variable, sample N_BG realistic background rows (real combinations of
# the OTHER variables, not just the median), sweep the variable of interest across
# each, and quantify how often the model's prediction (a) nets in the expected
# direction end-to-end, and (b) moves the expected direction at each local step.
print()
print(f"=== Robust physics-consistency test ({best_name}, N_BG background conditions) ===")
rng = np.random.default_rng(42)
N_BG = 500
bg_rows = X.iloc[rng.choice(len(X), size=N_BG, replace=False)].reset_index(drop=True)
plot_idx = set(rng.choice(N_BG, size=25, replace=False))

fig4, axes4 = plt.subplots(1, 4, figsize=(16, 4))
robust_summary = {}
for ax, (feat, expected) in zip(axes4, checks.items()):
    grid = np.linspace(X[feat].quantile(0.05), X[feat].quantile(0.95), 15)
    endpoint_matches, local_matches, local_total = 0, 0, 0

    for i in range(N_BG):
        curve_rows = pd.DataFrame([bg_rows.iloc[i].values] * len(grid), columns=X.columns)
        curve_rows[feat] = grid
        Xs2 = scaler.transform(curve_rows) if needs_x_scaling else curve_rows
        pred_curve = best_model.predict(Xs2)
        if needs_y_scaling:
            pred_curve = y_scaler.inverse_transform(pred_curve.reshape(-1, 1)).ravel()

        if i in plot_idx:
            ax.plot(grid, pred_curve, color="#2E5EAA", alpha=0.3, linewidth=1)

        if (pred_curve[-1] > pred_curve[0]) == (expected == "increase"):
            endpoint_matches += 1
        diffs = np.diff(pred_curve)
        local_matches += (diffs >= 0).sum() if expected == "increase" else (diffs <= 0).sum()
        local_total += len(diffs)

    endpoint_pct = 100 * endpoint_matches / N_BG
    local_pct = 100 * local_matches / local_total
    robust_summary[feat] = (endpoint_pct, local_pct)
    print(f"{feat:28s} expected={expected:8s}  net-direction consistent: {endpoint_pct:5.1f}% of {N_BG} backgrounds"
          f"  |  step-wise consistent: {local_pct:5.1f}% of {local_total} steps")

    ax.set_xlabel(feat)
    ax.set_ylabel("Predicted fouling rate")
    ax.set_title(f"Expected: {expected}\n{endpoint_pct:.0f}% net-consistent, {local_pct:.0f}% step-consistent")

plt.suptitle(f"Physics-consistency across {N_BG} random background conditions ({best_name}) -- "
             f"25 sampled curves shown per panel")
plt.tight_layout()
plt.savefig("figures/fouling_physics_sweeps.png", dpi=160)
print("Saved updated figure to fouling_physics_sweeps.png")

# ---- 13. Figure: generalization comparison (random-split vs. holdout R2) ----
gen_r2 = []
for name, pipe in cv_pipelines.items():
    pipe.fit(Xg_train, yg_train)
    gen_r2.append(r2_score(yg_test, pipe.predict(Xg_test)))

fig5, ax5 = plt.subplots(figsize=(7, 4.5))
ax5.bar(xpos - 0.2, test_r2, width=0.4, label="Random-split R2", color="#2E5EAA")
ax5.bar(xpos + 0.2, gen_r2, width=0.4, label="Holdout (unseen hot regime) R2", color="#B23A48")
ax5.axhline(0, color="black", linewidth=0.8)
ax5.set_xticks(xpos)
ax5.set_xticklabels(model_order, rotation=15)
ax5.set_ylabel("R2")
ax5.set_title("Generalization gap: random-split vs. extrapolation to unseen conditions")
ax5.legend()
plt.tight_layout()
plt.savefig("figures/fouling_generalization.png", dpi=160)
print("Saved figure to fouling_generalization.png")
