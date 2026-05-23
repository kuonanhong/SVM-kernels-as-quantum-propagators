#!/usr/bin/env python
# coding: utf-8

# -----------------------------------------------------------------------------
# Copyright (c) 2025 Renata Wong
#
# This code is supplementary material for the research paper:
# "Support Vector Machine Kernels as Quantum Propagators"
#
# Revised experimental version for major revision:
# Added
#   [ADDED 1/5] Cross-validation statistics
#   [ADDED 2/5] Error bars / confidence intervals
#   [ADDED 3/5] Hyperparameter tuning details
#   [ADDED 4/5] More rigorous benchmarking
#   [ADDED 5/5] Cost analysis
#
# NOTE:
# - The original physical data generation logic is preserved as much as possible.
# - I also fixed a few obvious runtime issues in the original plotting code so the
#   revised scripts can execute end-to-end.
# -----------------------------------------------------------------------------

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid, RepeatedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import KFold

OUT_DIR = Path(".")
CV_METRICS_CSV = OUT_DIR / "quasicrystal_kernel_cv_metrics.csv"
HOLDOUT_METRICS_CSV = OUT_DIR / "quasicrystal_kernel_holdout_metrics.csv"
HYPERPARAM_JSON = OUT_DIR / "quasicrystal_kernel_best_params.json"
COST_CSV = OUT_DIR / "quasicrystal_kernel_cost_analysis.csv"
FIG_BAR = OUT_DIR / "quasicrystals_bargraph.png"
FIG_CURVE = OUT_DIR / "quasicrystals_rbf_vs_custom_curve.png"

RANDOM_STATE = 42
OUTER_CV = RepeatedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE)
INNER_CV_SPLITS = 4


# ==========================================
# ORIGINAL PHYSICS SIMULATION (preserved)
# ==========================================
def generate_fibonacci_transmission(n_samples=200, energy_range=(0, 3)):
    s0, s1 = "A", "AB"
    for _ in range(5):
        s0, s1 = s1, s1 + s0
    layers = s1

    energies = np.linspace(energy_range[0], energy_range[1], n_samples)
    transmission = []

    k_A = lambda E: np.sqrt(E + 0j)
    k_B = lambda E: np.sqrt(E - 1.5 + 0j)

    for E in energies:
        M = np.eye(2, dtype=complex)
        for layer in layers:
            k = k_A(E) if layer == "A" else k_B(E)
            P = np.array([[np.exp(1j * k), 0], [0, np.exp(-1j * k)]])
            S = np.array([[1 - 1j / k, -1j / k], [1j / k, 1 + 1j / k]]) if k != 0 else np.eye(2)
            M = np.dot(S, np.dot(P, M))
        Tval = 1.0 / (np.abs(M[1, 1]) ** 2)
        transmission.append(float(Tval))

    return energies.reshape(-1, 1), np.array(transmission)


# ==========================================
# ORIGINAL CUSTOM KERNEL (preserved)
# ==========================================
class JacksonChebyshevKernel:
    def __init__(self, degree=50, epsilon=0.01, jitter=1e-12):
        self.degree = degree
        self.epsilon = float(epsilon)
        self.jitter = float(jitter)
        self.scaler = MinMaxScaler(feature_range=(-1 + self.epsilon, 1 - self.epsilon))
        self.jackson_coeffs = self._compute_jackson_coeffs()

    def _compute_jackson_coeffs(self):
        N = self.degree
        n = np.arange(N + 1)
        term1 = (N - n + 1) * np.cos(np.pi * n / (N + 1))
        term2 = np.sin(np.pi * n / (N + 1)) / np.tan(np.pi / (N + 1))
        g_n = (term1 + term2) / (N + 1)
        return np.clip(g_n, 0.0, None)

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.scaler.fit(X)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        X_scaled = self.scaler.transform(X)

        # 關鍵修正：避免外插後落到 [-1, 1] 之外
        bound = 1.0 - self.epsilon
        X_scaled = np.clip(X_scaled, -bound, bound)

        features = []
        orders = np.arange(self.degree + 1)

        for i in range(X_scaled.shape[0]):
            x_val = float(X_scaled[i, 0])
            theta = np.arccos(x_val)
            cheby_polys = np.cos(orders * theta)
            weighted_features = cheby_polys * np.sqrt(self.jackson_coeffs)
            features.append(weighted_features)

        Phi = np.array(features, dtype=float)
        Phi = np.nan_to_num(Phi, nan=0.0, posinf=0.0, neginf=0.0)
        return Phi

    def compute_gram_matrix(self, X1, X2):
        Phi1 = self.transform(X1)
        Phi2 = self.transform(X2)
        K = Phi1 @ Phi2.T
        K = np.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)
        return K

    def spectral_clip(self, K):
        K = np.asarray(K, dtype=float)
        K = 0.5 * (K + K.T)  # enforce symmetry
        vals, vecs = np.linalg.eigh(K)
        vals = np.clip(vals, 0.0, None)
        K_psd = vecs @ np.diag(vals) @ vecs.T
        K_psd += self.jitter * np.eye(K_psd.shape[0])
        return np.nan_to_num(K_psd, nan=0.0, posinf=0.0, neginf=0.0)

def regression_summary(scores):
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    mean = float(np.mean(scores))
    std = float(np.std(scores, ddof=1)) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 1 else 0.0
    tcrit = float(t.ppf(0.975, df=n - 1)) if n > 1 else 0.0
    ci95 = tcrit * sem if n > 1 else 0.0
    return mean, std, ci95


# ==========================================
# [ADDED 3/5] Hyperparameter tuning details
# ==========================================
STANDARD_MODEL_GRIDS = {
    "linear": {"C": [0.1, 1, 10, 100], "epsilon": [0.001, 0.01, 0.05]},
    "rbf": {"C": [1, 10, 100], "gamma": ["scale", 0.1, 1.0], "epsilon": [0.001, 0.01, 0.05]},
    "poly": {
        "C": [1, 10, 100],
        "degree": [2, 3, 4],
        "gamma": ["scale", 0.1, 1.0],
        "coef0": [0.0, 1.0],
        "epsilon": [0.001, 0.01],
    },
    "sigmoid": {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", 0.01, 0.1, 1.0],
        "coef0": [-1.0, 0.0, 1.0],
        "epsilon": [0.001, 0.01],
    },
    "dummy_mean": {"strategy": ["mean"]},
    "random_forest": {"n_estimators": [200, 400], "max_depth": [None, 6, 12]},
}
CUSTOM_GRID = {"degree": [40, 50, 60, 75], "C": [1, 10, 100], "epsilon": [0.001, 0.01, 0.05]}


def build_standard_model(name, params):
    if name == "dummy_mean":
        return DummyRegressor(**params)
    if name == "random_forest":
        return RandomForestRegressor(random_state=RANDOM_STATE, **params)
    model = SVR(kernel=name, **params)
    return Pipeline([("scaler", StandardScaler()), ("model", model)])

def inner_splits(n_samples):
    idx = np.arange(n_samples)
    kf = KFold(n_splits=INNER_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    return list(kf.split(idx))

def tune_standard_model(X_train, y_train, model_name):
    best_score = np.inf
    best_params = None
    for params in ParameterGrid(STANDARD_MODEL_GRIDS[model_name]):
        model = build_standard_model(model_name, params)
        fold_mse = []
        for tr_idx, va_idx in inner_splits(len(X_train)):
            model.fit(X_train[tr_idx], y_train[tr_idx])
            y_pred = model.predict(X_train[va_idx])
            fold_mse.append(mean_squared_error(y_train[va_idx], y_pred))
        score = float(np.mean(fold_mse))
        if score < best_score:
            best_score = score
            best_params = params
    return best_params, best_score


def fit_predict_standard(X_train, y_train, X_test, model_name, params):
    model = build_standard_model(model_name, params)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_elapsed = time.perf_counter() - t0
    t1 = time.perf_counter()
    y_pred = model.predict(X_test)
    pred_elapsed = time.perf_counter() - t1
    return y_pred, fit_elapsed, pred_elapsed


def tune_custom_model(X_train, y_train):
    best_score = np.inf
    best_params = None
    for params in ParameterGrid(CUSTOM_GRID):
        fold_mse = []
        for tr_idx, va_idx in inner_splits(len(X_train)):
            X_tr = X_train[tr_idx]
            y_tr = y_train[tr_idx]
            X_va = X_train[va_idx]
            y_va = y_train[va_idx]

            kpm = JacksonChebyshevKernel(degree=params["degree"], epsilon=0.01)
            kpm.fit(X_tr)
            K_train = kpm.compute_gram_matrix(X_tr, X_tr)
            K_val = kpm.compute_gram_matrix(X_va, X_tr)
            K_train_psd = kpm.spectral_clip(K_train)

            if not np.isfinite(K_train_psd).all():
                raise ValueError("K_train_psd contains non-finite values.")
            if not np.isfinite(K_val).all():
                raise ValueError("K_val contains non-finite values.")

            model = SVR(kernel="precomputed", C=params["C"], epsilon=params["epsilon"])
            model.fit(K_train_psd, y_tr)
            y_pred = model.predict(K_val)
            fold_mse.append(mean_squared_error(y_va, y_pred))
        score = float(np.mean(fold_mse))
        if score < best_score:
            best_score = score
            best_params = params
    return best_params, best_score


def fit_predict_custom(X_train, y_train, X_test, params):
    t_kernel0 = time.perf_counter()
    kpm = JacksonChebyshevKernel(degree=params["degree"], epsilon=0.01)
    kpm.fit(X_train)
    K_train = kpm.compute_gram_matrix(X_train, X_train)
    K_test = kpm.compute_gram_matrix(X_test, X_train)
    K_train_psd = kpm.spectral_clip(K_train)

    if not np.isfinite(K_train_psd).all():
        raise ValueError("K_train_psd contains non-finite values.")
    if not np.isfinite(K_test).all():
        raise ValueError("K_test contains non-finite values.")
    kernel_build_elapsed = time.perf_counter() - t_kernel0

    model = SVR(kernel="precomputed", C=params["C"], epsilon=params["epsilon"])
    t0 = time.perf_counter()
    model.fit(K_train_psd, y_train)
    fit_elapsed = time.perf_counter() - t0
    t1 = time.perf_counter()
    y_pred = model.predict(K_test)
    pred_elapsed = time.perf_counter() - t1

    return y_pred, kernel_build_elapsed, fit_elapsed, pred_elapsed, K_train_psd.nbytes / (1024**2)


def main():
    X, y = generate_fibonacci_transmission(n_samples=300, energy_range=(0.1, 4.0))
    model_names = ["linear", "rbf", "poly", "custom", "sigmoid", "dummy_mean", "random_forest"]

    # [ADDED 1/5], [ADDED 2/5], [ADDED 3/5], [ADDED 5/5]
    cv_rows = []
    cost_rows = []
    best_param_history = {name: [] for name in model_names}

    for split_id, (tr_idx, te_idx) in enumerate(OUTER_CV.split(X, y), start=1):
        X_train, X_test = X[tr_idx], X[te_idx]
        y_train, y_test = y[tr_idx], y[te_idx]

        for name in model_names:
            if name == "custom":
                best_params, _ = tune_custom_model(X_train, y_train)
                best_param_history[name].append(best_params)

                y_pred, kernel_build_elapsed, fit_elapsed, pred_elapsed, mem_mb = fit_predict_custom(
                    X_train, y_train, X_test, best_params
                )
                cost_rows.append(
                    {
                        "Model": name,
                        "Split": split_id,
                        "KernelBuildTimeSec": kernel_build_elapsed,
                        "FitTimeSec": fit_elapsed,
                        "PredictTimeSec": pred_elapsed,
                        "ApproxKernelMemoryMB": mem_mb,
                    }
                )
            else:
                best_params, _ = tune_standard_model(X_train, y_train, name)
                best_param_history[name].append(best_params)

                y_pred, fit_elapsed, pred_elapsed = fit_predict_standard(
                    X_train, y_train, X_test, name, best_params
                )
                cost_rows.append(
                    {
                        "Model": name,
                        "Split": split_id,
                        "KernelBuildTimeSec": 0.0,
                        "FitTimeSec": fit_elapsed,
                        "PredictTimeSec": pred_elapsed,
                        "ApproxKernelMemoryMB": 0.0,
                    }
                )

            cv_rows.append(
                {
                    "Model": name,
                    "Split": split_id,
                    "MSE": mean_squared_error(y_test, y_pred),
                    "R2": r2_score(y_test, y_pred),
                }
            )

    cv_fold_df = pd.DataFrame(cv_rows)
    cost_df = pd.DataFrame(cost_rows)

    summary_rows = []
    for name in model_names:
        sub = cv_fold_df[cv_fold_df["Model"] == name]
        mse_mean, mse_std, mse_ci95 = regression_summary(sub["MSE"])
        r2_mean, r2_std, r2_ci95 = regression_summary(sub["R2"])
        sub_cost = cost_df[cost_df["Model"] == name]
        fit_mean, fit_std, fit_ci95 = regression_summary(sub_cost["FitTimeSec"])
        pred_mean, pred_std, pred_ci95 = regression_summary(sub_cost["PredictTimeSec"])
        kb_mean, kb_std, kb_ci95 = regression_summary(sub_cost["KernelBuildTimeSec"])
        mem_mean, mem_std, mem_ci95 = regression_summary(sub_cost["ApproxKernelMemoryMB"])

        summary_rows.append(
            {
                "Model": name,
                "CV_MSE_Mean": mse_mean,
                "CV_MSE_STD": mse_std,
                "CV_MSE_CI95": mse_ci95,
                "CV_R2_Mean": r2_mean,
                "CV_R2_STD": r2_std,
                "CV_R2_CI95": r2_ci95,
                "KernelBuildTime_MeanSec": kb_mean,
                "KernelBuildTime_STD": kb_std,
                "KernelBuildTime_CI95": kb_ci95,
                "FitTime_MeanSec": fit_mean,
                "FitTime_STD": fit_std,
                "FitTime_CI95": fit_ci95,
                "PredictTime_MeanSec": pred_mean,
                "PredictTime_STD": pred_std,
                "PredictTime_CI95": pred_ci95,
                "ApproxKernelMemoryMB_Mean": mem_mean,
                "ApproxKernelMemoryMB_STD": mem_std,
                "ApproxKernelMemoryMB_CI95": mem_ci95,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("CV_MSE_Mean")
    summary_df.to_csv(CV_METRICS_CSV, index=False)
    cost_df.to_csv(COST_CSV, index=False)

    with open(HYPERPARAM_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                name: [{k: str(v) for k, v in params.items()} for params in params_list]
                for name, params_list in best_param_history.items()
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Holdout evaluation for the original curve plot
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    holdout_rows = []
    predictions = {}
    holdout_best = {}

    for name in model_names:
        if name == "custom":
            best_params, _ = tune_custom_model(X_train, y_train)
            holdout_best[name] = best_params
            y_pred, _, _, _, _ = fit_predict_custom(X_train, y_train, X_test, best_params)
        else:
            best_params, _ = tune_standard_model(X_train, y_train, name)
            holdout_best[name] = best_params
            y_pred, _, _ = fit_predict_standard(X_train, y_train, X_test, name, best_params)

        predictions[name] = y_pred
        holdout_rows.append(
            {
                "Model": name,
                "BestParams": json.dumps(best_params, sort_keys=True),
                "Holdout_MSE": mean_squared_error(y_test, y_pred),
                "Holdout_R2": r2_score(y_test, y_pred),
            }
        )

    pd.DataFrame(holdout_rows).sort_values("Holdout_MSE").to_csv(HOLDOUT_METRICS_CSV, index=False)

    # [ADDED 2/5] error bars in bar chart
    plot_df = summary_df.copy()
    colors = ["purple" if m == "custom" else "tab:blue" for m in plot_df["Model"]]

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    x_pos = np.arange(len(plot_df))

    ax[0].bar(x_pos, plot_df["CV_MSE_Mean"], yerr=plot_df["CV_MSE_CI95"], color=colors, capsize=4)
    ax[0].set_ylabel("MSE (mean ± 95% CI)")
    ax[0].set_title("Mean Squared Error")
    ax[0].set_xticks(x_pos)
    ax[0].set_xticklabels(plot_df["Model"], rotation=30)
    ax[0].set_yscale("log")

    ax[1].bar(x_pos, plot_df["CV_R2_Mean"], yerr=plot_df["CV_R2_CI95"], color=colors, capsize=4)
    ax[1].set_ylabel("$R^2$ (mean ± 95% CI)")
    ax[1].set_title("$R^2$ Score")
    ax[1].set_xticks(x_pos)
    ax[1].set_xticklabels(plot_df["Model"], rotation=30)

    plt.tight_layout()
    plt.savefig(FIG_BAR, dpi=300)
    plt.show(block=False)
    plt.pause(3)
    plt.close(fig)

    # Original RBF vs custom curve retained
    sort_idx = np.argsort(X_test.flatten())
    X_test_sorted = X_test[sort_idx]

    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color="black", alpha=0.3, label="Ground Truth", zorder=1)

    y_rbf_sorted = predictions["rbf"][sort_idx]
    plt.plot(
        X_test_sorted,
        y_rbf_sorted,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"RBF (R²={r2_score(y_test, predictions['rbf']):.3f})",
    )

    y_custom_sorted = predictions["custom"][sort_idx]
    plt.plot(
        X_test_sorted,
        y_custom_sorted,
        color="purple",
        linewidth=2,
        label=f"Custom (R²={r2_score(y_test, predictions['custom']):.3f})",
    )

    plt.title("Prediction Comparison: RBF vs Custom Kernel")
    plt.xlabel("Energy (dimensionless)")
    plt.ylabel("Transmission T(E)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(FIG_CURVE, dpi=300)
    plt.show()

    print("\nSaved:")
    print(f"  - {CV_METRICS_CSV}")
    print(f"  - {HOLDOUT_METRICS_CSV}")
    print(f"  - {HYPERPARAM_JSON}")
    print(f"  - {COST_CSV}")
    print(f"  - {FIG_BAR}")
    print(f"  - {FIG_CURVE}")


if __name__ == "__main__":
    main()
