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
from scipy.linalg import eigh
from scipy.stats import t
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid, RepeatedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import KFold

OUT_DIR = Path(".")
CV_METRICS_CSV = OUT_DIR / "photonic_kernel_cv_metrics.csv"
HOLDOUT_METRICS_CSV = OUT_DIR / "photonic_kernel_holdout_metrics.csv"
HYPERPARAM_JSON = OUT_DIR / "photonic_kernel_best_params.json"
COST_CSV = OUT_DIR / "photonic_kernel_cost_analysis.csv"
FIG_BAR = OUT_DIR / "jackson_metrics.png"
FIG_CURVE = OUT_DIR / "jackson_spectrum.png"

RANDOM_STATE = 42
OUTER_CV = RepeatedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE)
INNER_CV_SPLITS = 4


# -----------------------------------------------------------------------------
# ORIGINAL PHYSICAL SIMULATION (preserved)
# -----------------------------------------------------------------------------
print("--- Simulating 1D Photonic Crystal ---")
energies = np.linspace(0.5, 3.0, 500)


def n_Si(E):
    lam_microns = 1.2398 / (E + 1e-6)
    epsilon = 11.6858 + 0.939816 / (lam_microns**2) + 8.10461e-3 * lam_microns**2
    return np.sqrt(epsilon)


def n_SiO2(E):
    return 1.45 * np.ones_like(E)


def solve_tmm(energies, layers):
    T_list = []
    for E in energies:
        lam = 1.2398 / E
        k0 = 2 * np.pi / lam
        M = np.eye(2, dtype=complex)
        for (n_func, d) in layers:
            n = n_func(E)
            phi = n * k0 * d
            M_layer = np.array(
                [
                    [np.cos(phi), -1j / n * np.sin(phi)],
                    [-1j * n * np.sin(phi), np.cos(phi)],
                ]
            )
            M = np.dot(M, M_layer)
        m11, m12, m21, m22 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
        tcoef = 2 / (m11 + m12 + m21 + m22)
        T_list.append(np.abs(tcoef) ** 2)
    return np.array(T_list)


center_E = 1.24
d_Si = (1.2398 / center_E) / (4 * n_Si(center_E))
d_SiO2 = (1.2398 / center_E) / (4 * n_SiO2(center_E))
layers = []
for _ in range(10):
    layers.append((n_Si, d_Si))
    layers.append((n_SiO2, d_SiO2))
transmission = solve_tmm(energies, layers)


# -----------------------------------------------------------------------------
# ORIGINAL CUSTOM KERNEL (preserved, plus helper timings)
# -----------------------------------------------------------------------------
class JacksonChebyshevKernel:
    def __init__(self, degree=60, clip_eigen=True, epsilon=1e-6, jitter=1e-12):
        self.degree = degree
        self.clip_eigen = clip_eigen
        self.epsilon = float(epsilon)
        self.jitter = float(jitter)
        self.x_min = None
        self.x_max = None

    def _scale(self, X):
        X = np.asarray(X, dtype=float)
        denom = float(self.x_max - self.x_min)

        # avoid divide-by-zero in pathological cases
        if abs(denom) < np.finfo(float).eps:
            return np.zeros_like(X, dtype=float)

        # map to [0,1]
        X_std = (X - self.x_min) / denom

        # map to [-1+eps, 1-eps]
        bound = 1.0 - self.epsilon
        X_scaled = X_std * (2.0 * bound) - bound

        # hard clip to keep arccos safe
        return np.clip(X_scaled, -bound, bound)

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.x_min = np.min(X)
        self.x_max = np.max(X)
        return self

    def compute_gram_matrix(self, X1, X2):
        x1_s = np.asarray(self._scale(X1), dtype=float).reshape(-1)
        x2_s = np.asarray(self._scale(X2), dtype=float).reshape(-1)

        N = self.degree
        n = np.arange(N + 1)   # 用 N+1 較一致，含 T_0 ... T_N
        theta = np.pi / (N + 1)
        sin_n = np.sin(n * theta)
        cos_n = np.cos(n * theta)
        cot_theta = 1.0 / np.tan(theta)
        g_n = ((N - n + 1) * cos_n + sin_n * cot_theta) / (N + 1)
        g_n = np.clip(g_n, 0.0, None)

        feats1 = np.cos(np.arccos(x1_s)[:, None] * n)
        feats2 = np.cos(np.arccos(x2_s)[:, None] * n)

        weighted_feats1 = feats1 * np.sqrt(g_n)
        weighted_feats2 = feats2 * np.sqrt(g_n)

        K = weighted_feats1 @ weighted_feats2.T
        K = np.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)
        return K

    def get_train_kernel(self, X_train):
        self.fit(X_train)
        K = self.compute_gram_matrix(X_train, X_train)
        K = 0.5 * (K + K.T)

        if self.clip_eigen:
            vals, vecs = eigh(K)
            vals = np.clip(vals, 0.0, None)
            K = vecs @ np.diag(vals) @ vecs.T

        K += self.jitter * np.eye(K.shape[0])
        K = np.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)
        return K

    def get_test_kernel(self, X_train, X_test):
        K = self.compute_gram_matrix(X_test, X_train)
        K = np.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)
        return K

def regression_summary(scores):
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    mean = float(np.mean(scores))
    std = float(np.std(scores, ddof=1)) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 1 else 0.0
    tcrit = float(t.ppf(0.975, df=n - 1)) if n > 1 else 0.0
    ci95 = tcrit * sem if n > 1 else 0.0
    return mean, std, ci95


# -----------------------------------------------------------------------------
# [ADDED 3/5] Hyperparameter tuning details
# -----------------------------------------------------------------------------
STANDARD_MODEL_GRIDS = {
    "rbf": {
        "C": [1, 10, 100],
        "gamma": ["scale", 0.1, 1.0],
        "epsilon": [0.001, 0.01, 0.05],
    },
    "linear": {
        "C": [0.1, 1, 10, 100],
        "epsilon": [0.001, 0.01, 0.05],
    },
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

CUSTOM_GRID = {
    "degree": [30, 45, 60, 75],
    "C": [1, 10, 100],
    "epsilon": [0.001, 0.01, 0.05],
}


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
        mean_mse = float(np.mean(fold_mse))
        if mean_mse < best_score:
            best_score = mean_mse
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

            kernel = JacksonChebyshevKernel(degree=params["degree"], clip_eigen=True)
            K_tr = kernel.get_train_kernel(X_tr)
            K_va = kernel.get_test_kernel(X_tr, X_va)

            if not np.isfinite(K_tr).all():
                raise ValueError("K_tr contains non-finite values.")
            if not np.isfinite(K_va).all():
                raise ValueError("K_va contains non-finite values.")

            model = SVR(kernel="precomputed", C=params["C"], epsilon=params["epsilon"])
            model.fit(K_tr, y_tr)
            y_pred = model.predict(K_va)
            fold_mse.append(mean_squared_error(y_va, y_pred))

        mean_mse = float(np.mean(fold_mse))
        if mean_mse < best_score:
            best_score = mean_mse
            best_params = params

    return best_params, best_score


def fit_predict_custom(X_train, y_train, X_test, params):
    t_kernel0 = time.perf_counter()
    kernel = JacksonChebyshevKernel(degree=params["degree"], clip_eigen=True)
    K_train = kernel.get_train_kernel(X_train)
    K_test = kernel.get_test_kernel(X_train, X_test)

    if not np.isfinite(K_train).all():
        raise ValueError("K_train contains non-finite values.")
    if not np.isfinite(K_test).all():
        raise ValueError("K_test contains non-finite values.")

    kernel_build_elapsed = time.perf_counter() - t_kernel0

    model = SVR(kernel="precomputed", C=params["C"], epsilon=params["epsilon"])
    t0 = time.perf_counter()
    model.fit(K_train, y_train)
    fit_elapsed = time.perf_counter() - t0
    t1 = time.perf_counter()
    y_pred = model.predict(K_test)
    pred_elapsed = time.perf_counter() - t1

    return y_pred, kernel_build_elapsed, fit_elapsed, pred_elapsed, K_train.nbytes / (1024**2)


def main():
    X = energies.reshape(-1, 1)
    y = transmission

    model_names = ["rbf", "linear", "poly", "sigmoid", "custom_jackson", "dummy_mean", "random_forest"]

    # [ADDED 1/5], [ADDED 2/5], [ADDED 3/5], [ADDED 5/5]
    cv_rows = []
    best_param_history = {name: [] for name in model_names}
    cost_rows = []

    for split_id, (tr_idx, te_idx) in enumerate(OUTER_CV.split(X, y), start=1):
        X_train, X_test = X[tr_idx], X[te_idx]
        y_train, y_test = y[tr_idx], y[te_idx]

        for name in model_names:
            if name == "custom_jackson":
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
        cost_sub = cost_df[cost_df["Model"] == name]
        fit_mean, fit_std, fit_ci95 = regression_summary(cost_sub["FitTimeSec"])
        pred_mean, pred_std, pred_ci95 = regression_summary(cost_sub["PredictTimeSec"])
        kb_mean, kb_std, kb_ci95 = regression_summary(cost_sub["KernelBuildTimeSec"])
        mem_mean, mem_std, mem_ci95 = regression_summary(cost_sub["ApproxKernelMemoryMB"])

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

    # Holdout split for continuity with the original curve plot
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    holdout_rows = []

    best_models_for_curve = {}
    for name in model_names:
        if name == "custom_jackson":
            best_params, _ = tune_custom_model(X_train, y_train)
            y_pred, kernel_build_elapsed, fit_elapsed, pred_elapsed, mem_mb = fit_predict_custom(
                X_train, y_train, X_test, best_params
            )
            best_models_for_curve[name] = best_params
        else:
            best_params, _ = tune_standard_model(X_train, y_train, name)
            y_pred, fit_elapsed, pred_elapsed = fit_predict_standard(
                X_train, y_train, X_test, name, best_params
            )
            best_models_for_curve[name] = best_params

        holdout_rows.append(
            {
                "Model": name,
                "BestParams": json.dumps(best_params, sort_keys=True),
                "Holdout_MSE": mean_squared_error(y_test, y_pred),
                "Holdout_R2": r2_score(y_test, y_pred),
            }
        )

    pd.DataFrame(holdout_rows).sort_values("Holdout_MSE").to_csv(HOLDOUT_METRICS_CSV, index=False)

    # [ADDED 2/5] Error bars in bar plot
    plot_df = summary_df.copy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    colors = ["purple" if m == "custom_jackson" else "tab:blue" for m in plot_df["Model"]]

    ax1.bar(plot_df["Model"], plot_df["CV_MSE_Mean"], yerr=plot_df["CV_MSE_CI95"], capsize=4, color=colors)
    ax1.set_title("Mean Squared Error")
    ax1.set_ylabel("MSE (mean ± 95% CI)")
    ax1.set_yscale("log")
    ax1.tick_params(axis="x", rotation=30)

    ax2.bar(plot_df["Model"], plot_df["CV_R2_Mean"], yerr=plot_df["CV_R2_CI95"], capsize=4, color=colors)
    ax2.set_title("$R^2$ Score")
    ax2.set_ylabel("$R^2$ (mean ± 95% CI)")
    ax2.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(FIG_BAR, dpi=300)
    plt.show(block=False)
    plt.pause(3)
    plt.close()

    # Original curve plot retained for RBF vs custom_jackson
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color="black", alpha=0.3, label="Simulated Data")

    X_grid = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)

    # Custom curve
    custom_params = best_models_for_curve["custom_jackson"]
    kernel = JacksonChebyshevKernel(degree=custom_params["degree"], clip_eigen=True)
    K_train = kernel.get_train_kernel(X_train)
    K_grid = kernel.get_test_kernel(X_train, X_grid)
    model_custom = SVR(kernel="precomputed", C=custom_params["C"], epsilon=custom_params["epsilon"])
    model_custom.fit(K_train, y_train)
    y_grid_custom = model_custom.predict(K_grid)
    plt.plot(X_grid, y_grid_custom, color="purple", linewidth=2, label="Custom Jackson-Chebyshev Kernel")

    # RBF curve
    rbf_params = best_models_for_curve["rbf"]
    model_rbf = build_standard_model("rbf", rbf_params)
    model_rbf.fit(X_train, y_train)
    y_grid_rbf = model_rbf.predict(X_grid)
    plt.plot(X_grid, y_grid_rbf, color="red", linestyle="--", linewidth=1.5, label="RBF Kernel")

    plt.title("Photonic Crystal Transmission: Jackson-Chebyshev vs RBF")
    plt.xlabel("Photon Energy (eV)")
    plt.ylabel("Transmission")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(FIG_CURVE, dpi=300)
    plt.show(block=False)
    plt.pause(3)
    plt.close(fig)

    print("\nSaved:")
    print(f"  - {CV_METRICS_CSV}")
    print(f"  - {HOLDOUT_METRICS_CSV}")
    print(f"  - {HYPERPARAM_JSON}")
    print(f"  - {COST_CSV}")
    print(f"  - {FIG_BAR}")
    print(f"  - {FIG_CURVE}")


if __name__ == "__main__":
    main()
