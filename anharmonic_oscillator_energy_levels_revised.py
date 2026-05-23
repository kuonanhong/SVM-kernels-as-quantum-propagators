#!/usr/bin/env python
# coding: utf-8
# -----------------------------------------------------------------------------
# Original code:
#   Copyright (c) 2025 Renata Wong
#   Repository: https://github.com/renatawong/svm-kernels-as-quantum-propagators
#
# Revised version:
#   Copyright (c) 2026 Nan-Hong Kuo
#   Repository: https://github.com/kuonanhong/SVM-kernels-as-quantum-propagators
#
# This revised script is derived from the original experimental code associated
# with the manuscript:
#   "Support Vector Machine Kernels as Quantum Propagators"
#
# Major-revision additions include:
#   [1] repeated cross-validation statistics
#   [2] error bars and confidence intervals
#   [3] explicit hyperparameter tuning
#   [4] additional benchmark baselines
#   [5] computational-cost reporting
#
# Licensed under the MIT License.
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Copyright (c) 2025 Renata Wong
#
# This code is supplementary material for the research paper:
# "Support Vector Machine Kernels as Quantum Propagators"
#
# NOTE:
# - The original physical data generation logic is preserved as much as possible.
# - I also fixed a few obvious runtime issues in the original plotting code so the
#   revised scripts can execute end-to-end.
# -----------------------------------------------------------------------------

import json
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from scipy.stats import t
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RepeatedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

OUT_DIR = Path(".")
FIG_KERNELS = OUT_DIR / "anharmonic-kernels.png"
FIG_DEGREES = OUT_DIR / "anharmonic-degrees.png"
CV_KERNELS_CSV = OUT_DIR / "anharmonic_kernel_family_cv_metrics.csv"
CV_DEGREES_CSV = OUT_DIR / "anharmonic_poly_degree_cv_metrics.csv"
HOLDOUT_KERNELS_CSV = OUT_DIR / "anharmonic_kernel_family_holdout_metrics.csv"
HOLDOUT_DEGREES_CSV = OUT_DIR / "anharmonic_poly_degree_holdout_metrics.csv"
HYPERPARAM_JSON = OUT_DIR / "anharmonic_best_params.json"
COST_CSV = OUT_DIR / "anharmonic_cost_analysis.csv"

RANDOM_STATE = 42
OUTER_CV = RepeatedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
#INNER_CV = 5
INNER_CV = 3
N_SAMPLES = 500


# ==========================================
# ORIGINAL PHYSICS SIMULATION (preserved)
# ==========================================
def solve_anharmonic_oscillator(n_levels, k, alpha, grid_size=1000, x_max=10):
    """
    Numerically solves the Schrodinger equation for an anharmonic oscillator
    H = -0.5 * d^2/dx^2 + 0.5 * k * x^2 + alpha * x^4
    Returns the n-th energy eigenvalue.
    """
    x = np.linspace(-x_max, x_max, grid_size)
    dx = x[1] - x[0]
    V = 0.5 * k * x**2 + alpha * x**4
    diag_main = 1.0 / (dx**2) * np.ones(grid_size)
    diag_off = -0.5 / (dx**2) * np.ones(grid_size - 1)
    H_diag = diag_main + V
    H = diags([diag_off, H_diag, diag_off], [-1, 0, 1])
    eigenvalues, _ = eigsh(H, k=n_levels + 5, which="SA")
    return eigenvalues


def generate_dataset():
    print("Generating synthetic quantum data...")
    rng = np.random.default_rng(RANDOM_STATE)
    X_data, y_data = [], []
    ns = rng.integers(0, 5, N_SAMPLES)
    ks = rng.uniform(0.5, 2.0, N_SAMPLES)
    alphas = rng.uniform(0.01, 0.2, N_SAMPLES)

    sim_times = []
    for i in range(N_SAMPLES):
        t0 = time.perf_counter()
        n = int(ns[i])
        k = float(ks[i])
        alpha = float(alphas[i])
        energies = solve_anharmonic_oscillator(n_levels=n + 1, k=k, alpha=alpha)
        sim_times.append(time.perf_counter() - t0)
        X_data.append([n, k, alpha])
        y_data.append(float(energies[n]))

    X = np.array(X_data)
    y = np.array(y_data)
    return X, y, sim_times


def regression_summary(scores):
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    mean = float(np.mean(scores))
    std = float(np.std(scores, ddof=1)) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 1 else 0.0
    tcrit = float(t.ppf(0.975, df=n - 1)) if n > 1 else 0.0
    ci95 = tcrit * sem if n > 1 else 0.0
    return mean, std, ci95


def stringify_counter(values):
    return {str(k): int(v) for k, v in Counter(values).items()}


def nested_cv_for_model(X, y, estimator, param_grid, model_name):
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=INNER_CV,
        n_jobs=-1,
        refit=True,
    )

    rows = []
    best_params_history = []
    fit_times, pred_times = [], []
    split_id = 0

    for tr_idx, te_idx in OUTER_CV.split(X, y):
        split_id += 1
        print(f"    [{model_name}] outer split {split_id}/15")
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        t0 = time.perf_counter()
        search.fit(X_tr, y_tr)
        fit_elapsed = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred = search.predict(X_te)
        pred_elapsed = time.perf_counter() - t1

        rows.append(
            {
                "Model": model_name,
                "Split": split_id,
                "MSE": mean_squared_error(y_te, y_pred),
                "R2": r2_score(y_te, y_pred),
                "FitTimeSec": fit_elapsed,
                "PredictTimeSec": pred_elapsed,
            }
        )
        fit_times.append(fit_elapsed)
        pred_times.append(pred_elapsed)
        best_params_history.append(json.dumps(search.best_params_, sort_keys=True))

    fold_df = pd.DataFrame(rows)
    mse_mean, mse_std, mse_ci95 = regression_summary(fold_df["MSE"])
    r2_mean, r2_std, r2_ci95 = regression_summary(fold_df["R2"])
    fit_mean, fit_std, fit_ci95 = regression_summary(fit_times)
    pred_mean, pred_std, pred_ci95 = regression_summary(pred_times)

    return {
        "Model": model_name,
        "CV_MSE_Mean": mse_mean,
        "CV_MSE_STD": mse_std,
        "CV_MSE_CI95": mse_ci95,
        "CV_R2_Mean": r2_mean,
        "CV_R2_STD": r2_std,
        "CV_R2_CI95": r2_ci95,
        "FitTime_MeanSec": fit_mean,
        "FitTime_STD": fit_std,
        "FitTime_CI95": fit_ci95,
        "PredictTime_MeanSec": pred_mean,
        "PredictTime_STD": pred_std,
        "PredictTime_CI95": pred_ci95,
        "BestParamsFrequency": stringify_counter(best_params_history),
    }


# ==========================================
# [ADDED 4/5] More rigorous benchmarking
# ==========================================
def build_kernel_family_spaces():
    return {
        "Linear": (
            Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="linear"))]),
            {"model__C": [0.1, 1, 10, 100], "model__epsilon": [0.01, 0.05, 0.1]},
        ),
        "RBF": (
            Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="rbf"))]),
            {
                "model__C": [1, 10, 100],
                "model__gamma": ["scale", 0.1, 1.0, 10.0],
                "model__epsilon": [0.01, 0.05, 0.1],
            },
        ),
        "Poly (Deg 3)": (
            Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="poly", degree=3))]),
            {
                #"model__C": [1, 10, 100],
                "model__C": [10, 100],
                #"model__gamma": ["scale", 0.1, 1.0],
                "model__gamma": ["scale", 0.1],
                "model__coef0": [0.0, 1.0],
                #"model__epsilon": [0.01, 0.05],
                "model__epsilon": [0.01],
            },
        ),
        "Sigmoid": (
            Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="sigmoid"))]),
            {
                "model__C": [0.1, 1, 10, 100],
                "model__gamma": ["scale", 0.01, 0.1, 1.0],
                "model__coef0": [-1.0, 0.0, 1.0],
                "model__epsilon": [0.01, 0.05],
            },
        ),
        "Dummy Mean": (DummyRegressor(strategy="mean"), {"strategy": ["mean"]}),
        "Random Forest": (
            RandomForestRegressor(random_state=RANDOM_STATE),
            {"n_estimators": [200, 400], "max_depth": [None, 6, 12]},
        ),
    }


def build_polynomial_degree_spaces():
    spaces = {}
    for degree in [2, 3, 4, 5, 6, 7]:
        spaces[f"Poly (Deg {degree})"] = (
            Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="poly", degree=degree))]),
            {
                "model__C": [1, 10, 100],
                "model__gamma": ["scale", 0.1, 1.0],
                "model__coef0": [0.0, 1.0],
                "model__epsilon": [0.01, 0.05],
            },
        )
    return spaces


def holdout_eval(X, y, model_spaces):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    rows, costs, best_params = [], [], {}
    for model_name, (estimator, grid) in model_spaces.items():
        search = GridSearchCV(
            estimator=estimator,
            param_grid=grid,
            scoring="neg_mean_squared_error",
            cv=INNER_CV,
            n_jobs=-1,
            refit=True,
        )
        t0 = time.perf_counter()
        search.fit(X_train, y_train)
        fit_elapsed = time.perf_counter() - t0
        t1 = time.perf_counter()
        y_pred = search.predict(X_test)
        pred_elapsed = time.perf_counter() - t1

        rows.append(
            {
                "Model": model_name,
                "BestParams": json.dumps(search.best_params_, sort_keys=True),
                "Holdout_MSE": mean_squared_error(y_test, y_pred),
                "Holdout_R2": r2_score(y_test, y_pred),
            }
        )
        costs.append(
            {
                "Model": model_name,
                "FitTimeSec": fit_elapsed,
                "PredictTimeSec": pred_elapsed,
                "TrainingSamples": len(X_train),
                "TestSamples": len(X_test),
            }
        )
        best_params[model_name] = search.best_params_
    return pd.DataFrame(rows), pd.DataFrame(costs), best_params


def plot_metric_bars(df, title_prefix, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.bar(df["Model"], df["CV_MSE_Mean"], yerr=df["CV_MSE_CI95"], capsize=4)
    ax1.set_title(f"{title_prefix}: Mean Squared Error")
    ax1.set_ylabel("MSE (mean ± 95% CI)")
    ax1.tick_params(axis="x", rotation=30)
    ax1.set_yscale("log")

    ax2.bar(df["Model"], df["CV_R2_Mean"], yerr=df["CV_R2_CI95"], capsize=4)
    ax2.set_title(f"{title_prefix}: $R^2$")
    ax2.set_ylabel("$R^2$ (mean ± 95% CI)")
    ax2.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show(block=False)
    plt.pause(3)
    plt.close(fig)


def main():
    X, y, sim_times = generate_dataset()

    # [ADDED 5/5] Physics simulation cost is also recorded.
    sim_mean, sim_std, sim_ci95 = regression_summary(sim_times)
    cost_rows = [
        {
            "Stage": "PhysicsSimulation",
            "Model": "FiniteDifferenceEigenSolve",
            "FitTimeSec": sim_mean,
            "FitTimeSTD": sim_std,
            "FitTimeCI95": sim_ci95,
            "PredictTimeSec": 0.0,
            "TrainingSamples": N_SAMPLES,
            "TestSamples": 0,
        }
    ]

    # =========================================================
    # EXPERIMENT A: kernel family comparison (original figure 1)
    # =========================================================
    kernel_spaces = build_kernel_family_spaces()
    kernel_summaries = []
    hyperparam_dump = {}

    print("\n[ADDED 1/5] Running repeated nested CV for kernel family comparison...")
    for model_name, (estimator, grid) in kernel_spaces.items():
        print(f"  -> {model_name}")
        summary = nested_cv_for_model(X, y, estimator, grid, model_name)
        kernel_summaries.append(summary)
        hyperparam_dump[model_name] = summary["BestParamsFrequency"]

    kernel_cv_df = pd.DataFrame(kernel_summaries).sort_values("CV_MSE_Mean")
    kernel_cv_df.to_csv(CV_KERNELS_CSV, index=False)
    plot_metric_bars(kernel_cv_df, "Anharmonic Oscillator Kernel Family", FIG_KERNELS)

    kernel_holdout_df, kernel_cost_df, kernel_best = holdout_eval(X, y, kernel_spaces)
    kernel_holdout_df.sort_values("Holdout_MSE").to_csv(HOLDOUT_KERNELS_CSV, index=False)
    cost_rows.extend(kernel_cost_df.to_dict("records"))

    # =========================================================
    # EXPERIMENT B: polynomial degree comparison (original figure 2)
    # =========================================================
    degree_spaces = build_polynomial_degree_spaces()
    degree_summaries = {}

    print("\n[ADDED 1/5] Running repeated nested CV for polynomial degree comparison...")
    degree_rows = []
    for model_name, (estimator, grid) in degree_spaces.items():
        print(f"  -> {model_name}")
        summary = nested_cv_for_model(X, y, estimator, grid, model_name)
        degree_rows.append(summary)
        degree_summaries[model_name] = summary["BestParamsFrequency"]

    degree_cv_df = pd.DataFrame(degree_rows).sort_values("CV_MSE_Mean")
    degree_cv_df.to_csv(CV_DEGREES_CSV, index=False)
    plot_metric_bars(degree_cv_df, "Polynomial Degree Comparison", FIG_DEGREES)

    degree_holdout_df, degree_cost_df, degree_best = holdout_eval(X, y, degree_spaces)
    degree_holdout_df.sort_values("Holdout_MSE").to_csv(HOLDOUT_DEGREES_CSV, index=False)
    cost_rows.extend(degree_cost_df.to_dict("records"))

    # [ADDED 3/5] Save hyperparameter histories / best-param frequencies
    hyperparam_dump["kernel_family"] = kernel_best
    hyperparam_dump["poly_degrees"] = degree_best
    hyperparam_dump["kernel_family_cv_frequency"] = {k: v for k, v in hyperparam_dump.items() if isinstance(v, dict)}

    with open(HYPERPARAM_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "kernel_family_nested_cv_best_params_frequency": {
                    row["Model"]: row["BestParamsFrequency"] for row in kernel_summaries
                },
                "poly_degree_nested_cv_best_params_frequency": degree_summaries,
                "kernel_family_holdout_best_params": {
                    k: {kk: str(vv) for kk, vv in v.items()} for k, v in kernel_best.items()
                },
                "poly_degree_holdout_best_params": {
                    k: {kk: str(vv) for kk, vv in v.items()} for k, v in degree_best.items()
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    pd.DataFrame(cost_rows).to_csv(COST_CSV, index=False)

    print("\nSaved:")
    print(f"  - {CV_KERNELS_CSV}")
    print(f"  - {CV_DEGREES_CSV}")
    print(f"  - {HOLDOUT_KERNELS_CSV}")
    print(f"  - {HOLDOUT_DEGREES_CSV}")
    print(f"  - {HYPERPARAM_JSON}")
    print(f"  - {COST_CSV}")
    print(f"  - {FIG_KERNELS}")
    print(f"  - {FIG_DEGREES}")


if __name__ == "__main__":
    main()
