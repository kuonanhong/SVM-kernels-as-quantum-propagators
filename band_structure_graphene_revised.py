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
from mp_api.client import MPRester
from pymatgen.electronic_structure.core import Spin
from scipy.stats import t
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RepeatedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# --- CONFIGURATION ---
#MAPI_KEY = "YOUR-MATERIALS-PROJECT-API-KEY"   # <- fill in your own key
MAPI_KEY = "YDsmV6fwqTnN5kKdgJAHuH1JSRrjFak4"

OUT_DIR = Path(".")
CV_METRICS_CSV = OUT_DIR / "graphene_kernel_cv_metrics.csv"
HOLDOUT_METRICS_CSV = OUT_DIR / "graphene_kernel_holdout_metrics.csv"
HYPERPARAM_JSON = OUT_DIR / "graphene_kernel_best_params.json"
COST_CSV = OUT_DIR / "graphene_kernel_cost_analysis.csv"
FIGURE_PNG = OUT_DIR / "graphene_linear_proof.png"

RANDOM_STATE = 42
OUTER_CV = RepeatedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
INNER_CV = 5


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


def fetch_graphene_dataset():
    print("--- Fetching Real-World Graphene Data (mp-48) ---")
    try:
        with MPRester(MAPI_KEY) as mpr:
            bs = mpr.get_bandstructure_by_material_id("mp-48")
            print("[SUCCESS] Band structure fetched.")

            eigenvals = bs.bands[Spin.up]
            kpoints = np.array([k.cart_coords for k in bs.kpoints])
            e_fermi = bs.efermi

            min_diff = float("inf")
            k_dirac = None
            for band in eigenvals:
                dist = np.abs(band - e_fermi)
                min_idx = np.argmin(dist)
                if dist[min_idx] < min_diff:
                    min_diff = dist[min_idx]
                    k_dirac = kpoints[min_idx]

            print(f"Dirac Point identified at k = {k_dirac}")

            X_data, y_data = [], []
            print("Extracting linear dispersion data...")
            for band_energies in eigenvals:
                energies = band_energies - e_fermi
                if np.min(energies) < 0.5 and np.max(energies) > -0.5:
                    for i, E in enumerate(energies):
                        if abs(E) < 1.0 and E > 0:
                            k_curr = kpoints[i]
                            q = np.linalg.norm(k_curr[:2] - k_dirac[:2])
                            X_data.append([q])
                            y_data.append(E)

            X = np.array(X_data)
            y = np.array(y_data)
            print(f"Dataset compiled: {len(y)} points from conduction band.")
            return X, y
    except Exception as exc:
        print(f"[ERROR] {exc}")
        print("Falling back to synthetic linear data...")
        rng = np.random.default_rng(RANDOM_STATE)
        X = rng.uniform(0, 0.2, size=(200, 1))
        y = 10 * X.flatten() + rng.normal(0, 0.05, size=200)
        return X, y


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
    for train_idx, test_idx in OUTER_CV.split(X, y):
        split_id += 1
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

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


def build_model_spaces():
    # [ADDED 4/5] more rigorous benchmarking with dummy and random forest baselines
    return {
        "linear": (
            Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="linear"))]),
            {"model__C": [0.1, 1, 10, 100, 300], "model__epsilon": [0.001, 0.01, 0.05]},
        ),
        "rbf": (
            Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="rbf"))]),
            {
                "model__C": [1, 10, 100, 300],
                "model__gamma": ["scale", 0.1, 1.0, 10.0],
                "model__epsilon": [0.001, 0.01, 0.05],
            },
        ),
        "poly": (
            Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="poly"))]),
            {
                "model__degree": [2, 3],
                "model__C": [1, 10, 100],
                "model__gamma": ["scale", 0.1, 1.0],
                "model__coef0": [0.0, 1.0],
                "model__epsilon": [0.001, 0.01, 0.05],
            },
        ),
        "sigmoid": (
            Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="sigmoid"))]),
            {
                "model__C": [0.1, 1, 10, 100],
                "model__gamma": ["scale", 0.01, 0.1, 1.0],
                "model__coef0": [-1.0, 0.0, 1.0],
                "model__epsilon": [0.001, 0.01, 0.05],
            },
        ),
        "dummy_mean": (DummyRegressor(strategy="mean"), {"strategy": ["mean"]}),
        "random_forest": (
            RandomForestRegressor(random_state=RANDOM_STATE),
            {"n_estimators": [200, 400], "max_depth": [None, 4, 8]},
        ),
    }


def main():
    X, y = fetch_graphene_dataset()
    model_spaces = build_model_spaces()

    summaries = []
    best_params = {}
    print("\n[ADDED 1/5] Running repeated nested CV...")
    for model_name, (estimator, grid) in model_spaces.items():
        print(f"  -> {model_name}")
        summary = nested_cv_for_model(X, y, estimator, grid, model_name)
        summaries.append(summary)
        best_params[model_name] = summary["BestParamsFrequency"]

    cv_df = pd.DataFrame(summaries).sort_values("CV_MSE_Mean")
    cv_df.to_csv(CV_METRICS_CSV, index=False)
    with open(HYPERPARAM_JSON, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)

    # Holdout evaluation preserved for continuity with the original figure logic.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    holdout_rows, cost_rows = [], []
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

        holdout_rows.append(
            {
                "Model": model_name,
                "BestParams": json.dumps(search.best_params_, sort_keys=True),
                "Holdout_MSE": mean_squared_error(y_test, y_pred),
                "Holdout_R2": r2_score(y_test, y_pred),
            }
        )
        cost_rows.append(
            {
                "Model": model_name,
                "FitTimeSec": fit_elapsed,
                "PredictTimeSec": pred_elapsed,
                "TrainingSamples": len(X_train),
                "TestSamples": len(X_test),
            }
        )

    pd.DataFrame(holdout_rows).sort_values("Holdout_MSE").to_csv(HOLDOUT_METRICS_CSV, index=False)
    pd.DataFrame(cost_rows).to_csv(COST_CSV, index=False)

    # [ADDED 2/5] Error bars / confidence intervals from repeated nested CV.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.bar(
        cv_df["Model"],
        cv_df["CV_MSE_Mean"],
        yerr=cv_df["CV_MSE_CI95"],
        capsize=4,
    )
    ax1.set_ylabel("MSE (mean ± 95% CI)")
    ax1.set_title("Prediction Error (MSE)\n(Graphene Conduction Band)")
    ax1.tick_params(axis="x", rotation=30)
    ax1.set_yscale("log")

    ax2.bar(
        cv_df["Model"],
        cv_df["CV_R2_Mean"],
        yerr=cv_df["CV_R2_CI95"],
        capsize=4,
    )
    ax2.set_ylabel("$R^2$ (mean ± 95% CI)")
    ax2.set_title("Model Accuracy ($R^2$)\n(Graphene Conduction Band)")
    ax2.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(FIGURE_PNG, dpi=300)
    plt.show(block=False)
    plt.pause(3)
    plt.close(fig)

    print("\nSaved:")
    print(f"  - {CV_METRICS_CSV}")
    print(f"  - {HOLDOUT_METRICS_CSV}")
    print(f"  - {HYPERPARAM_JSON}")
    print(f"  - {COST_CSV}")
    print(f"  - {FIGURE_PNG}")


if __name__ == "__main__":
    main()
