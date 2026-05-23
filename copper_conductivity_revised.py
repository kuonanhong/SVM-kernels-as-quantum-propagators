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
from pymatgen.electronic_structure.dos import CompleteDos
from scipy.stats import t
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RepeatedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
#API_KEY = "YOUR-MATERIALS-PROJECT-API-KEY"   # <- fill in your own key
API_KEY = "YDsmV6fwqTnN5kKdgJAHuH1JSRrjFak4"
SEARCH_CRITERIA = {
    "elements": ["Cu"],
    "num_elements": (1, 3),
    "is_stable": True,
}

OUT_DIR = Path(".")
DATA_CSV = OUT_DIR / "copper_materials_conductivity_proxy.csv"
HOLDOUT_METRICS_CSV = OUT_DIR / "svm_kernel_performance_metrics.csv"
CV_METRICS_CSV = OUT_DIR / "svm_kernel_cv_metrics.csv"
HYPERPARAM_JSON = OUT_DIR / "svm_kernel_best_params.json"
COST_CSV = OUT_DIR / "svm_kernel_cost_analysis.csv"
FIGURE_PNG = OUT_DIR / "svm_kernel_performance_csv.png"

RANDOM_STATE = 42
OUTER_CV = RepeatedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
INNER_CV = 5

# -----------------------------------------------------------------------------
# HELPER: Robust DOS Fetcher
# -----------------------------------------------------------------------------
def get_dos_robust(mpr, material_id):
    methods = [
        (mpr, "get_dos_by_material_id"),
        (getattr(mpr, "electronic_structure", None), "get_dos_by_material_id"),
        (getattr(mpr, "electronic_structure", None), "retrieve"),
        (getattr(mpr.materials, "electronic_structure", None), "get_dos_by_material_id"),
    ]
    for obj, method_name in methods:
        if obj is not None and hasattr(obj, method_name):
            try:
                result = getattr(obj, method_name)(material_id)
                if result:
                    return result
            except Exception:
                continue
    return None


# -----------------------------------------------------------------------------
# HELPER: Manual Data Extraction
# -----------------------------------------------------------------------------
def extract_dos_at_fermi(dos):
    if isinstance(dos, dict):
        try:
            dos = CompleteDos.from_dict(dos)
        except Exception:
            pass

    try:
        efermi = getattr(dos, "efermi", None)
        if efermi is None and isinstance(dos, dict):
            efermi = dos.get("efermi")

        energies = getattr(dos, "energies", None)
        if energies is None and isinstance(dos, dict):
            energies = dos.get("energies")

        densities = getattr(dos, "densities", None)
        if densities is None and isinstance(dos, dict):
            densities = dos.get("densities")
    except Exception as exc:
        print(f"[Parse Error] Could not access DOS attributes: {exc}")
        return None

    if energies is None or densities is None or efermi is None:
        return None

    idx = np.abs(np.asarray(energies) - efermi).argmin()

    total_dos = 0.0
    if isinstance(densities, dict):
        for _, values in densities.items():
            total_dos += float(values[idx])
    return total_dos


# -----------------------------------------------------------------------------
# ORIGINAL PART A: DATA GENERATION (preserved)
# -----------------------------------------------------------------------------
def generate_dataset_from_materials_project():
    print("Fetching materials data from Materials Project...")
    data_points = []

    with MPRester(API_KEY) as mpr:
        docs = mpr.materials.summary.search(
            elements=SEARCH_CRITERIA["elements"],
            num_elements=SEARCH_CRITERIA["num_elements"],
            is_stable=SEARCH_CRITERIA["is_stable"],
            fields=["material_id", "band_gap", "density", "is_metal", "formula_pretty"],
        )

        print(f"Found {len(docs)} materials.")

        for doc in docs:
            try:
                dos = get_dos_robust(mpr, doc.material_id)
                if dos is None:
                    print(f"[MISSING] DOS not found for {doc.material_id}")
                    continue

                val_dos_ef = extract_dos_at_fermi(dos)
                if val_dos_ef is None:
                    print(f"[DATA ERROR] Could not extract numeric DOS(Ef) for {doc.material_id}")
                    continue

                eg = float(doc.band_gap)
                rho = float(doc.density)
                kT = 0.02585

                if bool(doc.is_metal) or eg < 0.01:
                    sigma_real_proxy = val_dos_ef * rho
                    regime = "Metal"
                else:
                    sigma_real_proxy = np.exp(-eg / (2 * kT)) * rho
                    regime = "Semiconductor"

                log_sigma = np.log10(sigma_real_proxy + 1e-10)

                data_points.append(
                    {
                        "Material": doc.formula_pretty,
                        "Eg_eV": eg,
                        "Density_g_cm3": rho,
                        "DOS_at_Ef": val_dos_ef,
                        "Conductivity_Proxy": sigma_real_proxy,
                        "Log_Sigma": log_sigma,
                        "Regime": regime,
                    }
                )
                print(f"[SUCCESS] {doc.formula_pretty} | DOS(Ef): {val_dos_ef:.4f}")

            except Exception as exc:
                print(f"[ERROR] {doc.material_id}: {exc}")
                continue

    df = pd.DataFrame(data_points)
    if not df.empty:
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        df.to_csv(DATA_CSV, index=False)
        print(f"Saved dataset to {DATA_CSV}")
    else:
        print("Dataset is empty. Please check API access / logs.")
    return df


# -----------------------------------------------------------------------------
# [ADDED 0/5] Shared evaluation helpers
# -----------------------------------------------------------------------------
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
    counter = Counter(values)
    return {str(k): int(v) for k, v in counter.items()}


def nested_cv_for_model(X, y, estimator, param_grid, model_name):
    """[ADDED 1/5], [ADDED 2/5], [ADDED 3/5], [ADDED 5/5]"""
    scoring = {
        "mse": make_scorer(mean_squared_error, greater_is_better=False),
        "r2": "r2",
    }

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
    fit_times = []
    pred_times = []

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

        mse = mean_squared_error(y_te, y_pred)
        r2 = r2_score(y_te, y_pred)

        rows.append(
            {
                "Model": model_name,
                "Split": split_id,
                "MSE": mse,
                "R2": r2,
                "FitTimeSec": fit_elapsed,
                "PredictTimeSec": pred_elapsed,
            }
        )
        best_params_history.append(json.dumps(search.best_params_, sort_keys=True))
        fit_times.append(fit_elapsed)
        pred_times.append(pred_elapsed)

    fold_df = pd.DataFrame(rows)
    mse_mean, mse_std, mse_ci95 = regression_summary(fold_df["MSE"])
    r2_mean, r2_std, r2_ci95 = regression_summary(fold_df["R2"])
    fit_mean, fit_std, fit_ci95 = regression_summary(fit_times)
    pred_mean, pred_std, pred_ci95 = regression_summary(pred_times)

    summary = {
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
    return fold_df, summary


# -----------------------------------------------------------------------------
# [ADDED 4/5] More rigorous benchmarking
# - We keep the four original SVR kernels
# - We add DummyRegressor and RandomForestRegressor as stronger baselines
# -----------------------------------------------------------------------------
def build_model_spaces():
    model_spaces = {
        "rbf": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", SVR(kernel="rbf")),
                ]
            ),
            {
                "model__C": [1, 10, 100, 300],
                "model__gamma": ["scale", 0.1, 1.0, 10.0],
                "model__epsilon": [0.01, 0.05, 0.1],
            },
        ),
        "linear": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", SVR(kernel="linear")),
                ]
            ),
            {
                "model__C": [0.1, 1, 10, 100, 300],
                "model__epsilon": [0.01, 0.05, 0.1],
            },
        ),
        "poly": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", SVR(kernel="poly")),
                ]
            ),
            {
                "model__degree": [2, 3, 4],
                "model__C": [1, 10, 100],
                "model__gamma": ["scale", 0.1, 1.0],
                "model__coef0": [0.0, 1.0],
                "model__epsilon": [0.01, 0.05],
            },
        ),
        "sigmoid": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", SVR(kernel="sigmoid")),
                ]
            ),
            {
                "model__C": [0.1, 1, 10, 100],
                "model__gamma": ["scale", 0.01, 0.1, 1.0],
                "model__coef0": [-1.0, 0.0, 1.0],
                "model__epsilon": [0.01, 0.05],
            },
        ),
        "dummy_mean": (
            DummyRegressor(strategy="mean"),
            {"strategy": ["mean"]},
        ),
        "random_forest": (
            RandomForestRegressor(random_state=RANDOM_STATE),
            {
                "n_estimators": [200, 400],
                "max_depth": [None, 6, 12],
                "min_samples_split": [2, 5],
            },
        ),
    }
    return model_spaces


# -----------------------------------------------------------------------------
# ORIGINAL PART B + ADDED SECTIONS
# -----------------------------------------------------------------------------
def main():
    if not DATA_CSV.exists():
        df = generate_dataset_from_materials_project()
    else:
        df = pd.read_csv(DATA_CSV)
        print(f"Loaded existing dataset from {DATA_CSV}")

    if df.empty:
        raise RuntimeError("Copper dataset is empty.")

    X = df[["Eg_eV", "Density_g_cm3"]].values
    y = df["Log_Sigma"].values

    model_spaces = build_model_spaces()
    cv_summaries = []
    best_param_dump = {}

    print("\n[ADDED 1/5] Running repeated outer CV with inner hyperparameter tuning...")
    for model_name, (estimator, grid) in model_spaces.items():
        print(f"  -> Evaluating {model_name}")
        _, summary = nested_cv_for_model(X, y, estimator, grid, model_name)
        cv_summaries.append(summary)
        best_param_dump[model_name] = summary["BestParamsFrequency"]

    cv_df = pd.DataFrame(cv_summaries).sort_values("CV_MSE_Mean")
    cv_df.to_csv(CV_METRICS_CSV, index=False)
    with open(HYPERPARAM_JSON, "w", encoding="utf-8") as f:
        json.dump(best_param_dump, f, indent=2, ensure_ascii=False)

    # Hold-out evaluation retained for continuity with the original paper figure.
    print("\nRetaining original 80/20 hold-out evaluation for continuity...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    holdout_rows = []
    cost_rows = []

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

    holdout_df = pd.DataFrame(holdout_rows).sort_values("Holdout_MSE")
    holdout_df.to_csv(HOLDOUT_METRICS_CSV, index=False)
    pd.DataFrame(cost_rows).to_csv(COST_CSV, index=False)

    # Plot using CV means + 95% CIs to answer reviewer #1.
    plot_df = cv_df.copy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.bar(
        plot_df["Model"],
        plot_df["CV_MSE_Mean"],
        yerr=plot_df["CV_MSE_CI95"],
        capsize=4,
    )
    ax1.set_title("Copper Conductivity: CV Mean Squared Error")
    ax1.set_ylabel("MSE (mean ± 95% CI)")
    ax1.set_yscale("log")
    ax1.tick_params(axis="x", rotation=30)

    ax2.bar(
        plot_df["Model"],
        plot_df["CV_R2_Mean"],
        yerr=plot_df["CV_R2_CI95"],
        capsize=4,
    )
    ax2.set_title("Copper Conductivity: CV $R^2$")
    ax2.set_ylabel("$R^2$ (mean ± 95% CI)")
    ax2.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(FIGURE_PNG, dpi=300)
    plt.show(block=False)
    plt.pause(3)
    plt.close(fig)

    print("\nSaved:")
    print(f"  - {HOLDOUT_METRICS_CSV}")
    print(f"  - {CV_METRICS_CSV}")
    print(f"  - {HYPERPARAM_JSON}")
    print(f"  - {COST_CSV}")
    print(f"  - {FIGURE_PNG}")


if __name__ == "__main__":
    main()
