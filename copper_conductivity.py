#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reviewer-focused round-2 copper conductivity-proxy benchmark.

Purpose
-------
This script implements only the copper experiment needed for the second-round
COMPHY revision:

1. obtain a bounded, deterministic subset of stable Cu-containing materials
   from the Materials Project summary endpoint;
2. construct an explicitly labelled conductivity *proxy* from band gap,
   density, and metallic/gapped regime information;
3. compare standard SVR kernels with Dummy, Random-Forest, and MLP baselines;
4. use repeated nested cross-validation, 95% confidence intervals, timings,
   support-vector fractions, and repeated learning curves;
5. generate the exact two figures expected by the revised manuscript:
      copper_round2_benchmark.png
      copper_round2_learning_curves.png

Important methodological point
------------------------------
The reviewer did not request an exhaustive database of every Cu compound.
Copper is a proof-of-concept standard-kernel experiment, not the principal
validation of the Jackson-Chebyshev kernel.  Therefore, this script deliberately
limits the Materials Project summary download and does NOT issue one DOS request
for every returned material.

The target is not measured conductivity.  It is the following constructed proxy
at T = 300 K:

    carrier_factor = 1                              for metallic entries
                   = exp[-Eg/(2 k_B T)]             for gapped entries
    sigma_proxy    = density * carrier_factor
    target         = log10(max(sigma_proxy, 1e-12))

This summary-only definition avoids the unstable per-material DOS endpoint that
caused the reported run to return zero valid records.  The manuscript must use
the same definition; suggested replacement wording is given in the accompanying
answer.

Direct execution
----------------
    python copper_conductivity_revised_round2_reviewer_minimal.py --quick
    python copper_conductivity_revised_round2_reviewer_minimal.py

Offline smoke test
------------------
    python copper_conductivity_revised_round2_reviewer_minimal.py \
        --smoke-test --quick
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import matplotlib

# Safe for PyCharm, terminal, SSH, and headless execution.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    from scipy.stats import t as student_t
except ImportError:  # pragma: no cover - 1.96 fallback remains available
    student_t = None

# -----------------------------------------------------------------------------
# MATERIALS PROJECT CONFIGURATION
# -----------------------------------------------------------------------------
# Explicitly embedded at the user's request so this file can run directly.
API_KEY = ""

RANDOM_STATE = 42
TEMPERATURE_K = 300.0
KB_EV_PER_K = 8.617333262e-5
METAL_GAP_THRESHOLD_EV = 0.01

DEFAULT_DATA_CSV = "copper_materials_conductivity_proxy_round2.csv"
DEFAULT_OUTPUT_DIR = "copper_round2_results"
DEFAULT_TARGET_SAMPLES = 120
DEFAULT_MAX_SUMMARY_RECORDS = 300
MINIMUM_MODEL_SAMPLES = 30


# -----------------------------------------------------------------------------
# GENERAL UTILITIES
# -----------------------------------------------------------------------------
def ensure_output_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(data: Any, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, default=str)


def get_field(document: Any, name: str, default: Any = None) -> Any:
    """Read a field from either an mp-api document model or a raw dictionary."""
    if isinstance(document, dict):
        return document.get(name, default)
    return getattr(document, name, default)


def is_conventional_numeric_mpid(material_id: str) -> bool:
    """Exclude alpha/GNoME IDs while retaining conventional mp-1234 IDs."""
    return re.fullmatch(r"mp-\d+", material_id) is not None


def mean_std_ci95(values: pd.Series | np.ndarray | list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    n = len(array)
    if n == 0:
        return math.nan, math.nan, math.nan
    mean = float(np.mean(array))
    if n == 1:
        return mean, 0.0, 0.0
    std = float(np.std(array, ddof=1))
    sem = std / math.sqrt(n)
    critical = (
        float(student_t.ppf(0.975, df=n - 1))
        if student_t is not None
        else 1.96
    )
    return mean, std, critical * sem


def stringify_counter(values: list[dict[str, Any]]) -> dict[str, int]:
    encoded = [json.dumps(value, sort_keys=True, default=str) for value in values]
    return dict(Counter(encoded))


# -----------------------------------------------------------------------------
# DATA ACQUISITION AND PROXY DEFINITION
# -----------------------------------------------------------------------------
def _query_summary_documents(mpr: Any, max_summary_records: int) -> list[Any]:
    """Query only summary metadata, preferably with a hard result cap.

    No DOS object is downloaded here.  The first call is compatible with current
    mp-api releases.  The fallback removes pagination controls for older clients;
    even then, only lightweight summary fields are transferred.
    """
    query = {
        "elements": ["Cu"],
        "num_elements": (1, 3),
        "is_stable": True,
        "fields": [
            "material_id",
            "formula_pretty",
            "band_gap",
            "density",
            "is_metal",
        ],
        "chunk_size": max_summary_records,
        "num_chunks": 1,
    }
    try:
        return list(mpr.materials.summary.search(**query))
    except TypeError:
        # Older mp-api versions may not expose num_chunks through this method.
        query.pop("chunk_size", None)
        query.pop("num_chunks", None)
        print(
            "[Compatibility] This mp-api version does not accept capped chunk "
            "arguments; retrieving summary metadata only (no DOS downloads)."
        )
        return list(mpr.materials.summary.search(**query))


def _records_from_summary_documents(documents: list[Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    skipped_alpha = 0
    skipped_invalid = 0

    for document in documents:
        material_id = str(get_field(document, "material_id", ""))
        if not is_conventional_numeric_mpid(material_id):
            skipped_alpha += 1
            continue

        try:
            band_gap = float(get_field(document, "band_gap"))
            density = float(get_field(document, "density"))
        except (TypeError, ValueError):
            skipped_invalid += 1
            continue

        if not (np.isfinite(band_gap) and np.isfinite(density) and density > 0):
            skipped_invalid += 1
            continue

        raw_is_metal = get_field(document, "is_metal", None)
        is_metal = (
            bool(raw_is_metal)
            if raw_is_metal is not None
            else band_gap < METAL_GAP_THRESHOLD_EV
        )
        regime = (
            "metal"
            if is_metal or band_gap < METAL_GAP_THRESHOLD_EV
            else "gapped"
        )
        carrier_factor = (
            1.0
            if regime == "metal"
            else float(
                np.exp(
                    -band_gap
                    / (2.0 * KB_EV_PER_K * TEMPERATURE_K)
                )
            )
        )
        conductivity_proxy = max(density * carrier_factor, 1e-12)

        rows.append(
            {
                "MaterialID": material_id,
                "Formula": str(get_field(document, "formula_pretty", "")),
                "Eg_eV": band_gap,
                "Density_g_cm3": density,
                "IsMetal": bool(is_metal),
                "CarrierFactor": carrier_factor,
                "Conductivity_Proxy": conductivity_proxy,
                "Log_Sigma": float(np.log10(conductivity_proxy)),
                "Regime": regime,
            }
        )

    frame = pd.DataFrame(rows)
    print(
        "Summary filtering: "
        f"received={len(documents)}, conventional_numeric={len(frame)}, "
        f"alpha_or_gnome_skipped={skipped_alpha}, invalid_skipped={skipped_invalid}"
    )
    return frame


def _evenly_spaced_selection(
    frame: pd.DataFrame,
    n_select: int,
    sort_columns: list[str],
) -> pd.DataFrame:
    """Select a deterministic subset spanning the sorted descriptor range."""
    if n_select <= 0 or frame.empty:
        return frame.iloc[0:0].copy()
    ordered = frame.sort_values(sort_columns).reset_index(drop=True)
    if len(ordered) <= n_select:
        return ordered.copy()
    positions = np.linspace(0, len(ordered) - 1, n_select)
    indices = np.unique(np.rint(positions).astype(int))
    # Rounding can theoretically produce fewer indices; fill deterministically.
    if len(indices) < n_select:
        missing = [i for i in range(len(ordered)) if i not in set(indices)]
        indices = np.sort(np.concatenate([indices, missing[: n_select - len(indices)]]))
    return ordered.iloc[indices[:n_select]].copy()


def select_balanced_subset(frame: pd.DataFrame, target_samples: int) -> pd.DataFrame:
    """Select a deterministic metal/gapped subset for finite-data benchmarking."""
    if target_samples <= 0:
        raise ValueError("target_samples must be positive")

    metals = frame[frame["Regime"] == "metal"].copy()
    gapped = frame[frame["Regime"] == "gapped"].copy()

    target_metals = min(len(metals), target_samples // 2)
    target_gapped = min(len(gapped), target_samples - target_metals)

    selected_metals = _evenly_spaced_selection(
        metals,
        target_metals,
        ["Density_g_cm3", "MaterialID"],
    )
    selected_gapped = _evenly_spaced_selection(
        gapped,
        target_gapped,
        ["Eg_eV", "Density_g_cm3", "MaterialID"],
    )
    selected = pd.concat([selected_metals, selected_gapped], ignore_index=True)

    # If one regime is scarce, fill remaining slots from the unselected records.
    remaining_needed = min(target_samples, len(frame)) - len(selected)
    if remaining_needed > 0:
        used = set(selected["MaterialID"])
        remaining = frame[~frame["MaterialID"].isin(used)].sort_values(
            ["Regime", "Eg_eV", "Density_g_cm3", "MaterialID"]
        )
        selected = pd.concat(
            [selected, remaining.head(remaining_needed)], ignore_index=True
        )

    selected = (
        selected.sort_values(["Regime", "Eg_eV", "MaterialID"])
        .drop_duplicates("MaterialID")
        .reset_index(drop=True)
    )
    return selected


def fetch_materials_project_dataset(
    api_key: str,
    output_csv: Path,
    target_samples: int,
    max_summary_records: int,
) -> pd.DataFrame:
    try:
        from mp_api.client import MPRester
    except ImportError as exc:
        raise RuntimeError(
            "mp-api is required for Materials Project download. Install with: "
            "python -m pip install --upgrade mp-api"
        ) from exc

    print("Fetching a bounded Cu summary dataset from Materials Project...")
    with MPRester(api_key, mute_progress_bars=True) as mpr:
        documents = _query_summary_documents(mpr, max_summary_records)

    candidates = _records_from_summary_documents(documents)
    if candidates.empty:
        raise RuntimeError(
            "The Materials Project summary query returned no valid conventional "
            "numeric Cu-material records. Check the API key and mp-api version."
        )

    selected = select_balanced_subset(candidates, target_samples)
    if len(selected) < MINIMUM_MODEL_SAMPLES:
        raise RuntimeError(
            f"Only {len(selected)} valid records were selected; at least "
            f"{MINIMUM_MODEL_SAMPLES} are required for the requested nested-CV "
            "and learning-curve benchmark. Increase --max-summary-records."
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_csv, index=False)
    regime_counts = selected["Regime"].value_counts().to_dict()
    print(
        f"Saved {len(selected)} selected records to {output_csv.resolve()} "
        f"with regime counts {regime_counts}."
    )
    return selected


def make_smoke_test_dataset(n_samples: int = 120) -> pd.DataFrame:
    """Generate a deterministic non-publication dataset for code validation only."""
    rng = np.random.default_rng(RANDOM_STATE)
    n_metal = n_samples // 2
    n_gapped = n_samples - n_metal

    metal_gap = rng.uniform(0.0, 0.008, n_metal)
    gapped_gap = rng.uniform(0.05, 2.5, n_gapped)
    band_gap = np.concatenate([metal_gap, gapped_gap])
    density = rng.uniform(3.0, 12.0, n_samples)
    regime = np.array(["metal"] * n_metal + ["gapped"] * n_gapped)
    carrier_factor = np.where(
        regime == "metal",
        1.0,
        np.exp(-band_gap / (2.0 * KB_EV_PER_K * TEMPERATURE_K)),
    )
    proxy = np.maximum(density * carrier_factor, 1e-12)

    frame = pd.DataFrame(
        {
            "MaterialID": [f"smoke-{i:04d}" for i in range(n_samples)],
            "Formula": [f"CuSmoke{i:04d}" for i in range(n_samples)],
            "Eg_eV": band_gap,
            "Density_g_cm3": density,
            "IsMetal": regime == "metal",
            "CarrierFactor": carrier_factor,
            "Conductivity_Proxy": proxy,
            "Log_Sigma": np.log10(proxy),
            "Regime": regime,
        }
    )
    return frame.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)


def load_dataset(
    csv_path: Path,
    api_key: str,
    target_samples: int,
    max_summary_records: int,
    refresh_data: bool,
    smoke_test: bool,
) -> tuple[pd.DataFrame, str]:
    if smoke_test:
        print("[SMOKE TEST] Using deterministic synthetic records; not for publication.")
        return make_smoke_test_dataset(max(target_samples, 60)), "smoke_test"

    if csv_path.exists() and not refresh_data:
        print(f"Loading cached Materials Project dataset: {csv_path.resolve()}")
        frame = pd.read_csv(csv_path)
        source = "cached_materials_project_csv"
    else:
        frame = fetch_materials_project_dataset(
            api_key=api_key,
            output_csv=csv_path,
            target_samples=target_samples,
            max_summary_records=max_summary_records,
        )
        source = "materials_project_summary_api"

    required = {
        "Eg_eV",
        "Density_g_cm3",
        "Log_Sigma",
        "Regime",
        "MaterialID",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    cleaned = (
        frame.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["Eg_eV", "Density_g_cm3", "Log_Sigma"])
        .drop_duplicates("MaterialID")
        .reset_index(drop=True)
    )
    if len(cleaned) < MINIMUM_MODEL_SAMPLES:
        raise RuntimeError(
            f"Dataset has only {len(cleaned)} valid rows after cleaning; at least "
            f"{MINIMUM_MODEL_SAMPLES} are required."
        )
    return cleaned, source


# -----------------------------------------------------------------------------
# MODELS AND NESTED CROSS-VALIDATION
# -----------------------------------------------------------------------------
def model_spaces(quick: bool) -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    epsilon = [0.01] if quick else [0.01, 0.05, 0.1]
    return {
        "Linear SVR": (
            Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="linear"))]),
            {
                "model__C": [1, 10] if quick else [0.1, 1, 10, 100],
                "model__epsilon": epsilon,
            },
        ),
        "RBF SVR": (
            Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="rbf"))]),
            {
                "model__C": [10, 100] if quick else [1, 10, 100, 300],
                "model__gamma": ["scale", 0.1] if quick else ["scale", 0.1, 1.0, 10.0],
                "model__epsilon": epsilon,
            },
        ),
        "Polynomial SVR": (
            Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="poly"))]),
            {
                "model__degree": [2, 3] if quick else [2, 3, 4],
                "model__C": [10] if quick else [1, 10, 100],
                "model__gamma": ["scale"] if quick else ["scale", 0.1, 1.0],
                "model__coef0": [1.0] if quick else [0.0, 1.0],
                "model__epsilon": epsilon,
            },
        ),
        "Sigmoid SVR": (
            Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="sigmoid"))]),
            {
                "model__C": [1] if quick else [0.1, 1, 10],
                "model__gamma": [0.1] if quick else ["scale", 0.01, 0.1],
                "model__coef0": [0.0] if quick else [-1.0, 0.0, 1.0],
                "model__epsilon": epsilon,
            },
        ),
        "Random Forest": (
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "n_estimators": [150] if quick else [200, 400],
                "max_depth": [None, 8] if quick else [None, 6, 12],
                "min_samples_leaf": [1, 2],
            },
        ),
        "MLP": (
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        MLPRegressor(
                            random_state=RANDOM_STATE,
                            max_iter=800 if quick else 3000,
                            early_stopping=False,
                        ),
                    ),
                ]
            ),
            {
                "model__hidden_layer_sizes": [(16, 16)] if quick else [(32, 32), (64, 64)],
                "model__activation": ["tanh"] if quick else ["tanh", "relu"],
                "model__alpha": [1e-4] if quick else [1e-4, 1e-3],
                "model__learning_rate_init": [1e-3],
            },
        ),
        "Dummy Mean": (DummyRegressor(), {"strategy": ["mean"]}),
    }


def cv_configuration(n_samples: int, quick: bool) -> tuple[RepeatedKFold, int]:
    if quick:
        outer_splits, repeats, inner_splits = 3, 1, 3
    else:
        outer_splits, repeats, inner_splits = 5, 3, 5

    # Defensive adjustment for unexpectedly small cached datasets.
    outer_splits = min(outer_splits, max(2, n_samples // 8))
    inner_splits = min(inner_splits, max(2, n_samples // 10))
    outer = RepeatedKFold(
        n_splits=outer_splits,
        n_repeats=repeats,
        random_state=RANDOM_STATE,
    )
    return outer, inner_splits


def run_nested_cv(
    X: np.ndarray,
    y: np.ndarray,
    quick: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[dict[str, Any]]]]:
    outer, inner_splits = cv_configuration(len(X), quick)
    rows: list[dict[str, Any]] = []
    best_parameter_history: dict[str, list[dict[str, Any]]] = {}

    for model_name, (estimator, parameter_grid) in model_spaces(quick).items():
        print(f"Nested CV: {model_name}")
        best_parameter_history[model_name] = []

        for split_id, (train_index, test_index) in enumerate(outer.split(X, y), 1):
            search = GridSearchCV(
                estimator=estimator,
                param_grid=parameter_grid,
                scoring="neg_mean_squared_error",
                cv=inner_splits,
                n_jobs=1 if quick else -1,
                refit=True,
                error_score="raise",
            )

            start = time.perf_counter()
            search.fit(X[train_index], y[train_index])
            fit_time = time.perf_counter() - start

            start = time.perf_counter()
            prediction = search.predict(X[test_index])
            prediction_time = time.perf_counter() - start

            fitted = search.best_estimator_
            candidate = (
                fitted.named_steps.get("model")
                if hasattr(fitted, "named_steps")
                else fitted
            )
            support_fraction = (
                len(candidate.support_) / len(train_index)
                if candidate is not None and hasattr(candidate, "support_")
                else math.nan
            )

            rows.append(
                {
                    "Model": model_name,
                    "Split": split_id,
                    "MSE": mean_squared_error(y[test_index], prediction),
                    "R2": r2_score(y[test_index], prediction),
                    "FitTimeSec": fit_time,
                    "PredictTimeSec": prediction_time,
                    "SupportVectorFraction": support_fraction,
                }
            )
            best_parameter_history[model_name].append(search.best_params_)

    folds = pd.DataFrame(rows)
    summaries: list[dict[str, Any]] = []
    for model_name, group in folds.groupby("Model", sort=False):
        summary: dict[str, Any] = {"Model": model_name, "Splits": len(group)}
        for metric in [
            "MSE",
            "R2",
            "FitTimeSec",
            "PredictTimeSec",
            "SupportVectorFraction",
        ]:
            mean, std, ci95 = mean_std_ci95(group[metric])
            summary[f"{metric}_Mean"] = mean
            summary[f"{metric}_STD"] = std
            summary[f"{metric}_CI95"] = ci95
        summary["BestParamsFrequency"] = json.dumps(
            stringify_counter(best_parameter_history[model_name]),
            sort_keys=True,
        )
        summaries.append(summary)

    summary_frame = pd.DataFrame(summaries).sort_values("MSE_Mean").reset_index(drop=True)
    return folds, summary_frame, best_parameter_history


# -----------------------------------------------------------------------------
# LEARNING CURVES
# -----------------------------------------------------------------------------
def tune_learning_curve_models(
    X: np.ndarray,
    y: np.ndarray,
    quick: bool,
) -> dict[str, Any]:
    spaces = model_spaces(quick)
    _, inner_splits = cv_configuration(len(X), quick)
    tuned: dict[str, Any] = {}
    for model_name in ["RBF SVR", "Random Forest", "MLP"]:
        estimator, grid = spaces[model_name]
        print(f"Full-data tuning for learning curve: {model_name}")
        search = GridSearchCV(
            estimator,
            grid,
            scoring="neg_mean_squared_error",
            cv=inner_splits,
            n_jobs=1 if quick else -1,
            refit=True,
            error_score="raise",
        )
        tuned[model_name] = search.fit(X, y).best_estimator_
    return tuned


def learning_curve_sizes(n_samples: int, quick: bool) -> np.ndarray:
    n_points = 4 if quick else 7
    minimum = max(12, int(round(0.15 * n_samples)))
    maximum = max(minimum + 1, int(round(0.70 * n_samples)))
    sizes = np.unique(np.linspace(minimum, maximum, n_points).astype(int))
    return sizes[sizes < n_samples - 3]


def evaluate_learning_curve(
    estimator_factory: Callable[[], Any],
    X: np.ndarray,
    y: np.ndarray,
    train_sizes: np.ndarray,
    n_repeats: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n_samples = len(X)
    test_size = max(8, int(round(0.25 * n_samples)))

    for repeat in range(n_repeats):
        rng = np.random.default_rng(RANDOM_STATE + 1000 + repeat)
        permutation = rng.permutation(n_samples)
        test_index = permutation[:test_size]
        pool = permutation[test_size:]

        for train_size in train_sizes:
            actual_size = min(int(train_size), len(pool))
            if actual_size < 8:
                continue
            train_index = pool[:actual_size]
            estimator = estimator_factory()

            start = time.perf_counter()
            estimator.fit(X[train_index], y[train_index])
            fit_time = time.perf_counter() - start
            prediction = estimator.predict(X[test_index])

            rows.append(
                {
                    "Repeat": repeat + 1,
                    "TrainSize": actual_size,
                    "MSE": mean_squared_error(y[test_index], prediction),
                    "R2": r2_score(y[test_index], prediction),
                    "FitTimeSec": fit_time,
                }
            )
    return pd.DataFrame(rows)


def summarize_learning_curve(raw: pd.DataFrame) -> pd.DataFrame:
    summaries: list[dict[str, Any]] = []
    for train_size, group in raw.groupby("TrainSize"):
        row: dict[str, Any] = {"TrainSize": int(train_size), "Repeats": len(group)}
        for metric in ["MSE", "R2", "FitTimeSec"]:
            mean, std, ci95 = mean_std_ci95(group[metric])
            row[f"{metric}_Mean"] = mean
            row[f"{metric}_STD"] = std
            row[f"{metric}_CI95"] = ci95
        summaries.append(row)
    return pd.DataFrame(summaries).sort_values("TrainSize").reset_index(drop=True)


def run_learning_curves(
    X: np.ndarray,
    y: np.ndarray,
    quick: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    tuned = tune_learning_curve_models(X, y, quick)
    sizes = learning_curve_sizes(len(X), quick)
    repeats = 3 if quick else 10
    raw_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    summaries_by_model: dict[str, pd.DataFrame] = {}

    for model_name, estimator in tuned.items():
        print(f"Learning curve: {model_name}")
        raw = evaluate_learning_curve(
            estimator_factory=lambda fitted=estimator: clone(fitted),
            X=X,
            y=y,
            train_sizes=sizes,
            n_repeats=repeats,
        )
        raw["Model"] = model_name
        summary = summarize_learning_curve(raw)
        summary["Model"] = model_name
        raw_frames.append(raw)
        summary_frames.append(summary)
        summaries_by_model[model_name] = summary

    return (
        pd.concat(raw_frames, ignore_index=True),
        pd.concat(summary_frames, ignore_index=True),
        summaries_by_model,
    )


# -----------------------------------------------------------------------------
# FIGURES
# -----------------------------------------------------------------------------
def plot_benchmark(summary: pd.DataFrame, output_path: Path) -> None:
    plot_data = summary.sort_values("MSE_Mean").reset_index(drop=True)
    positions = np.arange(len(plot_data))

    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].bar(
        positions,
        plot_data["MSE_Mean"],
        yerr=plot_data["MSE_CI95"],
        capsize=4,
    )
    axes[0].set_yscale("log")
    axes[0].set_title("Copper proxy: repeated nested-CV MSE")
    axes[0].set_ylabel("MSE (mean ± 95% CI)")
    axes[0].set_xticks(positions, plot_data["Model"], rotation=30, ha="right")

    axes[1].bar(
        positions,
        plot_data["R2_Mean"],
        yerr=plot_data["R2_CI95"],
        capsize=4,
    )
    axes[1].axhline(0.0, linewidth=1)
    axes[1].set_title("Copper proxy: repeated nested-CV $R^2$")
    axes[1].set_ylabel("$R^2$ (mean ± 95% CI)")
    axes[1].set_xticks(positions, plot_data["Model"], rotation=30, ha="right")

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_learning_curves(
    summaries_by_model: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for model_name, summary in summaries_by_model.items():
        axes[0].errorbar(
            summary["TrainSize"],
            summary["MSE_Mean"],
            yerr=summary["MSE_CI95"],
            marker="o",
            capsize=3,
            label=model_name,
        )
        axes[1].errorbar(
            summary["TrainSize"],
            summary["R2_Mean"],
            yerr=summary["R2_CI95"],
            marker="o",
            capsize=3,
            label=model_name,
        )

    axes[0].set_yscale("log")
    axes[0].set_title("Sample-efficiency: test MSE")
    axes[0].set_xlabel("Training samples")
    axes[0].set_ylabel("MSE (mean ± 95% CI)")
    axes[0].legend()

    axes[1].axhline(0.0, linewidth=1)
    axes[1].set_title("Sample-efficiency: test $R^2$")
    axes[1].set_xlabel("Training samples")
    axes[1].set_ylabel("$R^2$ (mean ± 95% CI)")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reviewer-focused copper conductivity-proxy benchmark"
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-csv", default=DEFAULT_DATA_CSV)
    parser.add_argument(
        "--api-key",
        default=None,
        help="Materials Project API key; prefer MP_API_KEY environment variable",
    )
    parser.add_argument(
        "--target-samples",
        type=int,
        default=DEFAULT_TARGET_SAMPLES,
        help="Maximum balanced subset retained for the benchmark",
    )
    parser.add_argument(
        "--max-summary-records",
        type=int,
        default=DEFAULT_MAX_SUMMARY_RECORDS,
        help="Maximum summary documents requested from the API when supported",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Ignore an existing CSV and query Materials Project again",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use deterministic synthetic data only to validate execution",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_arguments(argv)
    if args.target_samples < MINIMUM_MODEL_SAMPLES:
        raise ValueError(
            f"--target-samples must be at least {MINIMUM_MODEL_SAMPLES}"
        )
    if args.max_summary_records < args.target_samples:
        print(
            "[Notice] --max-summary-records is smaller than --target-samples; "
            "the final dataset may contain fewer rows than requested."
        )

    output_dir = ensure_output_dir(args.output_dir)
    data_path = Path(args.data_csv)
    frame, data_source = load_dataset(
        csv_path=data_path,
        api_key=args.api_key or os.getenv("MP_API_KEY", ""),
        target_samples=args.target_samples,
        max_summary_records=args.max_summary_records,
        refresh_data=args.refresh_data,
        smoke_test=args.smoke_test,
    )

    X = frame[["Eg_eV", "Density_g_cm3"]].to_numpy(dtype=float)
    y = frame["Log_Sigma"].to_numpy(dtype=float)

    print(
        f"Benchmark dataset: n={len(frame)}, "
        f"regimes={frame['Regime'].value_counts().to_dict()}"
    )

    folds, cv_summary, parameter_history = run_nested_cv(X, y, args.quick)
    folds.to_csv(output_dir / "copper_round2_cv_folds.csv", index=False)
    cv_summary.to_csv(output_dir / "copper_round2_cv_summary.csv", index=False)
    save_json(parameter_history, output_dir / "copper_round2_best_params.json")
    plot_benchmark(cv_summary, output_dir / "copper_round2_benchmark.png")

    learning_raw, learning_summary, learning_by_model = run_learning_curves(
        X, y, args.quick
    )
    learning_raw.to_csv(
        output_dir / "copper_round2_learning_curve_folds.csv", index=False
    )
    learning_summary.to_csv(
        output_dir / "copper_round2_learning_curve_summary.csv", index=False
    )
    plot_learning_curves(
        learning_by_model,
        output_dir / "copper_round2_learning_curves.png",
    )

    frame.to_csv(output_dir / "copper_round2_selected_dataset.csv", index=False)
    save_json(
        {
            "data_source": data_source,
            "n_samples": len(frame),
            "features": ["Eg_eV", "Density_g_cm3"],
            "target": "Log_Sigma of a constructed carrier-activation proxy",
            "target_definition": {
                "temperature_K": TEMPERATURE_K,
                "metal": "density * 1",
                "gapped": "density * exp(-Eg/(2*k_B*T))",
                "log_transform": "log10(max(proxy, 1e-12))",
            },
            "regime_counts": frame["Regime"].value_counts().to_dict(),
            "selection_policy": (
                "bounded deterministic subset spanning metallic density and "
                "gapped band-gap ranges; no exhaustive per-material DOS download"
            ),
            "api_key_policy": (
                "API key is read from --api-key or MP_API_KEY and is never committed; "
                "remove or rotate it before public release"
            ),
            "random_state": RANDOM_STATE,
            "quick_mode": bool(args.quick),
            "smoke_test": bool(args.smoke_test),
        },
        output_dir / "copper_round2_dataset_notes.json",
    )

    print("\nCompleted. Generated:")
    for name in [
        "copper_round2_benchmark.png",
        "copper_round2_learning_curves.png",
        "copper_round2_cv_folds.csv",
        "copper_round2_cv_summary.csv",
        "copper_round2_learning_curve_folds.csv",
        "copper_round2_learning_curve_summary.csv",
        "copper_round2_best_params.json",
        "copper_round2_selected_dataset.csv",
        "copper_round2_dataset_notes.json",
    ]:
        print(f"  - {(output_dir / name).resolve()}")


if __name__ == "__main__":
    main()
