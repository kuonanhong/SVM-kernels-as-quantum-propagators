#!/usr/bin/env python3
"""Reviewer-focused second-round benchmark for quartic-oscillator energies.

This streamlined version keeps only the analyses needed for the revision:
1. repeated nested-CV comparison with SVR, random-forest and MLP baselines;
2. polynomial-degree ablation with degree selected empirically;
3. first-order perturbation-theory baseline and residual diagnostic.

The unsupported claim that degree 3 follows from the cubic force is not used.
The leading perturbative correction is quadratic in n, while the optimal
kernel degree is selected inside the inner cross-validation loop.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import eigh_tridiagonal
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from svmprop_common import (
    aggregate_fold_metrics,
    ensure_output_dir,
    plot_metric_bars,
    save_json,
    summary95,
)

RANDOM_STATE = 42

# Reviewer-focused defaults: repeated nested CV, but not an exhaustive search.
DEFAULT_SAMPLES = 120
OUTER_SPLITS = 5
OUTER_REPEATS = 2
INNER_SPLITS = 3
SVR_KWARGS = {"tol": 1e-3, "cache_size": 1000}


class FirstOrderPerturbationRegressor(BaseEstimator, RegressorMixin):
    """First-order quartic-oscillator approximation for hbar=m=1."""

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        n, k, lam = X[:, 0], X[:, 1], X[:, 2]
        omega = np.sqrt(k)
        harmonic = omega * (n + 0.5)
        correction = 3.0 * lam * (2.0 * n**2 + 2.0 * n + 1.0) / (4.0 * k)
        return harmonic + correction

def make_scaled_svr(kernel: str, **kwargs):
    """Scale both descriptors and target inside each CV fit to stabilize SVR."""
    regressor = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", SVR(kernel=kernel, max_iter=20000, **SVR_KWARGS, **kwargs)),
        ]
    )
    return TransformedTargetRegressor(
        regressor=regressor,
        transformer=StandardScaler(),
    )


def make_scaled_mlp(max_iter: int):
    """Small deterministic neural baseline with target scaling inside CV."""
    regressor = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                    max_iter=max_iter,
                    max_fun=5000,
                ),
            ),
        ]
    )
    return TransformedTargetRegressor(
        regressor=regressor,
        transformer=StandardScaler(),
    )


def solve_quartic_oscillator(
    n_max: int,
    k: float,
    lam: float,
    grid_size: int = 360,
    x_max: float = 7.0,
) -> np.ndarray:
    """Return the lowest n_max+1 eigenvalues of the finite-difference Hamiltonian.

    The Hamiltonian is tridiagonal, so ``eigh_tridiagonal`` is substantially
    faster than repeatedly invoking a general sparse eigensolver, while solving
    the same finite-difference eigenproblem.
    """
    x = np.linspace(-x_max, x_max, grid_size)
    dx = x[1] - x[0]
    potential = 0.5 * k * x**2 + lam * x**4
    diagonal = np.full(grid_size, 1.0 / dx**2) + potential
    off_diagonal = np.full(grid_size - 1, -0.5 / dx**2)
    values = eigh_tridiagonal(
        diagonal,
        off_diagonal,
        select="i",
        select_range=(0, int(n_max)),
        eigvals_only=True,
        check_finite=False,
    )
    return np.asarray(values, dtype=float)


def generate_dataset(n_samples: int, cache_csv: Path, refresh: bool = False):
    required = {"n", "k", "lambda", "energy", "solve_time_sec"}
    if cache_csv.exists() and not refresh:
        cached = pd.read_csv(cache_csv)
        if required.issubset(cached.columns) and len(cached) >= n_samples:
            cached = cached.iloc[:n_samples].copy()
            print(f"Loaded {len(cached)} cached numerical samples from {cache_csv}")
            return (
                cached[["n", "k", "lambda"]].to_numpy(dtype=float),
                cached["energy"].to_numpy(dtype=float),
                cached,
            )

    rng = np.random.default_rng(RANDOM_STATE)
    ns = rng.integers(0, 6, n_samples)
    ks = rng.uniform(0.6, 2.0, n_samples)
    lams = rng.uniform(0.005, 0.20, n_samples)
    rows: list[dict[str, float | int]] = []

    print(f"Generating {n_samples} quartic-oscillator samples...")
    for i, (n, k, lam) in enumerate(zip(ns, ks, lams), 1):
        start = time.perf_counter()
        values = solve_quartic_oscillator(int(n), float(k), float(lam))
        rows.append(
            {
                "n": int(n),
                "k": float(k),
                "lambda": float(lam),
                "energy": float(values[int(n)]),
                "solve_time_sec": time.perf_counter() - start,
            }
        )
        if i % 40 == 0 or i == n_samples:
            print(f"  generated {i}/{n_samples}")

    df = pd.DataFrame(rows)
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_csv, index=False)
    return (
        df[["n", "k", "lambda"]].to_numpy(dtype=float),
        df["energy"].to_numpy(dtype=float),
        df,
    )


def kernel_family_spaces(quick: bool = False):
    """Compact grids sufficient for a reviewer-facing model-family comparison."""
    if quick:
        return {
            "Linear SVR": (
                make_scaled_svr("linear"),
                {"regressor__model__C": [10], "regressor__model__epsilon": [0.01]},
            ),
            "RBF SVR": (
                make_scaled_svr("rbf"),
                {"regressor__model__C": [10, 100], "regressor__model__gamma": ["scale"], "regressor__model__epsilon": [0.01]},
            ),
            "Polynomial SVR": (
                make_scaled_svr("poly"),
                {
                    "regressor__model__C": [0.1, 1.0],
                    "regressor__model__degree": [1, 2, 3, 4],
                    "regressor__model__gamma": ["scale"],
                    "regressor__model__coef0": [1.0],
                    "regressor__model__epsilon": [0.01],
                },
            ),
            "Random Forest": (
                RandomForestRegressor(random_state=RANDOM_STATE),
                {"n_estimators": [150], "max_depth": [None, 10], "min_samples_leaf": [1]},
            ),
            "MLP": (
                make_scaled_mlp(max_iter=400),
                {
                    "regressor__model__hidden_layer_sizes": [(16,)],
                    "regressor__model__activation": ["tanh"],
                    "regressor__model__alpha": [1e-4],
                },
            ),
            "First-order perturbation": (FirstOrderPerturbationRegressor(), {}),
            "Dummy Mean": (DummyRegressor(), {"strategy": ["mean"]}),
        }

    return {
        "Linear SVR": (
            make_scaled_svr("linear"),
            {"regressor__model__C": [0.1, 1.0], "regressor__model__epsilon": [0.01]},
        ),
        "RBF SVR": (
            make_scaled_svr("rbf"),
            {
                "regressor__model__C": [10, 100],
                "regressor__model__gamma": ["scale", 0.1],
                "regressor__model__epsilon": [0.01],
            },
        ),
        "Polynomial SVR": (
            make_scaled_svr("poly"),
            {
                "regressor__model__C": [0.1, 1.0],
                "regressor__model__degree": [1, 2, 3, 4],
                "regressor__model__gamma": ["scale"],
                "regressor__model__coef0": [1.0],
                "regressor__model__epsilon": [0.01],
            },
        ),
        "Sigmoid SVR": (
            make_scaled_svr("sigmoid"),
            {
                "regressor__model__C": [0.1, 1.0],
                "regressor__model__gamma": ["scale", 0.01],
                "regressor__model__coef0": [0.0],
                "regressor__model__epsilon": [0.01],
            },
        ),
        "Random Forest": (
            RandomForestRegressor(random_state=RANDOM_STATE),
            {
                "n_estimators": [200],
                "max_depth": [None, 10],
                "min_samples_leaf": [1, 2],
            },
        ),
        "MLP": (
            make_scaled_mlp(max_iter=600),
            {
                "regressor__model__hidden_layer_sizes": [(16,), (32,)],
                "regressor__model__activation": ["tanh"],
                "regressor__model__alpha": [1e-4, 1e-3],
            },
        ),
        "First-order perturbation": (FirstOrderPerturbationRegressor(), {}),
        "Dummy Mean": (DummyRegressor(), {"strategy": ["mean"]}),
    }


def cv_configuration(quick: bool):
    if quick:
        return (
            RepeatedKFold(n_splits=3, n_repeats=1, random_state=RANDOM_STATE),
            3,
            3,
        )
    return (
        RepeatedKFold(
            n_splits=OUTER_SPLITS,
            n_repeats=OUTER_REPEATS,
            random_state=RANDOM_STATE,
        ),
        INNER_SPLITS,
        OUTER_SPLITS * OUTER_REPEATS,
    )


def run_nested_cv(X, y, quick: bool, n_jobs: int):
    outer, inner, total_splits = cv_configuration(quick)
    spaces = kernel_family_spaces(quick)
    rows: list[dict[str, object]] = []
    best: dict[str, list[dict[str, object]]] = {name: [] for name in spaces}

    for split_id, (tr, te) in enumerate(outer.split(X, y), 1):
        print(f"[model benchmark] outer split {split_id}/{total_splits}")
        for name, (estimator, grid) in spaces.items():
            start = time.perf_counter()
            if grid:
                search = GridSearchCV(
                    estimator,
                    grid,
                    scoring="neg_mean_squared_error",
                    cv=inner,
                    n_jobs=n_jobs,
                    refit=True,
                    error_score="raise",
                )
                search.fit(X[tr], y[tr])
                model = search.best_estimator_
                params = search.best_params_
            else:
                model = clone(estimator).fit(X[tr], y[tr])
                params = {}

            fit_time = time.perf_counter() - start
            start = time.perf_counter()
            pred = model.predict(X[te])
            pred_time = time.perf_counter() - start
            fitted = model.regressor_ if hasattr(model, "regressor_") else model
            candidate = (
                fitted.named_steps["model"]
                if hasattr(fitted, "named_steps") and "model" in fitted.named_steps
                else fitted
            )
            sv_fraction = (
                float(len(candidate.support_) / len(tr))
                if hasattr(candidate, "support_")
                else math.nan
            )
            rows.append(
                {
                    "Model": name,
                    "Split": split_id,
                    "MSE": mean_squared_error(y[te], pred),
                    "R2": r2_score(y[te], pred),
                    "FitTimeSec": fit_time,
                    "PredictTimeSec": pred_time,
                    "SupportVectorFraction": sv_fraction,
                }
            )
            best[name].append(params)

    return pd.DataFrame(rows), best


def degree_ablation(X, y, quick: bool, n_jobs: int):
    """Compare degrees 1--4 with modest inner-loop tuning.

    Four degrees are sufficient to test and reject the previous post-hoc claim
    that degree 3 is dictated by the cubic force.  The degree is evaluated as a
    hyperparameter, not derived from the force gradient.
    """
    degrees = [1, 2, 3, 4]
    outer, inner, total_splits = cv_configuration(quick)
    rows: list[dict[str, object]] = []

    for degree in degrees:
        print(f"[degree ablation] degree {degree}/{degrees[-1]}")
        estimator = make_scaled_svr(
            "poly", degree=degree, gamma="scale", coef0=1.0
        )
        grid = (
            {"regressor__model__C": [0.1, 1.0], "regressor__model__epsilon": [0.01]}
            if quick
            else {"regressor__model__C": [0.1, 1.0], "regressor__model__epsilon": [0.01]}
        )
        for split_id, (tr, te) in enumerate(outer.split(X, y), 1):
            search = GridSearchCV(
                estimator,
                grid,
                scoring="neg_mean_squared_error",
                cv=inner,
                n_jobs=n_jobs,
                refit=True,
                error_score="raise",
            ).fit(X[tr], y[tr])
            pred = search.predict(X[te])
            rows.append(
                {
                    "Degree": degree,
                    "Split": split_id,
                    "MSE": mean_squared_error(y[te], pred),
                    "R2": r2_score(y[te], pred),
                    "BestParams": json.dumps(search.best_params_, sort_keys=True),
                }
            )
        print(f"  completed {total_splits} outer splits")

    folds = pd.DataFrame(rows)
    summary_rows: list[dict[str, float | int]] = []
    for degree, subset in folds.groupby("Degree"):
        mse_stats = summary95(subset["MSE"])
        r2_stats = summary95(subset["R2"])
        summary_rows.append(
            {
                "Degree": int(degree),
                "MSE_Mean": mse_stats["mean"],
                "MSE_CI95": mse_stats["ci95"],
                "R2_Mean": r2_stats["mean"],
                "R2_CI95": r2_stats["ci95"],
            }
        )
    return folds, pd.DataFrame(summary_rows)


def plot_degree_ablation(df: pd.DataFrame, path: Path):
    x = np.arange(len(df))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].bar(x, df["MSE_Mean"], yerr=df["MSE_CI95"], capsize=4)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("MSE")
    axes[1].bar(x, df["R2_Mean"], yerr=df["R2_CI95"], capsize=4)
    axes[1].set_ylabel(r"$R^2$")
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(df["Degree"])
        axis.set_xlabel("Polynomial-kernel degree")
    fig.suptitle("Anharmonic-oscillator polynomial-degree ablation")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def perturbation_diagnostic(X, y, path: Path):
    pred = FirstOrderPerturbationRegressor().predict(X)
    residual = y - pred
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].scatter(pred, y, s=12, alpha=0.6)
    lo, hi = min(pred.min(), y.min()), max(pred.max(), y.max())
    axes[0].plot([lo, hi], [lo, hi], linestyle="--")
    axes[0].set_xlabel("First-order perturbative energy")
    axes[0].set_ylabel("Numerical energy")
    axes[1].scatter(X[:, 2], residual, s=12, alpha=0.6)
    axes[1].axhline(0.0, linestyle="--")
    axes[1].set_xlabel(r"Quartic coupling $\lambda$")
    axes[1].set_ylabel("Numerical - first-order energy")
    fig.suptitle("Perturbation-theory diagnostic; no force-gradient degree claim")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="anharmonic_round2_results")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--quick", action="store_true", help="Reduced execution check; not for submission")
    parser.add_argument("--refresh-data", action="store_true", help="Regenerate the cached numerical dataset")
    parser.add_argument("--n-jobs", type=int, default=1, help="GridSearchCV parallel jobs")
    args = parser.parse_args(argv)

    out = ensure_output_dir(args.output_dir)
    n_samples = 60 if args.quick and args.samples == DEFAULT_SAMPLES else args.samples
    if n_samples < 40:
        raise ValueError("Use at least 40 samples so that nested CV remains meaningful.")

    X, y, raw = generate_dataset(
        n_samples,
        out / "anharmonic_round2_dataset.csv",
        refresh=args.refresh_data,
    )

    folds, best = run_nested_cv(X, y, args.quick, args.n_jobs)
    summary = aggregate_fold_metrics(folds)
    folds.to_csv(out / "anharmonic_round2_cv_folds.csv", index=False)
    summary.to_csv(out / "anharmonic_round2_cv_summary.csv", index=False)
    save_json(best, out / "anharmonic_round2_best_params.json")
    plot_metric_bars(
        summary,
        out / "anharmonic_round2_kernel_benchmark.png",
        "Anharmonic-oscillator repeated nested-CV benchmark",
    )

    degree_folds, degree_summary = degree_ablation(X, y, args.quick, args.n_jobs)
    degree_folds.to_csv(out / "anharmonic_round2_degree_folds.csv", index=False)
    degree_summary.to_csv(out / "anharmonic_round2_degree_summary.csv", index=False)
    plot_degree_ablation(degree_summary, out / "anharmonic_round2_degree_ablation.png")
    perturbation_diagnostic(X, y, out / "anharmonic_round2_perturbation_diagnostic.png")

    first_order = FirstOrderPerturbationRegressor().predict(X)
    diagnostics = {
        "hamiltonian": "H = -1/2 d^2/dx^2 + 1/2 k x^2 + lambda x^4 (hbar=m=1)",
        "first_order_energy": "sqrt(k)(n+1/2) + [3 lambda/(4 k)](2 n^2+2 n+1)",
        "degree_policy": "No degree is inferred from the cubic force. Degrees 1-4 are compared by repeated nested CV.",
        "outer_cv": f"{3 if args.quick else OUTER_SPLITS} folds x {1 if args.quick else OUTER_REPEATS} repeats",
        "inner_cv": 3 if args.quick else INNER_SPLITS,
        "n_samples": int(len(X)),
        "perturbation_MSE_full_sample": float(mean_squared_error(y, first_order)),
        "perturbation_R2_full_sample": float(r2_score(y, first_order)),
        "mean_numerical_solve_time_sec": float(raw["solve_time_sec"].mean()),
        "learning_curves": "Not generated for this proof-of-concept task; the manuscript Figure 3 requires degree ablation, model-family benchmark, and perturbation diagnostic.",
    }
    save_json(diagnostics, out / "anharmonic_round2_physics_diagnostics.json")

    print("\nCompleted reviewer-focused anharmonic benchmark.")
    print(f"Results saved in: {out.resolve()}")
    print("Generated manuscript figures:")
    for filename in (
        "anharmonic_round2_degree_ablation.png",
        "anharmonic_round2_kernel_benchmark.png",
        "anharmonic_round2_perturbation_diagnostic.png",
    ):
        print(f"  - {out / filename}")


if __name__ == "__main__":
    main()
