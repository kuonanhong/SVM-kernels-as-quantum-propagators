#!/usr/bin/env python3
"""Second-round COMPHY benchmark for the 1D photonic-crystal task.

Key revisions responding to Reviewer 3:
1. Adds an MLP baseline and repeated learning curves to delimit the finite-data
   setting in which kernel methods are useful.
2. Removes the O(n^3) full-matrix spectral-clipping step.  The Jackson-
   Chebyshev Gram matrix is PSD by construction because K = Phi Phi^T.
3. Adds Nyström rank/accuracy/time/memory trade-off experiments.
4. Saves every metric, configuration, and figure automatically.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, ParameterGrid, RepeatedKFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVR

from svmprop_common import (
    JacksonChebyshevFeatureMap,
    NystromFeatureMap,
    aggregate_fold_metrics,
    centered_kernel_target_alignment,
    ensure_output_dir,
    evaluate_learning_curve,
    plot_learning_curves,
    plot_metric_bars,
    save_json,
    simple_kfold,
    summarize_learning_curve,
    timed_fit_predict,
)

RANDOM_STATE = 42


# -----------------------------------------------------------------------------
# Physics simulation
# -----------------------------------------------------------------------------
def n_si(energy_ev: float | np.ndarray) -> np.ndarray:
    energy_ev = np.asarray(energy_ev, dtype=float)
    wavelength_um = 1.239841984 / np.maximum(energy_ev, 1e-9)
    epsilon = 11.6858 + 0.939816 / wavelength_um**2 + 8.10461e-3 * wavelength_um**2
    return np.sqrt(np.maximum(epsilon, 1e-12))


def n_sio2(energy_ev: float | np.ndarray) -> np.ndarray:
    return np.full_like(np.asarray(energy_ev, dtype=float), 1.45)


def build_quarter_wave_stack(n_periods: int = 10, center_energy_ev: float = 1.24):
    lambda0_um = 1.239841984 / center_energy_ev
    d_si = lambda0_um / (4.0 * float(n_si(center_energy_ev)))
    d_sio2 = lambda0_um / (4.0 * float(n_sio2(center_energy_ev)))
    layers = []
    for _ in range(n_periods):
        layers.extend([(n_si, d_si), (n_sio2, d_sio2)])
    return layers


def transfer_matrix_transmission(energies_ev: np.ndarray, n_periods: int = 10) -> np.ndarray:
    """Normal-incidence characteristic-matrix calculation in air.

    Absorption is deliberately neglected, matching the assumptions stated in
    the manuscript.  With incident and exit refractive indices both equal to
    one, T = |2/(M11+M12+M21+M22)|^2.
    """
    layers = build_quarter_wave_stack(n_periods=n_periods)
    values: list[float] = []
    for energy in np.asarray(energies_ev, dtype=float):
        wavelength_um = 1.239841984 / energy
        k0 = 2.0 * np.pi / wavelength_um
        M = np.eye(2, dtype=complex)
        for n_func, thickness_um in layers:
            n = float(n_func(energy))
            phase = n * k0 * thickness_um
            layer = np.array(
                [
                    [np.cos(phase), -1j * np.sin(phase) / n],
                    [-1j * n * np.sin(phase), np.cos(phase)],
                ],
                dtype=complex,
            )
            M = M @ layer
        denominator = M[0, 0] + M[0, 1] + M[1, 0] + M[1, 1]
        transmission = abs(2.0 / denominator) ** 2
        values.append(float(np.clip(transmission.real, 0.0, 1.0 + 1e-9)))
    return np.asarray(values)


def generate_dataset(n_samples: int = 500) -> tuple[np.ndarray, np.ndarray]:
    energies = np.linspace(0.5, 3.0, n_samples)
    return energies.reshape(-1, 1), transfer_matrix_transmission(energies)


# -----------------------------------------------------------------------------
# PSD-by-construction Jackson-Chebyshev estimators
# -----------------------------------------------------------------------------
class JacksonSVR(BaseEstimator, RegressorMixin):
    def __init__(self, degree: int = 60, C: float = 10.0, epsilon: float = 0.01, endpoint_epsilon: float = 1e-6):
        self.degree = degree
        self.C = C
        self.epsilon = epsilon
        self.endpoint_epsilon = endpoint_epsilon

    def fit(self, X, y):
        self.X_train_ = np.asarray(X, dtype=float)
        self.feature_map_ = JacksonChebyshevFeatureMap(
            degree=int(self.degree), endpoint_epsilon=float(self.endpoint_epsilon)
        ).fit(self.X_train_)
        K = self.feature_map_.gram(self.X_train_, add_jitter=True)
        self.model_ = SVR(kernel="precomputed", C=float(self.C), epsilon=float(self.epsilon))
        self.model_.fit(K, np.asarray(y, dtype=float))
        self.support_vector_fraction_ = len(self.model_.support_) / len(self.X_train_)
        return self

    def predict(self, X):
        K = self.feature_map_.gram(np.asarray(X, dtype=float), self.X_train_)
        return self.model_.predict(K)


class ExplicitJacksonLinearSVR(BaseEstimator, RegressorMixin):
    """Linear epsilon-SVR on the exact Jackson-Chebyshev features.

    This estimator is used as the solver-matched reference in the Nyström
    rank study, so differences from NystromJacksonSVR reflect the feature
    approximation rather than a change from a kernel solver to a linear one.
    """

    def __init__(self, degree: int = 60, C: float = 10.0, epsilon: float = 0.01, random_state: int = RANDOM_STATE):
        self.degree = degree
        self.C = C
        self.epsilon = epsilon
        self.random_state = random_state

    def fit(self, X, y):
        self.X_train_ = np.asarray(X, dtype=float)
        self.feature_map_ = JacksonChebyshevFeatureMap(degree=int(self.degree)).fit(self.X_train_)
        self.Phi_train_ = self.feature_map_.transform(self.X_train_)
        self.model_ = LinearSVR(
            C=float(self.C), epsilon=float(self.epsilon),
            loss="squared_epsilon_insensitive", dual="auto",
            random_state=int(self.random_state), max_iter=20000,
        )
        self.model_.fit(self.Phi_train_, np.asarray(y, dtype=float))
        residual = np.abs(np.asarray(y, dtype=float) - self.model_.predict(self.Phi_train_))
        self.support_vector_fraction_ = float(np.mean(residual >= float(self.epsilon)))
        return self

    def predict(self, X):
        return self.model_.predict(self.feature_map_.transform(np.asarray(X, dtype=float)))


class NystromJacksonSVR(BaseEstimator, RegressorMixin):
    def __init__(self, degree: int = 60, n_landmarks: int = 32, C: float = 10.0, epsilon: float = 0.01, random_state: int = RANDOM_STATE):
        self.degree = degree
        self.n_landmarks = n_landmarks
        self.C = C
        self.epsilon = epsilon
        self.random_state = random_state

    def fit(self, X, y):
        self.X_train_ = np.asarray(X, dtype=float)
        self.base_kernel_ = JacksonChebyshevFeatureMap(degree=int(self.degree)).fit(self.X_train_)
        self.nystrom_ = NystromFeatureMap(
            self.base_kernel_, n_landmarks=int(self.n_landmarks), random_state=int(self.random_state)
        ).fit(self.X_train_)
        Z = self.nystrom_.transform(self.X_train_)
        # A linear epsilon-SVR on the explicit Nyström features avoids forming
        # an n-by-n approximate Gram matrix.  The stored representation is
        # therefore O(nm), and only the m-by-m landmark matrix is decomposed.
        self.model_ = LinearSVR(
            C=float(self.C),
            epsilon=float(self.epsilon),
            loss="squared_epsilon_insensitive",
            dual="auto",
            random_state=int(self.random_state),
            max_iter=20000,
        )
        self.model_.fit(Z, np.asarray(y, dtype=float))
        self.Z_train_ = Z
        train_residual = np.abs(np.asarray(y, dtype=float) - self.model_.predict(Z))
        self.support_vector_fraction_ = float(np.mean(train_residual >= float(self.epsilon)))
        return self

    def predict(self, X):
        Z = self.nystrom_.transform(np.asarray(X, dtype=float))
        return self.model_.predict(Z)


# -----------------------------------------------------------------------------
# Model search and repeated CV
# -----------------------------------------------------------------------------
def model_spaces(quick: bool = False):
    if quick:
        return {
            "Linear SVR": (Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="linear"))]), {"model__C": [1, 10], "model__epsilon": [0.01]}),
            "RBF SVR": (Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="rbf"))]), {"model__C": [10, 100], "model__gamma": ["scale", 1.0], "model__epsilon": [0.01]}),
            "Polynomial SVR": (Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="poly"))]), {"model__C": [10], "model__degree": [2, 3], "model__gamma": ["scale"], "model__coef0": [1.0], "model__epsilon": [0.01]}),
            "Sigmoid SVR": (Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="sigmoid"))]), {"model__C": [1], "model__gamma": [0.1], "model__coef0": [0.0], "model__epsilon": [0.01]}),
            "Random Forest": (RandomForestRegressor(random_state=RANDOM_STATE), {"n_estimators": [150], "max_depth": [None, 10]}),
            "MLP": (Pipeline([("scale", StandardScaler()), ("model", MLPRegressor(random_state=RANDOM_STATE, max_iter=600, early_stopping=True, n_iter_no_change=25))]), {"model__hidden_layer_sizes": [(16, 16)], "model__alpha": [1e-4], "model__learning_rate_init": [1e-3]}),
            "Dummy Mean": (DummyRegressor(), {"strategy": ["mean"]}),
        }
    return {
        "Linear SVR": (Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="linear"))]), {"model__C": [0.1, 1, 10, 100], "model__epsilon": [0.001, 0.01, 0.05]}),
        "RBF SVR": (Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="rbf"))]), {"model__C": [1, 10, 100], "model__gamma": ["scale", 0.1, 1.0, 10.0], "model__epsilon": [0.001, 0.01, 0.05]}),
        "Polynomial SVR": (Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="poly"))]), {"model__C": [1, 10, 100], "model__degree": [2, 3, 4], "model__gamma": ["scale", 0.1, 1.0], "model__coef0": [0.0, 1.0], "model__epsilon": [0.001, 0.01]}),
        "Sigmoid SVR": (Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="sigmoid"))]), {"model__C": [0.1, 1, 10], "model__gamma": ["scale", 0.01, 0.1], "model__coef0": [-1.0, 0.0, 1.0], "model__epsilon": [0.001, 0.01]}),
        "Random Forest": (RandomForestRegressor(random_state=RANDOM_STATE), {"n_estimators": [200, 400], "max_depth": [None, 8, 16], "min_samples_leaf": [1, 2]}),
        "MLP": (Pipeline([("scale", StandardScaler()), ("model", MLPRegressor(random_state=RANDOM_STATE, max_iter=3000, early_stopping=True, n_iter_no_change=50))]), {"model__hidden_layer_sizes": [(32, 32), (64, 64)], "model__activation": ["tanh", "relu"], "model__alpha": [1e-4, 1e-3], "model__learning_rate_init": [1e-3]}),
        "Dummy Mean": (DummyRegressor(), {"strategy": ["mean"]}),
    }


def tune_custom(X, y, quick: bool, inner_folds: int):
    grid = {
        "degree": [30, 60] if quick else [30, 45, 60, 75, 90],
        "C": [10, 100] if quick else [1, 10, 100, 1000],
        "epsilon": [0.01] if quick else [0.001, 0.01, 0.05],
    }
    best_score = math.inf
    best = None
    for params in ParameterGrid(grid):
        errors = []
        for tr, va in simple_kfold(len(X), inner_folds):
            model = JacksonSVR(**params)
            model.fit(X[tr], y[tr])
            errors.append(mean_squared_error(y[va], model.predict(X[va])))
        score = float(np.mean(errors))
        if score < best_score:
            best_score, best = score, params
    return best, best_score


def run_nested_cv(X, y, quick: bool):
    outer = RepeatedKFold(n_splits=3 if quick else 5, n_repeats=1 if quick else 2, random_state=RANDOM_STATE)
    inner = 3 if quick else 4
    folds = []
    best_history: dict[str, list[dict]] = {"Jackson-Chebyshev SVR": []}
    spaces = model_spaces(quick)
    for name in spaces:
        best_history[name] = []
    for split_id, (tr, te) in enumerate(outer.split(X, y), 1):
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        for name, (estimator, grid) in spaces.items():
            search = GridSearchCV(estimator, grid, scoring="neg_mean_squared_error", cv=inner, n_jobs=1 if quick else -1, refit=True)
            start = time.perf_counter(); search.fit(Xtr, ytr); fit_time = time.perf_counter() - start
            start = time.perf_counter(); pred = search.predict(Xte); pred_time = time.perf_counter() - start
            candidate = search.best_estimator_
            svfrac = math.nan
            if hasattr(candidate, "named_steps") and hasattr(candidate.named_steps.get("model"), "support_"):
                svfrac = len(candidate.named_steps["model"].support_) / len(ytr)
            best_history[name].append(search.best_params_)
            folds.append({"Model": name, "Split": split_id, "MSE": mean_squared_error(yte, pred), "R2": r2_score(yte, pred), "FitTimeSec": fit_time, "PredictTimeSec": pred_time, "KernelBuildTimeSec": 0.0, "ApproxKernelMemoryMB": 0.0, "SupportVectorFraction": svfrac})

        params, _ = tune_custom(Xtr, ytr, quick, inner)
        model = JacksonSVR(**params)
        start = time.perf_counter(); model.fit(Xtr, ytr); total_fit = time.perf_counter() - start
        start = time.perf_counter(); pred = model.predict(Xte); pred_time = time.perf_counter() - start
        fmap = model.feature_map_
        K = fmap.gram(Xtr, add_jitter=True)
        alignment = centered_kernel_target_alignment(K, ytr)
        best_history["Jackson-Chebyshev SVR"].append({**params, "training_alignment": alignment})
        folds.append({"Model": "Jackson-Chebyshev SVR", "Split": split_id, "MSE": mean_squared_error(yte, pred), "R2": r2_score(yte, pred), "FitTimeSec": total_fit, "PredictTimeSec": pred_time, "KernelBuildTimeSec": 0.0, "ApproxKernelMemoryMB": K.nbytes / 1024**2, "SupportVectorFraction": model.support_vector_fraction_})
    return pd.DataFrame(folds), best_history


# -----------------------------------------------------------------------------
# Reviewer-3 scalability and sample-efficiency analyses
# -----------------------------------------------------------------------------
def rank_tradeoff(X, y, best_custom: dict, quick: bool) -> pd.DataFrame:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)
    full = ExplicitJacksonLinearSVR(**best_custom)
    t0 = time.perf_counter(); full.fit(Xtr, ytr); full_fit = time.perf_counter() - t0
    pred = full.predict(Xte)
    K_full = full.feature_map_.gram(Xtr, add_jitter=True)
    rows = [{"Approximation": "Exact explicit features", "Landmarks": len(Xtr), "EffectiveRank": min(best_custom["degree"] + 1, len(Xtr)), "RelativeFrobeniusError": 0.0, "MinEigenvalue": float(np.linalg.eigvalsh(K_full)[0]), "MSE": mean_squared_error(yte, pred), "R2": r2_score(yte, pred), "FitTimeSec": full_fit, "KernelMemoryMB": full.Phi_train_.nbytes / 1024**2}]
    landmarks = [8, 16, 32] if quick else [8, 16, 32, 64, 128]
    for m in landmarks:
        if m >= len(Xtr):
            continue
        model = NystromJacksonSVR(degree=best_custom["degree"], n_landmarks=m, C=best_custom["C"], epsilon=best_custom["epsilon"])
        t0 = time.perf_counter(); model.fit(Xtr, ytr); fit = time.perf_counter() - t0
        p = model.predict(Xte)
        K_approx = model.Z_train_ @ model.Z_train_.T
        rel = np.linalg.norm(K_full - K_approx, "fro") / max(np.linalg.norm(K_full, "fro"), 1e-15)
        rows.append({"Approximation": "Nystrom", "Landmarks": m, "EffectiveRank": model.nystrom_.effective_rank_, "RelativeFrobeniusError": rel, "MinEigenvalue": float(np.linalg.eigvalsh(0.5 * (K_approx + K_approx.T))[0]), "MSE": mean_squared_error(yte, p), "R2": r2_score(yte, p), "FitTimeSec": fit, "KernelMemoryMB": model.Z_train_.nbytes / 1024**2})
    return pd.DataFrame(rows)


def plot_rank_tradeoff(df: pd.DataFrame, path: Path):
    x = np.arange(len(df))
    labels = ["Exact"] + [f"m={m}" for m in df["Landmarks"].iloc[1:]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].bar(x, df["MSE"]); axes[0].set_yscale("log"); axes[0].set_ylabel("Hold-out MSE")
    axes[1].bar(x, df["FitTimeSec"]); axes[1].set_ylabel("Fit time (s)")
    axes[2].bar(x, df["RelativeFrobeniusError"]); axes[2].set_ylabel("Relative Gram error")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
    fig.suptitle("Photonic-crystal Nyström rank trade-off")
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def scalability_benchmark(best_degree: int, quick: bool) -> pd.DataFrame:
    sizes = [100, 250, 500, 1000] if quick else [250, 500, 1000, 2000, 4000]
    rows = []
    for n in sizes:
        X, _ = generate_dataset(n)
        fmap = JacksonChebyshevFeatureMap(degree=best_degree).fit(X)
        t0 = time.perf_counter(); Phi = fmap.transform(X); feature_time = time.perf_counter() - t0
        t0 = time.perf_counter(); K = Phi @ Phi.T; gram_time = time.perf_counter() - t0
        rows.append({"N": n, "Method": "Exact Gram", "FeatureTimeSec": feature_time, "KernelTimeSec": gram_time, "MemoryMB": K.nbytes / 1024**2, "StoredValues": K.size})
        m = min(64, n)
        t0 = time.perf_counter(); nys = NystromFeatureMap(fmap, n_landmarks=m).fit(X); Z = nys.transform(X); nys_time = time.perf_counter() - t0
        rows.append({"N": n, "Method": f"Nystrom m={m}", "FeatureTimeSec": feature_time, "KernelTimeSec": nys_time, "MemoryMB": Z.nbytes / 1024**2, "StoredValues": Z.size})
    return pd.DataFrame(rows)


def plot_scalability(df: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for method, sub in df.groupby("Method"):
        axes[0].plot(sub["N"], sub["KernelTimeSec"], marker="o", label=method)
        axes[1].plot(sub["N"], sub["MemoryMB"], marker="o", label=method)
    axes[0].set_xlabel("Number of samples"); axes[0].set_ylabel("Kernel construction time (s)"); axes[0].set_yscale("log")
    axes[1].set_xlabel("Number of samples"); axes[1].set_ylabel("Stored representation (MB)"); axes[1].set_yscale("log")
    axes[0].legend(); axes[1].legend(); fig.suptitle("Photonic-crystal kernel scalability")
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def learning_curves(X, y, tuned_models: dict[str, object], quick: bool):
    max_train = int(0.75 * len(X))
    sizes = np.unique(np.linspace(max(20, int(0.1 * len(X))), max_train, 4 if quick else 7).astype(int))
    summaries = {}
    raw_frames = []
    for name, model in tuned_models.items():
        raw = evaluate_learning_curve(lambda m=model: clone(m), X, y, sizes, n_repeats=3 if quick else 10)
        raw["Model"] = name
        raw_frames.append(raw)
        summaries[name] = summarize_learning_curve(raw)
    return pd.concat(raw_frames, ignore_index=True), summaries


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="photonic_round2_results")
    parser.add_argument("--quick", action="store_true", help="Reduced smoke-test configuration")
    args = parser.parse_args(argv)
    out = ensure_output_dir(args.output_dir)
    X, y = generate_dataset(120 if args.quick else 500)

    fold_df, best_history = run_nested_cv(X, y, args.quick)
    summary = aggregate_fold_metrics(fold_df)
    fold_df.to_csv(out / "photonic_round2_cv_folds.csv", index=False)
    summary.to_csv(out / "photonic_round2_cv_summary.csv", index=False)
    save_json(best_history, out / "photonic_round2_best_params.json")
    plot_metric_bars(summary, out / "photonic_round2_benchmark.png", "Photonic-crystal repeated nested-CV benchmark")

    # Tune representative models on the full data for learning curves and rank tests.
    inner = 3 if args.quick else 5
    custom_params, _ = tune_custom(X, y, args.quick, inner)
    spaces = model_spaces(args.quick)
    tuned = {"Jackson-Chebyshev SVR": JacksonSVR(**custom_params)}
    for name in ["RBF SVR", "Random Forest", "MLP"]:
        est, grid = spaces[name]
        search = GridSearchCV(est, grid, scoring="neg_mean_squared_error", cv=inner, n_jobs=1 if args.quick else -1, refit=True).fit(X, y)
        tuned[name] = search.best_estimator_

    raw_learning, learning_summary = learning_curves(X, y, tuned, args.quick)
    raw_learning.to_csv(out / "photonic_round2_learning_curve_folds.csv", index=False)
    pd.concat([df.assign(Model=name) for name, df in learning_summary.items()], ignore_index=True).to_csv(out / "photonic_round2_learning_curve_summary.csv", index=False)
    plot_learning_curves(learning_summary, out / "photonic_round2_learning_curves.png", "Photonic-crystal sample-efficiency comparison")

    rank_df = rank_tradeoff(X, y, custom_params, args.quick)
    rank_df.to_csv(out / "photonic_round2_nystrom_rank_tradeoff.csv", index=False)
    plot_rank_tradeoff(rank_df, out / "photonic_round2_nystrom_rank_tradeoff.png")

    scaling = scalability_benchmark(custom_params["degree"], args.quick)
    scaling.to_csv(out / "photonic_round2_scalability.csv", index=False)
    plot_scalability(scaling, out / "photonic_round2_scalability.png")

    # Kernel interpretability diagnostic on the full sample.
    fmap = JacksonChebyshevFeatureMap(degree=custom_params["degree"]).fit(X)
    K = fmap.gram(X, add_jitter=True)
    diagnostics = {
        "custom_parameters": custom_params,
        "minimum_eigenvalue": float(np.linalg.eigvalsh(K)[0]),
        "centered_kernel_target_alignment": centered_kernel_target_alignment(K, y),
        "feature_dimension": int(custom_params["degree"] + 1),
        "full_gram_complexity": "O(n^2 D) time and O(n^2) memory; no full eigendecomposition",
        "nystrom_complexity": "O(n m D + m^3) feature/decomposition cost and O(n m) stored features for m landmarks; the approximation is trained with LinearSVR without constructing an n-by-n Gram matrix",
    }
    save_json(diagnostics, out / "photonic_round2_kernel_diagnostics.json")

    print(f"Completed. Results saved in: {out.resolve()}")


if __name__ == "__main__":
    main()
