#!/usr/bin/env python3
"""Shared utilities for the second-round COMPHY revision benchmarks.

The module is intentionally lightweight and uses only NumPy, pandas,
SciPy, matplotlib, and scikit-learn.  It provides:

* repeated-CV summary statistics and 95% confidence intervals;
* an explicit Jackson-damped Chebyshev feature map whose Gram matrix is
  positive semidefinite by construction (no cubic spectral clipping);
* an optional Nyström approximation for large data sets;
* centered kernel-target alignment as an interpretable diagnostic;
* reproducible learning-curve evaluation;
* figure and CSV helpers.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

RANDOM_STATE = 42


def ensure_output_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def summary95(values: Iterable[float]) -> dict[str, float]:
    x = np.asarray(list(values), dtype=float)
    if x.size == 0:
        return {"mean": math.nan, "std": math.nan, "ci95": math.nan, "n": 0}
    mean = float(np.mean(x))
    if x.size == 1:
        return {"mean": mean, "std": 0.0, "ci95": 0.0, "n": 1}
    std = float(np.std(x, ddof=1))
    ci = float(t.ppf(0.975, x.size - 1) * std / np.sqrt(x.size))
    return {"mean": mean, "std": std, "ci95": ci, "n": int(x.size)}


def aggregate_fold_metrics(fold_df: pd.DataFrame, model_col: str = "Model") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, sub in fold_df.groupby(model_col, sort=False):
        mse = summary95(sub["MSE"])
        r2 = summary95(sub["R2"])
        row: dict[str, object] = {
            "Model": model,
            "CV_MSE_Mean": mse["mean"],
            "CV_MSE_STD": mse["std"],
            "CV_MSE_CI95": mse["ci95"],
            "CV_R2_Mean": r2["mean"],
            "CV_R2_STD": r2["std"],
            "CV_R2_CI95": r2["ci95"],
            "NFolds": mse["n"],
        }
        for col, prefix in [
            ("FitTimeSec", "FitTime"),
            ("PredictTimeSec", "PredictTime"),
            ("KernelBuildTimeSec", "KernelBuildTime"),
            ("ApproxKernelMemoryMB", "ApproxKernelMemoryMB"),
            ("SupportVectorFraction", "SupportVectorFraction"),
        ]:
            if col in sub:
                s = summary95(sub[col])
                row[f"{prefix}_Mean"] = s["mean"]
                row[f"{prefix}_STD"] = s["std"]
                row[f"{prefix}_CI95"] = s["ci95"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("CV_MSE_Mean").reset_index(drop=True)


def save_json(obj: object, path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def center_gram(K: np.ndarray) -> np.ndarray:
    K = np.asarray(K, dtype=float)
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def centered_kernel_target_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """Centered alignment between K and yy^T.

    The statistic is in [-1, 1] up to floating-point roundoff.  A larger
    positive value indicates that the kernel geometry is more consistent
    with target similarity on the evaluated sample.
    """
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    Kc = center_gram(K)
    Yc = center_gram(y @ y.T)
    denom = np.linalg.norm(Kc, "fro") * np.linalg.norm(Yc, "fro")
    if denom <= np.finfo(float).eps:
        return 0.0
    return float(np.sum(Kc * Yc) / denom)


@dataclass
class JacksonChebyshevFeatureMap:
    degree: int = 60
    endpoint_epsilon: float = 1e-6
    jitter: float = 1e-12

    def __post_init__(self) -> None:
        if self.degree < 0:
            raise ValueError("degree must be non-negative")
        if not 0.0 < self.endpoint_epsilon < 1.0:
            raise ValueError("endpoint_epsilon must lie in (0, 1)")
        self.x_min_: float | None = None
        self.x_max_: float | None = None
        self.jackson_coeffs_: np.ndarray = self._jackson_coefficients(self.degree)

    @staticmethod
    def _jackson_coefficients(degree: int) -> np.ndarray:
        N = int(degree)
        n = np.arange(N + 1, dtype=float)
        denom = N + 1.0
        theta = np.pi / denom
        coeff = (
            (N - n + 1.0) * np.cos(n * theta)
            + np.sin(n * theta) / np.tan(theta)
        ) / denom
        # Jackson coefficients are non-negative analytically; clipping only
        # removes tiny negative roundoff.
        return np.clip(coeff, 0.0, None)

    def fit(self, X: np.ndarray) -> "JacksonChebyshevFeatureMap":
        x = np.asarray(X, dtype=float).reshape(-1)
        if x.size == 0 or not np.isfinite(x).all():
            raise ValueError("X must contain at least one finite scalar input")
        self.x_min_ = float(np.min(x))
        self.x_max_ = float(np.max(x))
        return self

    def _scale(self, X: np.ndarray) -> np.ndarray:
        if self.x_min_ is None or self.x_max_ is None:
            raise RuntimeError("fit must be called before transform")
        x = np.asarray(X, dtype=float).reshape(-1)
        span = self.x_max_ - self.x_min_
        if abs(span) <= np.finfo(float).eps:
            z = np.zeros_like(x)
        else:
            z = 2.0 * (x - self.x_min_) / span - 1.0
        bound = 1.0 - self.endpoint_epsilon
        return np.clip(z, -bound, bound)

    def transform(self, X: np.ndarray, n_modes: int | None = None) -> np.ndarray:
        z = self._scale(X)
        max_modes = self.degree + 1
        modes = max_modes if n_modes is None else int(n_modes)
        if modes < 1 or modes > max_modes:
            raise ValueError(f"n_modes must be in [1, {max_modes}]")
        orders = np.arange(modes, dtype=float)
        theta = np.arccos(z)
        T = np.cos(theta[:, None] * orders[None, :])
        weights = np.sqrt(self.jackson_coeffs_[:modes])
        Phi = T * weights[None, :]
        return np.nan_to_num(Phi, nan=0.0, posinf=0.0, neginf=0.0)

    def gram(
        self,
        X1: np.ndarray,
        X2: np.ndarray | None = None,
        n_modes: int | None = None,
        add_jitter: bool = False,
    ) -> np.ndarray:
        if X2 is None:
            X2 = X1
        Phi1 = self.transform(X1, n_modes=n_modes)
        Phi2 = self.transform(X2, n_modes=n_modes)
        K = Phi1 @ Phi2.T
        if add_jitter and X2 is X1:
            K = 0.5 * (K + K.T)
            K += self.jitter * np.eye(K.shape[0])
        return np.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)

    def min_eigenvalue(self, X: np.ndarray, n_modes: int | None = None) -> float:
        K = self.gram(X, n_modes=n_modes, add_jitter=True)
        return float(np.linalg.eigvalsh(0.5 * (K + K.T))[0])


@dataclass
class NystromFeatureMap:
    """Nyström approximation for a fitted Jackson-Chebyshev kernel.

    The approximation uses uniformly sampled training landmarks and returns
    an explicit low-rank feature map Z = K(X,L) W^{-1/2}.  Consequently,
    Z Z^T is PSD by construction.  Only the landmark matrix W is decomposed,
    giving O(m^3) rather than O(n^3) decomposition cost for m << n.
    """

    base_kernel: JacksonChebyshevFeatureMap
    n_landmarks: int
    random_state: int = RANDOM_STATE
    eigen_tol: float = 1e-12

    def fit(self, X: np.ndarray) -> "NystromFeatureMap":
        X = np.asarray(X, dtype=float)
        n = len(X)
        m = min(int(self.n_landmarks), n)
        if m < 1:
            raise ValueError("n_landmarks must be positive")
        rng = np.random.default_rng(self.random_state)
        self.landmark_indices_ = np.sort(rng.choice(n, size=m, replace=False))
        self.landmarks_ = X[self.landmark_indices_]
        W = self.base_kernel.gram(self.landmarks_, self.landmarks_, add_jitter=True)
        vals, vecs = np.linalg.eigh(0.5 * (W + W.T))
        keep = vals > max(self.eigen_tol, self.eigen_tol * float(np.max(vals)))
        if not np.any(keep):
            raise np.linalg.LinAlgError("Nyström landmark matrix has no retained positive eigenvalues")
        self.eigenvalues_ = vals[keep]
        self.eigenvectors_ = vecs[:, keep]
        self.W_inv_sqrt_ = self.eigenvectors_ @ np.diag(self.eigenvalues_ ** -0.5)
        self.effective_rank_ = int(np.sum(keep))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        C = self.base_kernel.gram(np.asarray(X, dtype=float), self.landmarks_)
        return C @ self.W_inv_sqrt_


@dataclass
class TimedPrediction:
    y_pred: np.ndarray
    fit_time: float
    predict_time: float
    support_vector_fraction: float = math.nan


def timed_fit_predict(model, X_train, y_train, X_test) -> TimedPrediction:
    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start
    start = time.perf_counter()
    pred = model.predict(X_test)
    predict_time = time.perf_counter() - start
    sv_fraction = math.nan
    candidate = model
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        candidate = model.named_steps["model"]
    if hasattr(candidate, "support_"):
        sv_fraction = float(len(candidate.support_) / max(1, len(y_train)))
    return TimedPrediction(np.asarray(pred), fit_time, predict_time, sv_fraction)


def evaluate_learning_curve(
    model_factory: Callable[[], object],
    X: np.ndarray,
    y: np.ndarray,
    train_sizes: Sequence[int],
    n_repeats: int = 10,
    test_fraction: float = 0.25,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Repeated subsampling learning curve with a fixed test fraction.

    The model factory should return a fully configured estimator.  Tuning is
    intentionally performed before this routine; otherwise learning-curve
    differences would conflate sample size with different search budgets.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, float | int]] = []
    n = len(X)
    n_test = max(1, int(round(test_fraction * n)))
    for repeat in range(n_repeats):
        perm = rng.permutation(n)
        test_idx = perm[:n_test]
        pool_idx = perm[n_test:]
        for size in train_sizes:
            size_eff = min(int(size), len(pool_idx))
            if size_eff < 2:
                continue
            train_idx = pool_idx[:size_eff]
            model = model_factory()
            out = timed_fit_predict(model, X[train_idx], y[train_idx], X[test_idx])
            rows.append(
                {
                    "Repeat": repeat + 1,
                    "TrainSize": size_eff,
                    "MSE": mean_squared_error(y[test_idx], out.y_pred),
                    "R2": r2_score(y[test_idx], out.y_pred),
                    "FitTimeSec": out.fit_time,
                    "PredictTimeSec": out.predict_time,
                    "SupportVectorFraction": out.support_vector_fraction,
                }
            )
    return pd.DataFrame(rows)


def summarize_learning_curve(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for size, sub in df.groupby("TrainSize"):
        mse = summary95(sub["MSE"])
        r2 = summary95(sub["R2"])
        fit = summary95(sub["FitTimeSec"])
        rows.append(
            {
                "TrainSize": int(size),
                "MSE_Mean": mse["mean"],
                "MSE_CI95": mse["ci95"],
                "R2_Mean": r2["mean"],
                "R2_CI95": r2["ci95"],
                "FitTime_MeanSec": fit["mean"],
                "FitTime_CI95": fit["ci95"],
            }
        )
    return pd.DataFrame(rows).sort_values("TrainSize")


def plot_metric_bars(summary: pd.DataFrame, path: str | Path, title: str) -> None:
    labels = summary["Model"].tolist()
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].bar(x, summary["CV_MSE_Mean"], yerr=summary["CV_MSE_CI95"], capsize=4)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("MSE (mean and 95% CI)")
    axes[0].set_title("Mean squared error")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].bar(x, summary["CV_R2_Mean"], yerr=summary["CV_R2_CI95"], capsize=4)
    axes[1].set_ylabel(r"$R^2$ (mean and 95% CI)")
    axes[1].set_title(r"Coefficient of determination")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curves(curves: Mapping[str, pd.DataFrame], path: str | Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for label, df in curves.items():
        axes[0].errorbar(df["TrainSize"], df["MSE_Mean"], yerr=df["MSE_CI95"], marker="o", label=label)
        axes[1].errorbar(df["TrainSize"], df["R2_Mean"], yerr=df["R2_CI95"], marker="o", label=label)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Training samples")
    axes[0].set_ylabel("MSE")
    axes[1].set_xlabel("Training samples")
    axes[1].set_ylabel(r"$R^2$")
    axes[0].legend()
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def simple_kfold(n_samples: int, n_splits: int, random_state: int = RANDOM_STATE):
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(np.arange(n_samples))
