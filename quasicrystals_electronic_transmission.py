#!/usr/bin/env python3
"""Second-round COMPHY benchmark for Fibonacci-chain transmission.

The physical data generator is revised to a boundary-conditioned Landauer-
Green-function calculation for a finite Fibonacci tight-binding chain coupled
to two semi-infinite leads.  The ML benchmark uses the PSD-by-construction
Jackson-Chebyshev feature map and adds Nyström scalability and learning-curve
experiments required by Reviewer 3.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV

from svmprop_common import (
    JacksonChebyshevFeatureMap,
    NystromFeatureMap,
    aggregate_fold_metrics,
    centered_kernel_target_alignment,
    ensure_output_dir,
    plot_learning_curves,
    plot_metric_bars,
    save_json,
)
from photonic_crystals_revised_round2 import (
    JacksonSVR,
    learning_curves,
    model_spaces,
    plot_rank_tradeoff,
    rank_tradeoff,
    run_nested_cv,
    tune_custom,
)

RANDOM_STATE = 42


def fibonacci_word(generation: int = 8) -> str:
    """Return S_generation for S_0=A, S_1=AB, S_n=S_{n-1}S_{n-2}."""
    if generation < 0:
        raise ValueError("generation must be non-negative")
    s0, s1 = "A", "AB"
    if generation == 0:
        return s0
    if generation == 1:
        return s1
    for _ in range(2, generation + 1):
        s0, s1 = s1, s1 + s0
    return s1


def lead_self_energy(energy: float, hopping: float = 1.0, coupling: float = 1.0, eta: float = 1e-9) -> complex:
    """Retarded surface self-energy for a semi-infinite 1D nearest-neighbor lead."""
    z = complex(energy, eta)
    root = np.lib.scimath.sqrt(z**2 - 4.0 * hopping**2)
    # Choose the retarded branch with Im(g_surface) <= 0.
    g1 = (z - root) / (2.0 * hopping**2)
    g2 = (z + root) / (2.0 * hopping**2)
    g_surface = g1 if g1.imag <= 0 else g2
    return coupling**2 * g_surface


def fibonacci_transmission(
    energies: np.ndarray,
    generation: int = 8,
    onsite_a: float = 0.0,
    onsite_b: float = 1.5,
    hopping: float = 1.0,
    lead_hopping: float = 1.0,
    contact_coupling: float = 1.0,
) -> np.ndarray:
    """Landauer transmission T(E)=Gamma_L Gamma_R |G^r_{1N}|^2.

    The semi-infinite lead self-energies specify the open boundary conditions.
    This makes the physical data generator boundary-aware even though the
    regression kernel itself is not claimed to encode those boundary operators.
    """
    word = fibonacci_word(generation)
    onsite = np.array([onsite_a if symbol == "A" else onsite_b for symbol in word], dtype=float)
    n = len(onsite)
    H = np.diag(onsite)
    off = -hopping * np.ones(n - 1)
    H += np.diag(off, 1) + np.diag(off, -1)
    eye = np.eye(n, dtype=complex)
    values: list[float] = []
    for energy in np.asarray(energies, dtype=float):
        sigma_l = lead_self_energy(energy, lead_hopping, contact_coupling)
        sigma_r = lead_self_energy(energy, lead_hopping, contact_coupling)
        Sigma = np.zeros((n, n), dtype=complex)
        Sigma[0, 0] = sigma_l
        Sigma[-1, -1] = sigma_r
        G = np.linalg.inv((energy + 1j * 1e-9) * eye - H - Sigma)
        gamma_l = max(0.0, float(-2.0 * sigma_l.imag))
        gamma_r = max(0.0, float(-2.0 * sigma_r.imag))
        T = gamma_l * gamma_r * abs(G[0, -1]) ** 2
        values.append(float(np.clip(T.real, 0.0, 1.0 + 1e-8)))
    return np.asarray(values)


def generate_dataset(n_samples: int = 400, generation: int = 8):
    energies = np.linspace(-1.95, 1.95, n_samples)
    y = fibonacci_transmission(energies, generation=generation)
    return energies.reshape(-1, 1), y


def scalability_benchmark(best_degree: int, quick: bool) -> pd.DataFrame:
    sizes = [80, 120, 200, 300] if quick else [200, 400, 800, 1600, 3200]
    rows = []
    for n in sizes:
        X = np.linspace(-1.95, 1.95, n).reshape(-1, 1)
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
    axes[0].legend(); axes[1].legend(); fig.suptitle("Fibonacci-transmission kernel scalability")
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def plot_spectrum_and_predictions(X, y, custom_model, rbf_model, path: Path):
    order = np.argsort(X[:, 0])
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(X[order, 0], y[order], label="Boundary-conditioned transmission")
    ax.plot(X[order, 0], custom_model.predict(X[order]), label="Jackson-Chebyshev SVR")
    ax.plot(X[order, 0], rbf_model.predict(X[order]), linestyle="--", label="RBF SVR")
    ax.set_xlabel("Energy / hopping")
    ax.set_ylabel("Transmission")
    ax.legend()
    ax.set_title("Fibonacci-chain transmission reconstruction")
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="quasicrystal_round2_results")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    out = ensure_output_dir(args.output_dir)
    X, y = generate_dataset(120 if args.quick else 400, generation=6 if args.quick else 8)

    fold_df, best_history = run_nested_cv(X, y, args.quick)
    summary = aggregate_fold_metrics(fold_df)
    fold_df.to_csv(out / "quasicrystal_round2_cv_folds.csv", index=False)
    summary.to_csv(out / "quasicrystal_round2_cv_summary.csv", index=False)
    save_json(best_history, out / "quasicrystal_round2_best_params.json")
    plot_metric_bars(summary, out / "quasicrystal_round2_benchmark.png", "Fibonacci-transmission repeated nested-CV benchmark")

    inner = 3 if args.quick else 5
    custom_params, _ = tune_custom(X, y, args.quick, inner)
    spaces = model_spaces(args.quick)
    tuned = {"Jackson-Chebyshev SVR": JacksonSVR(**custom_params)}
    for name in ["RBF SVR", "Random Forest", "MLP"]:
        est, grid = spaces[name]
        search = GridSearchCV(est, grid, scoring="neg_mean_squared_error", cv=inner, n_jobs=1 if args.quick else -1, refit=True).fit(X, y)
        tuned[name] = search.best_estimator_

    raw_learning, learning_summary = learning_curves(X, y, tuned, args.quick)
    raw_learning.to_csv(out / "quasicrystal_round2_learning_curve_folds.csv", index=False)
    pd.concat([df.assign(Model=name) for name, df in learning_summary.items()], ignore_index=True).to_csv(out / "quasicrystal_round2_learning_curve_summary.csv", index=False)
    plot_learning_curves(learning_summary, out / "quasicrystal_round2_learning_curves.png", "Fibonacci-transmission sample-efficiency comparison")

    rank_df = rank_tradeoff(X, y, custom_params, args.quick)
    rank_df.to_csv(out / "quasicrystal_round2_nystrom_rank_tradeoff.csv", index=False)
    plot_rank_tradeoff(rank_df, out / "quasicrystal_round2_nystrom_rank_tradeoff.png")

    scaling = scalability_benchmark(custom_params["degree"], args.quick)
    scaling.to_csv(out / "quasicrystal_round2_scalability.csv", index=False)
    plot_scalability(scaling, out / "quasicrystal_round2_scalability.png")

    # Fit two representative models for a dense reconstruction figure.
    custom = JacksonSVR(**custom_params).fit(X, y)
    rbf_est, rbf_grid = spaces["RBF SVR"]
    rbf = GridSearchCV(rbf_est, rbf_grid, scoring="neg_mean_squared_error", cv=inner, n_jobs=1 if args.quick else -1, refit=True).fit(X, y).best_estimator_
    plot_spectrum_and_predictions(X, y, custom, rbf, out / "quasicrystal_round2_reconstruction.png")

    fmap = custom.feature_map_
    K = fmap.gram(X, add_jitter=True)
    diagnostics = {
        "physical_boundary_condition": "Open system coupled to left/right semi-infinite leads through retarded self-energies",
        "custom_kernel_boundary_claim": "The regression kernel is not claimed to compensate for or reproduce the lead boundary operator",
        "custom_parameters": custom_params,
        "minimum_eigenvalue": float(np.linalg.eigvalsh(K)[0]),
        "centered_kernel_target_alignment": centered_kernel_target_alignment(K, y),
        "feature_dimension": int(custom_params["degree"] + 1),
    }
    save_json(diagnostics, out / "quasicrystal_round2_kernel_diagnostics.json")
    print(f"Completed. Results saved in: {out.resolve()}")


if __name__ == "__main__":
    main()
