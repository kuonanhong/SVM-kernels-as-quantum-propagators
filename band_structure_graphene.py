#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reviewer-focused second-round graphene benchmark.

This script is the data-producing program that must be run *before*
``generate_round2_latex_tables_fixed.py``.  It does the following:

1. loads a cached ``graphene_band_dataset.csv`` when available;
2. otherwise downloads the Materials Project band structure for mp-48;
3. constructs the local conduction-side dispersion data used in the manuscript;
4. runs repeated nested cross-validation for SVR, random-forest, MLP, and dummy
   baselines;
5. writes ``graphene_round2_cv_summary.csv`` and all split-level diagnostics;
6. creates the two manuscript figures
   ``graphene_round2_benchmark.png`` and
   ``graphene_round2_learning_curves.png``.

Direct final run
----------------
    python band_structure_graphene_revised_round2_fixed.py

Fast execution check only (synthetic data; NOT manuscript evidence)
------------------------------------------------------------------
    python band_structure_graphene_revised_round2_fixed.py \
        --smoke-test --quick

The Materials Project API key is read from ``--api-key``, the ``MP_API_KEY``
environment variable, or an optional private local text file named
``materials_project_api_key.txt`` beside this script.  The key is never written
to result files.

Compatibility note
------------------
Materials Project repaired the 2026 band-structure retrieval incompatibility in
``mp-api==0.46.4`` together with ``emmet-core==0.87.1``.  Pymatgen 2026.3+
separates the electronic-structure classes into the ``pymatgen-core``
distribution, which still imports through the normal ``pymatgen`` namespace.
This program therefore checks both distribution metadata and the actual
``pymatgen.electronic_structure`` import before contacting Materials Project.
It uses the supported public ``MPRester.get_bandstructure_by_material_id``
route and deliberately avoids the obsolete task-ID/S3 fallback.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import matplotlib

# Safe in PyCharm, Terminal, SSH, and headless execution.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    from scipy.stats import t as student_t
except ImportError:  # pragma: no cover
    student_t = None

# -----------------------------------------------------------------------------
# LOCAL DIRECT-EXECUTION CONFIGURATION
# -----------------------------------------------------------------------------
# Never commit a private Materials Project key to a manuscript or repository.
# Direct execution can read it from materials_project_api_key.txt beside the
# script; --api-key and MP_API_KEY take precedence.
API_KEY = ""
API_KEY_FILE = Path(__file__).resolve().parent / "materials_project_api_key.txt"

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_CSV = SCRIPT_DIR / "graphene_band_dataset.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "graphene_round2_results"
DEFAULT_MANUSCRIPT_FIGURE_DIR = SCRIPT_DIR / "Figures2"

MATERIAL_ID = "mp-48"
RANDOM_STATE = 42
MINIMUM_MODEL_SAMPLES = 30

REQUIRED_MP_API = "0.46.4"
REQUIRED_EMMET_CORE = "0.87.1"
REQUIRED_PYMATGEN_CORE = "2026.4.16"
VERIFIED_PYMATGEN_CORE = "2026.5.18"
RECOMMENDED_PYMATGEN = "2026.5.4"


# -----------------------------------------------------------------------------
# GENERAL UTILITIES
# -----------------------------------------------------------------------------
def ensure_output_dir(path: str | Path) -> Path:
    out = Path(path).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(data: Any, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, default=str)


def summary95(values: pd.Series | np.ndarray | list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    n = int(array.size)
    if n == 0:
        return {"mean": math.nan, "std": math.nan, "ci95": math.nan, "n": 0}
    mean = float(np.mean(array))
    if n == 1:
        return {"mean": mean, "std": 0.0, "ci95": 0.0, "n": 1}
    std = float(np.std(array, ddof=1))
    sem = std / math.sqrt(n)
    critical = (
        float(student_t.ppf(0.975, df=n - 1))
        if student_t is not None
        else 1.96
    )
    return {"mean": mean, "std": std, "ci95": critical * sem, "n": n}


def stringify_parameter_frequency(values: list[dict[str, Any]]) -> dict[str, int]:
    encoded = [json.dumps(value, sort_keys=True, default=str) for value in values]
    return dict(Counter(encoded))


def _resolve_user_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    # Resolve relative paths beside this script rather than against an arbitrary
    # PyCharm working directory.
    return (SCRIPT_DIR / expanded).resolve()


# -----------------------------------------------------------------------------
# MATERIALS PROJECT DATA ACQUISITION
# -----------------------------------------------------------------------------
def _package_version(distribution: str) -> str:
    """Return an installed distribution version without importing internals."""
    try:
        from importlib.metadata import version

        return version(distribution)
    except Exception:
        return "unknown"


def _print_materials_stack_versions() -> None:
    print(
        "Materials stack: "
        f"mp-api={_package_version('mp-api')}, "
        f"emmet-core={_package_version('emmet-core')}, "
        f"pymatgen={_package_version('pymatgen')}, "
        f"pymatgen-core={_package_version('pymatgen-core')}, "
        f"pydantic={_package_version('pydantic')}"
    )


def _module_search_locations(module_name: str) -> list[str]:
    """Return import locations to diagnose broken namespace installations."""
    try:
        from importlib.util import find_spec

        spec = find_spec(module_name)
    except Exception:
        return []
    if spec is None:
        return []
    locations = list(spec.submodule_search_locations or [])
    if spec.origin and spec.origin not in {"built-in", "namespace"}:
        locations.append(str(spec.origin))
    return locations


def _validate_pymatgen_runtime() -> None:
    """Verify that distribution metadata corresponds to importable modules."""
    pymatgen_version = _package_version("pymatgen")
    core_version = _package_version("pymatgen-core")
    locations = _module_search_locations("pymatgen")
    repair_command = (
        "python -m pip install --no-cache-dir --upgrade --force-reinstall "
        f"'pymatgen=={RECOMMENDED_PYMATGEN}' "
        f"'pymatgen-core=={VERIFIED_PYMATGEN_CORE}'"
    )

    if core_version == "unknown":
        raise RuntimeError(
            "The 'pymatgen' metadata package is installed, but the required "
            "'pymatgen-core' distribution is missing. Since pymatgen 2026.3, "
            "electronic-structure classes are supplied by pymatgen-core under "
            "the same pymatgen namespace. Repair this exact environment with:\n\n"
            f"  {repair_command}\n\n"
            "Then test:\n"
            "  python -c \"from pymatgen.electronic_structure.core import Spin; "
            "print(Spin.up)\"\n"
            f"Detected pymatgen locations: {locations or ['none']}"
        )

    if _parse_version(core_version) < _parse_version(REQUIRED_PYMATGEN_CORE):
        raise RuntimeError(
            f"pymatgen-core={core_version} is too old; need >= "
            f"{REQUIRED_PYMATGEN_CORE}. Install the verified pair:\n"
            f"  {repair_command}"
        )

    try:
        from pymatgen.electronic_structure.bandstructure import BandStructure  # noqa: F401
        from pymatgen.electronic_structure.core import Spin  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Package metadata reports pymatgen="
            f"{pymatgen_version} and pymatgen-core={core_version}, but Python "
            "cannot import pymatgen.electronic_structure. This normally means "
            "a partial/overlapping conda-pip installation or a local pymatgen "
            "directory is shadowing site-packages. Repair with:\n\n"
            f"  {repair_command}\n\n"
            "and ensure the script directory contains no file/folder named "
            "pymatgen.py or pymatgen/.\n"
            f"Detected pymatgen locations: {locations or ['none']}"
        ) from exc


def _read_local_api_key_file() -> str:
    """Read a private one-line API-key file beside the script, when present."""
    try:
        if API_KEY_FILE.exists():
            return API_KEY_FILE.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"Could not read API key file {API_KEY_FILE}: {exc}") from exc
    return ""


def _parse_version(value: str):
    """Parse a distribution version with a clear dependency error."""
    try:
        from packaging.version import Version
    except ImportError as exc:
        raise RuntimeError(
            "The 'packaging' package is required for version checks. Install it "
            "with: python -m pip install -U packaging"
        ) from exc
    try:
        return Version(value)
    except Exception as exc:
        raise RuntimeError(f"Could not parse installed version {value!r}.") from exc


def _require_supported_materials_stack() -> None:
    """Reject known-broken or incomplete Materials Project client stacks."""
    mp_api_version = _package_version("mp-api")
    emmet_version = _package_version("emmet-core")
    pymatgen_version = _package_version("pymatgen")
    pymatgen_core_version = _package_version("pymatgen-core")

    missing_or_old: list[str] = []
    if mp_api_version == "unknown" or _parse_version(mp_api_version) < _parse_version(REQUIRED_MP_API):
        missing_or_old.append(
            f"mp-api={mp_api_version} (need >= {REQUIRED_MP_API})"
        )
    if emmet_version == "unknown" or _parse_version(emmet_version) < _parse_version(REQUIRED_EMMET_CORE):
        missing_or_old.append(
            f"emmet-core={emmet_version} (need >= {REQUIRED_EMMET_CORE})"
        )
    if (
        pymatgen_core_version == "unknown"
        or _parse_version(pymatgen_core_version) < _parse_version(REQUIRED_PYMATGEN_CORE)
    ):
        missing_or_old.append(
            f"pymatgen-core={pymatgen_core_version} "
            f"(need >= {REQUIRED_PYMATGEN_CORE})"
        )

    if missing_or_old:
        details = ", ".join(missing_or_old)
        raise RuntimeError(
            "Unsupported or incomplete Materials Project client stack: "
            + details
            + "\n\nRepair this conda environment with:\n\n"
            "  python -m pip install --no-cache-dir --upgrade --force-reinstall "
            "'mp-api==0.46.4' 'emmet-core==0.87.1' "
            "'pymatgen==2026.5.4' 'pymatgen-core==2026.5.18'\n\n"
            "Then verify with:\n"
            "  python -m pip check\n"
            "  python -c \"from pymatgen.electronic_structure.core import Spin; "
            "print(Spin.up)\""
        )

    _validate_pymatgen_runtime()

    if (
        pymatgen_version != "unknown"
        and _parse_version(pymatgen_version) < _parse_version(RECOMMENDED_PYMATGEN)
    ):
        warnings.warn(
            f"pymatgen={pymatgen_version} is older than the verified working "
            f"version {RECOMMENDED_PYMATGEN}. The API call may still work, but "
            "upgrading the complete stack is recommended.",
            RuntimeWarning,
            stacklevel=2,
        )


def _open_mprester(api_key: str):
    """Instantiate the supported Materials Project client."""
    _require_supported_materials_stack()
    try:
        from mp_api.client import MPRester
    except ImportError as exc:
        raise RuntimeError(
            "mp-api is required. Install the verified stack with:\n"
            "python -m pip install --no-cache-dir --upgrade --force-reinstall "
            "'mp-api==0.46.4' 'emmet-core==0.87.1' 'pymatgen==2026.5.4' 'pymatgen-core==2026.5.18'"
        ) from exc

    try:
        return MPRester(api_key, mute_progress_bars=True)
    except TypeError:
        return MPRester(api_key)


def _get_bandstructure(mpr: Any, material_id: str):
    """Retrieve a line-mode band structure using the repaired public API route."""
    try:
        # Official mp-api examples use this location.
        from emmet.core.electronic_structure import BSPathType
    except ImportError:
        try:
            # Compatibility with releases that still expose the older alias.
            from emmet.core.band_theory import BSPathType
        except ImportError as exc:
            raise RuntimeError(
                "Could not import BSPathType from emmet-core. "
                "Install emmet-core==0.87.1."
            ) from exc

    errors: list[str] = []
    path_types = [
        BSPathType.setyawan_curtarolo,
        BSPathType.hinuma,
        BSPathType.latimer_munro,
    ]

    top_level = getattr(mpr, "get_bandstructure_by_material_id", None)
    if callable(top_level):
        for path_type in path_types:
            try:
                band_structure = top_level(
                    material_id,
                    path_type=path_type,
                    line_mode=True,
                )
                if band_structure is not None:
                    print(
                        "Band-structure retrieval: "
                        f"MPRester.get_bandstructure_by_material_id "
                        f"(path={path_type.value})"
                    )
                    return band_structure
            except TypeError:
                # Compatibility for a release whose helper does not expose
                # line_mode as a keyword but does support path_type.
                try:
                    band_structure = top_level(
                        material_id,
                        path_type=path_type,
                    )
                    if band_structure is not None:
                        print(
                            "Band-structure retrieval: "
                            f"MPRester.get_bandstructure_by_material_id "
                            f"(path={path_type.value})"
                        )
                        return band_structure
                except Exception as exc:
                    errors.append(
                        f"{path_type.value}: {type(exc).__name__}: {exc}"
                    )
            except Exception as exc:
                errors.append(f"{path_type.value}: {type(exc).__name__}: {exc}")

    # Route-level fallback remains within the supported material-ID API. It is
    # not the stale get_bandstructure_from_task_id/S3 path.
    route = getattr(
        getattr(mpr, "materials", None),
        "electronic_structure_bandstructure",
        None,
    )
    route_method = getattr(route, "get_bandstructure_from_material_id", None)
    if callable(route_method):
        for path_type in path_types:
            try:
                band_structure = route_method(
                    material_id,
                    path_type=path_type,
                )
                if band_structure is not None:
                    print(
                        "Band-structure retrieval: route-level material-ID "
                        f"helper (path={path_type.value})"
                    )
                    return band_structure
            except TypeError:
                try:
                    band_structure = route_method(material_id)
                    if band_structure is not None:
                        print(
                            "Band-structure retrieval: route-level material-ID "
                            "helper (default path)"
                        )
                        return band_structure
                except Exception as exc:
                    errors.append(
                        "route default: " f"{type(exc).__name__}: {exc}"
                    )
                break
            except Exception as exc:
                errors.append(
                    f"route {path_type.value}: {type(exc).__name__}: {exc}"
                )

    detail = " | ".join(errors) if errors else "No supported retrieval method exists."
    raise RuntimeError(
        f"Materials Project returned no usable line-mode band structure for "
        f"{material_id}. Installed stack: mp-api={_package_version('mp-api')}, "
        f"emmet-core={_package_version('emmet-core')}, "
        f"pymatgen={_package_version('pymatgen')}. Details: {detail}"
    )


def fetch_graphene_dataset(
    api_key: str,
    output_csv: Path,
    energy_window_ev: float,
) -> pd.DataFrame:
    """Download mp-48 and construct the local conduction-side data set.

    The sampled point closest to the Fermi level is used as the operational
    crossing point.  The input is the in-plane Cartesian distance q from that
    point, and the target is the positive energy relative to the Fermi level.
    This is a finite-path local regression data set, not a full Brillouin-zone
    Dirac-cone or operator-learning reconstruction.
    """
    _validate_pymatgen_runtime()

    if not api_key or api_key.strip().upper().startswith("YOUR_"):
        raise RuntimeError("A valid Materials Project API key is required.")

    print(f"Fetching graphene band structure: Materials Project {MATERIAL_ID}")
    mpr = _open_mprester(api_key)
    try:
        with mpr:
            band_structure = _get_bandstructure(mpr, MATERIAL_ID)
    except AttributeError:
        # Some older MPRester objects do not implement the context protocol.
        band_structure = _get_bandstructure(mpr, MATERIAL_ID)

    band_mapping = band_structure.bands
    if not band_mapping:
        raise RuntimeError("The downloaded band structure contains no band channels.")

    # Prefer the spin-up channel when present; otherwise use the first channel.
    spin_key = next(
        (
            key
            for key in band_mapping
            if getattr(key, "value", None) == 1
            or str(key).lower() in {"1", "up", "spin.up"}
        ),
        next(iter(band_mapping)),
    )

    bands = np.asarray(band_mapping[spin_key], dtype=float)
    kpoints = np.asarray(
        [np.asarray(kpoint.cart_coords, dtype=float) for kpoint in band_structure.kpoints],
        dtype=float,
    )
    efermi = float(band_structure.efermi)

    if bands.ndim != 2 or kpoints.ndim != 2 or bands.shape[1] != len(kpoints):
        raise RuntimeError(
            "Unexpected band-structure dimensions: "
            f"bands={bands.shape}, kpoints={kpoints.shape}"
        )

    shifted = bands - efermi
    finite_abs = np.where(np.isfinite(shifted), np.abs(shifted), np.inf)
    flat_index = int(np.argmin(finite_abs))
    _, crossing_k_index = np.unravel_index(flat_index, shifted.shape)
    crossing_k = kpoints[crossing_k_index]
    crossing_residual = float(finite_abs.ravel()[flat_index])

    rows: list[dict[str, float | int | str]] = []
    for band_index, band_energy in enumerate(shifted):
        finite_band = band_energy[np.isfinite(band_energy)]
        if finite_band.size == 0 or float(np.min(np.abs(finite_band))) > energy_window_ev:
            continue

        for path_index, energy in enumerate(band_energy):
            if not np.isfinite(energy) or not (0.0 < energy < energy_window_ev):
                continue
            q_inv_a = float(
                np.linalg.norm(kpoints[path_index, :2] - crossing_k[:2])
            )
            if np.isfinite(q_inv_a):
                rows.append(
                    {
                        "q_invA": q_inv_a,
                        "Energy_eV": float(energy),
                        "BandIndex": int(band_index),
                        "PathIndex": int(path_index),
                        "Source": f"Materials Project {MATERIAL_ID}",
                    }
                )

    data = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["q_invA", "Energy_eV"])
        .sort_values(["q_invA", "Energy_eV"])
        .reset_index(drop=True)
    )

    if len(data) < MINIMUM_MODEL_SAMPLES:
        raise RuntimeError(
            f"Only {len(data)} valid local conduction-side samples were extracted "
            f"with --energy-window {energy_window_ev:g} eV. Increase the window "
            "or inspect the current mp-48 band-structure record."
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_csv, index=False)
    print(
        f"Saved {len(data)} samples to: {output_csv}\n"
        f"Operational crossing residual |E-E_F| = {crossing_residual:.6g} eV"
    )
    return data


def synthetic_dirac_dataset(output_csv: Path, n_samples: int = 240) -> pd.DataFrame:
    """Deterministic execution-check data; never use as paper evidence."""
    rng = np.random.default_rng(RANDOM_STATE)
    q_inv_a = np.sort(rng.uniform(0.0, 0.22, n_samples))
    energy_ev = (
        5.8 * q_inv_a
        + 0.8 * q_inv_a**2
        + rng.normal(0.0, 0.008 + 0.015 * q_inv_a, n_samples)
    )
    frame = pd.DataFrame(
        {
            "q_invA": q_inv_a,
            "Energy_eV": energy_ev,
            "BandIndex": 0,
            "PathIndex": np.arange(n_samples),
            "Source": "explicit synthetic smoke test",
        }
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    return frame


def canonicalize_graphene_csv(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    aliases = {
        "q": "q_invA",
        "Q": "q_invA",
        "q_distance": "q_invA",
        "energy": "Energy_eV",
        "Energy": "Energy_eV",
        "E": "Energy_eV",
    }
    frame = frame.rename(
        columns={source: target for source, target in aliases.items() if source in frame.columns}
    )
    required = {"q_invA", "Energy_eV"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    cleaned = (
        frame.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["q_invA", "Energy_eV"])
        .drop_duplicates(subset=["q_invA", "Energy_eV"])
        .sort_values(["q_invA", "Energy_eV"])
        .reset_index(drop=True)
    )
    if "Source" not in cleaned.columns:
        cleaned["Source"] = f"cached CSV: {path.name}"
    if len(cleaned) < MINIMUM_MODEL_SAMPLES:
        raise ValueError(
            f"{path} contains only {len(cleaned)} usable samples; at least "
            f"{MINIMUM_MODEL_SAMPLES} are required."
        )
    return cleaned


def load_dataset(
    csv_path: Path,
    api_key: str,
    energy_window_ev: float,
    refresh_data: bool,
    smoke_test: bool,
) -> tuple[pd.DataFrame, str]:
    if smoke_test:
        smoke_csv = csv_path.with_name(csv_path.stem + "_smoke_test.csv")
        frame = synthetic_dirac_dataset(smoke_csv)
        return canonicalize_graphene_csv(frame, smoke_csv), "synthetic smoke test"

    if csv_path.exists() and not refresh_data:
        print(f"Loading cached graphene data: {csv_path}")
        return canonicalize_graphene_csv(pd.read_csv(csv_path), csv_path), "cached CSV"

    frame = fetch_graphene_dataset(
        api_key=api_key,
        output_csv=csv_path,
        energy_window_ev=energy_window_ev,
    )
    return canonicalize_graphene_csv(frame, csv_path), f"Materials Project {MATERIAL_ID}"


# -----------------------------------------------------------------------------
# MODELS AND REPEATED NESTED CROSS-VALIDATION
# -----------------------------------------------------------------------------
def model_spaces(quick: bool) -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    if quick:
        return {
            "Linear SVR": (
                Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="linear"))]),
                {"model__C": [10.0], "model__epsilon": [0.005]},
            ),
            "RBF SVR": (
                Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="rbf"))]),
                {
                    "model__C": [10.0],
                    "model__gamma": ["scale"],
                    "model__epsilon": [0.005],
                },
            ),
            "Polynomial SVR": (
                Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="poly"))]),
                {
                    "model__degree": [2, 3],
                    "model__C": [10.0],
                    "model__gamma": ["scale"],
                    "model__coef0": [1.0],
                    "model__epsilon": [0.005],
                },
            ),
            "Sigmoid SVR": (
                Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="sigmoid"))]),
                {
                    "model__C": [1.0],
                    "model__gamma": [0.1],
                    "model__coef0": [0.0],
                    "model__epsilon": [0.005],
                },
            ),
            "Random Forest": (
                RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
                {
                    "n_estimators": [100],
                    "max_depth": [8],
                    "min_samples_leaf": [1],
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
                                max_iter=600,
                                early_stopping=True,
                                n_iter_no_change=40,
                            ),
                        ),
                    ]
                ),
                {
                    "model__hidden_layer_sizes": [(16, 16)],
                    "model__activation": ["tanh"],
                    "model__alpha": [1e-4],
                    "model__learning_rate_init": [1e-3],
                },
            ),
            "Dummy Mean": (DummyRegressor(), {"strategy": ["mean"]}),
        }

    # Deliberately bounded grids: enough to perform a real nested selection while
    # avoiding an unnecessarily large tens-of-thousands-combination search.
    return {
        "Linear SVR": (
            Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="linear"))]),
            {
                "model__C": [0.1, 1.0, 10.0, 100.0, 300.0],
                "model__epsilon": [0.001, 0.005, 0.01, 0.05],
            },
        ),
        "RBF SVR": (
            Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="rbf"))]),
            {
                "model__C": [1.0, 10.0, 100.0, 300.0],
                "model__gamma": ["scale", 0.1, 1.0, 10.0],
                "model__epsilon": [0.001, 0.005, 0.01, 0.05],
            },
        ),
        "Polynomial SVR": (
            Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="poly"))]),
            {
                "model__degree": [2, 3, 4],
                "model__C": [1.0, 10.0, 100.0],
                "model__gamma": ["scale", 0.1, 1.0],
                "model__coef0": [0.0, 1.0],
                "model__epsilon": [0.001, 0.01],
            },
        ),
        "Sigmoid SVR": (
            Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="sigmoid"))]),
            {
                "model__C": [0.1, 1.0, 10.0, 100.0],
                "model__gamma": ["scale", 0.01, 0.1, 1.0],
                "model__coef0": [-1.0, 0.0, 1.0],
                "model__epsilon": [0.001, 0.01],
            },
        ),
        "Random Forest": (
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "n_estimators": [200, 400],
                "max_depth": [None, 4, 8, 12],
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
                            max_iter=2500,
                            early_stopping=True,
                            n_iter_no_change=60,
                        ),
                    ),
                ]
            ),
            {
                "model__hidden_layer_sizes": [(32, 32), (64, 64)],
                "model__activation": ["tanh", "relu"],
                "model__alpha": [1e-5, 1e-4, 1e-3],
                "model__learning_rate_init": [1e-3],
            },
        ),
        "Dummy Mean": (DummyRegressor(), {"strategy": ["mean"]}),
    }


def _support_vector_fraction(estimator: Any, n_train: int) -> float:
    candidate = estimator
    if hasattr(estimator, "named_steps") and "model" in estimator.named_steps:
        candidate = estimator.named_steps["model"]
    if hasattr(candidate, "support_"):
        return float(len(candidate.support_) / max(n_train, 1))
    return math.nan


def run_nested_cv(
    X: np.ndarray,
    y: np.ndarray,
    quick: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    outer = RepeatedKFold(
        n_splits=3 if quick else 5,
        n_repeats=1 if quick else 3,
        random_state=RANDOM_STATE,
    )
    inner_folds = 3 if quick else 5
    n_jobs = 1 if quick else -1

    rows: list[dict[str, Any]] = []
    parameter_history: dict[str, Any] = {}

    for model_name, (estimator, grid) in model_spaces(quick).items():
        print(f"Nested CV: {model_name}")
        best_params_for_model: list[dict[str, Any]] = []

        for split_id, (train_index, test_index) in enumerate(outer.split(X, y), 1):
            search = GridSearchCV(
                estimator=estimator,
                param_grid=grid,
                scoring="neg_mean_squared_error",
                cv=inner_folds,
                n_jobs=n_jobs,
                refit=True,
                error_score="raise",
            )

            fit_start = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                search.fit(X[train_index], y[train_index])
            fit_time = time.perf_counter() - fit_start

            predict_start = time.perf_counter()
            prediction = search.predict(X[test_index])
            predict_time = time.perf_counter() - predict_start

            rows.append(
                {
                    "Model": model_name,
                    "Split": split_id,
                    "TrainSamples": len(train_index),
                    "TestSamples": len(test_index),
                    "MSE": mean_squared_error(y[test_index], prediction),
                    "R2": r2_score(y[test_index], prediction),
                    "FitTimeSec": fit_time,
                    "PredictTimeSec": predict_time,
                    "SupportVectorFraction": _support_vector_fraction(
                        search.best_estimator_, len(train_index)
                    ),
                    "BestParams": json.dumps(
                        search.best_params_, sort_keys=True, default=str
                    ),
                }
            )
            best_params_for_model.append(search.best_params_)

        parameter_history[model_name] = {
            "frequency": stringify_parameter_frequency(best_params_for_model),
            "outer_splits": best_params_for_model,
        }

    folds = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for model_name, group in folds.groupby("Model", sort=False):
        mse = summary95(group["MSE"])
        r2 = summary95(group["R2"])
        fit = summary95(group["FitTimeSec"])
        predict = summary95(group["PredictTimeSec"])
        support = summary95(group["SupportVectorFraction"])
        summary_rows.append(
            {
                "Model": model_name,
                "CV_MSE_Mean": mse["mean"],
                "CV_MSE_Std": mse["std"],
                "CV_MSE_CI95": mse["ci95"],
                "CV_R2_Mean": r2["mean"],
                "CV_R2_Std": r2["std"],
                "CV_R2_CI95": r2["ci95"],
                # Exact names expected by generate_round2_latex_tables_fixed.py
                "FitTime_Mean": fit["mean"],
                "FitTime_Std": fit["std"],
                "FitTime_CI95": fit["ci95"],
                "PredictTime_Mean": predict["mean"],
                "PredictTime_CI95": predict["ci95"],
                "SupportVectorFraction_Mean": support["mean"],
                "SupportVectorFraction_CI95": support["ci95"],
                "OuterSplits": int(mse["n"]),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("CV_MSE_Mean").reset_index(drop=True)
    return folds, summary, parameter_history


# -----------------------------------------------------------------------------
# LEARNING CURVES
# -----------------------------------------------------------------------------
def tune_learning_curve_models(
    X: np.ndarray,
    y: np.ndarray,
    quick: bool,
) -> dict[str, Any]:
    spaces = model_spaces(quick)
    names = ["Linear SVR", "RBF SVR", "Random Forest", "MLP"]
    inner_folds = 3 if quick else 5
    n_jobs = 1 if quick else -1
    tuned: dict[str, Any] = {}

    for name in names:
        print(f"Learning-curve tuning: {name}")
        estimator, grid = spaces[name]
        search = GridSearchCV(
            estimator=estimator,
            param_grid=grid,
            scoring="neg_mean_squared_error",
            cv=inner_folds,
            n_jobs=n_jobs,
            refit=True,
            error_score="raise",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            search.fit(X, y)
        tuned[name] = search.best_estimator_
    return tuned


def evaluate_learning_curve(
    model_factory: Callable[[], Any],
    X: np.ndarray,
    y: np.ndarray,
    train_sizes: np.ndarray,
    n_repeats: int,
    test_fraction: float = 0.25,
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    n_samples = len(X)
    n_test = max(2, int(round(test_fraction * n_samples)))
    rows: list[dict[str, Any]] = []

    for repeat in range(1, n_repeats + 1):
        permutation = rng.permutation(n_samples)
        test_index = permutation[:n_test]
        training_pool = permutation[n_test:]

        for requested_size in train_sizes:
            effective_size = min(int(requested_size), len(training_pool))
            if effective_size < 5:
                continue
            train_index = training_pool[:effective_size]
            estimator = model_factory()

            fit_start = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                estimator.fit(X[train_index], y[train_index])
            fit_time = time.perf_counter() - fit_start

            predict_start = time.perf_counter()
            prediction = estimator.predict(X[test_index])
            predict_time = time.perf_counter() - predict_start

            rows.append(
                {
                    "Repeat": repeat,
                    "TrainSize": effective_size,
                    "MSE": mean_squared_error(y[test_index], prediction),
                    "R2": r2_score(y[test_index], prediction),
                    "FitTimeSec": fit_time,
                    "PredictTimeSec": predict_time,
                    "SupportVectorFraction": _support_vector_fraction(
                        estimator, effective_size
                    ),
                }
            )

    return pd.DataFrame(rows)


def summarize_learning_curve(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for train_size, group in raw.groupby("TrainSize"):
        mse = summary95(group["MSE"])
        r2 = summary95(group["R2"])
        fit = summary95(group["FitTimeSec"])
        rows.append(
            {
                "TrainSize": int(train_size),
                "MSE_Mean": mse["mean"],
                "MSE_CI95": mse["ci95"],
                "R2_Mean": r2["mean"],
                "R2_CI95": r2["ci95"],
                "FitTime_MeanSec": fit["mean"],
                "FitTime_CI95": fit["ci95"],
            }
        )
    return pd.DataFrame(rows).sort_values("TrainSize").reset_index(drop=True)


def run_learning_curves(
    X: np.ndarray,
    y: np.ndarray,
    quick: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    tuned = tune_learning_curve_models(X, y, quick)

    min_size = max(20, int(round(0.10 * len(X))))
    max_size = max(min_size, int(math.floor(0.75 * len(X))))
    train_sizes = np.unique(
        np.linspace(min_size, max_size, 4 if quick else 7).astype(int)
    )
    repeats = 3 if quick else 10

    raw_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    summaries_by_model: dict[str, pd.DataFrame] = {}

    for model_name, estimator in tuned.items():
        print(f"Learning curve: {model_name}")
        raw = evaluate_learning_curve(
            model_factory=lambda fitted=estimator: clone(fitted),
            X=X,
            y=y,
            train_sizes=train_sizes,
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
    plot_data = summary.sort_values("CV_MSE_Mean").reset_index(drop=True)
    positions = np.arange(len(plot_data))

    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].bar(
        positions,
        plot_data["CV_MSE_Mean"],
        yerr=plot_data["CV_MSE_CI95"],
        capsize=4,
    )
    axes[0].set_yscale("log")
    axes[0].set_title("Graphene local dispersion: repeated nested-CV MSE")
    axes[0].set_ylabel("MSE (mean ± 95% CI)")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(plot_data["Model"], rotation=30, ha="right")

    axes[1].bar(
        positions,
        plot_data["CV_R2_Mean"],
        yerr=plot_data["CV_R2_CI95"],
        capsize=4,
    )
    axes[1].axhline(0.0, linewidth=1)
    axes[1].set_title(r"Graphene local dispersion: repeated nested-CV $R^2$")
    axes[1].set_ylabel(r"$R^2$ (mean ± 95% CI)")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(plot_data["Model"], rotation=30, ha="right")

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
    axes[0].set_title("Graphene sample efficiency: test MSE")
    axes[0].set_xlabel("Training samples")
    axes[0].set_ylabel("MSE (mean ± 95% CI)")
    axes[0].legend()

    axes[1].axhline(0.0, linewidth=1)
    axes[1].set_title(r"Graphene sample efficiency: test $R^2$")
    axes[1].set_xlabel("Training samples")
    axes[1].set_ylabel(r"$R^2$ (mean ± 95% CI)")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reviewer-focused graphene local-dispersion benchmark"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV, JSON, and PNG outputs",
    )
    parser.add_argument(
        "--manuscript-figure-dir",
        type=Path,
        default=DEFAULT_MANUSCRIPT_FIGURE_DIR,
        help=(
            "Directory receiving copies of the two manuscript PNG files; "
            "default is Figures2 beside this script"
        ),
    )
    parser.add_argument(
        "--no-copy-manuscript-figures",
        action="store_true",
        help="Do not copy the two generated PNGs to --manuscript-figure-dir",
    )
    parser.add_argument(
        "--data-csv",
        type=Path,
        default=DEFAULT_DATA_CSV,
        help="Cached mp-48 local-dispersion CSV",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "Materials Project API key override. Precedence: --api-key, "
            "MP_API_KEY environment variable, materials_project_api_key.txt, "
            "then API_KEY in this file."
        ),
    )
    parser.add_argument(
        "--energy-window",
        type=float,
        default=1.0,
        help="Positive conduction-side energy window around E_F, in eV",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Ignore the cached CSV and fetch mp-48 again",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help=(
            "Fetch/validate graphene_band_dataset.csv and stop before the "
            "nested-CV benchmark"
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Reduced execution check; do not submit quick-mode results",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use deterministic synthetic data only to validate execution",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_arguments(argv)
    if args.energy_window <= 0:
        raise ValueError("--energy-window must be positive")

    output_dir = ensure_output_dir(_resolve_user_path(args.output_dir))
    data_csv = _resolve_user_path(args.data_csv)
    api_key = (
        args.api_key
        or os.getenv("MP_API_KEY")
        or _read_local_api_key_file()
        or API_KEY
    )
    _print_materials_stack_versions()

    frame, data_source = load_dataset(
        csv_path=data_csv,
        api_key=api_key,
        energy_window_ev=args.energy_window,
        refresh_data=args.refresh_data,
        smoke_test=args.smoke_test,
    )
    X = frame[["q_invA"]].to_numpy(dtype=float)
    y = frame["Energy_eV"].to_numpy(dtype=float)

    print(
        f"Benchmark dataset: n={len(frame)}, q-range="
        f"[{frame['q_invA'].min():.6g}, {frame['q_invA'].max():.6g}] 1/Å, "
        f"E-range=[{frame['Energy_eV'].min():.6g}, "
        f"{frame['Energy_eV'].max():.6g}] eV"
    )
    if args.quick or args.smoke_test:
        print("WARNING: quick/smoke outputs are for execution checking only.")

    if args.download_only:
        frame.to_csv(output_dir / "graphene_round2_selected_dataset.csv", index=False)
        print("\nDownload/validation completed; benchmark was skipped.")
        print(f"Cached data: {data_csv}")
        print(
            "Next run without --download-only to generate "
            "graphene_round2_cv_summary.csv and both figures."
        )
        return

    folds, summary, parameter_history = run_nested_cv(X, y, args.quick)
    folds.to_csv(output_dir / "graphene_round2_cv_folds.csv", index=False)
    summary.to_csv(output_dir / "graphene_round2_cv_summary.csv", index=False)
    save_json(parameter_history, output_dir / "graphene_round2_best_params.json")
    plot_benchmark(summary, output_dir / "graphene_round2_benchmark.png")

    learning_raw, learning_summary, learning_by_model = run_learning_curves(
        X, y, args.quick
    )
    learning_raw.to_csv(
        output_dir / "graphene_round2_learning_curve_folds.csv", index=False
    )
    learning_summary.to_csv(
        output_dir / "graphene_round2_learning_curve_summary.csv", index=False
    )
    plot_learning_curves(
        learning_by_model,
        output_dir / "graphene_round2_learning_curves.png",
    )

    if not args.no_copy_manuscript_figures:
        manuscript_figure_dir = ensure_output_dir(
            _resolve_user_path(args.manuscript_figure_dir)
        )
        for figure_name in [
            "graphene_round2_benchmark.png",
            "graphene_round2_learning_curves.png",
        ]:
            source = output_dir / figure_name
            destination = manuscript_figure_dir / figure_name
            shutil.copy2(source, destination)
            print(f"Copied manuscript figure: {source} -> {destination}")

    frame.to_csv(output_dir / "graphene_round2_selected_dataset.csv", index=False)
    save_json(
        {
            "data_source": data_source,
            "material_id": MATERIAL_ID,
            "n_samples": int(len(frame)),
            "input": "in-plane Cartesian q distance from the sampled near-Fermi crossing",
            "target": "conduction-side energy relative to the Fermi level",
            "energy_window_eV": float(args.energy_window),
            "scope": (
                "finite-path local scalar regression; not a full Brillouin-zone "
                "Dirac-cone or operator-learning reconstruction"
            ),
            "models": list(model_spaces(args.quick)),
            "outer_cv": "RepeatedKFold(5 splits x 3 repeats)" if not args.quick else "RepeatedKFold(3 splits x 1 repeat)",
            "inner_cv": 5 if not args.quick else 3,
            "random_state": RANDOM_STATE,
            "materials_stack": {
                "mp-api": _package_version("mp-api"),
                "emmet-core": _package_version("emmet-core"),
                "pymatgen": _package_version("pymatgen"),
                "pymatgen-core": _package_version("pymatgen-core"),
                "pydantic": _package_version("pydantic"),
            },
            "api_key_policy": (
                "Key read from --api-key, MP_API_KEY, local key file, or embedded local API_KEY; "
                "the key is not stored in output metadata"
            ),
            "quick_mode": bool(args.quick),
            "smoke_test": bool(args.smoke_test),
        },
        output_dir / "graphene_round2_dataset_notes.json",
    )

    print("\nCompleted. Generated:")
    for filename in [
        "graphene_round2_benchmark.png",
        "graphene_round2_learning_curves.png",
        "graphene_round2_cv_folds.csv",
        "graphene_round2_cv_summary.csv",
        "graphene_round2_learning_curve_folds.csv",
        "graphene_round2_learning_curve_summary.csv",
        "graphene_round2_best_params.json",
        "graphene_round2_selected_dataset.csv",
        "graphene_round2_dataset_notes.json",
    ]:
        print(f"  - {(output_dir / filename).resolve()}")


if __name__ == "__main__":
    main()
