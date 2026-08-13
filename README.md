Physics-Informed Support Vector Kernels via Green-Function
Analogies and Jackson–Chebyshev Spectral Design
========================================================================================

This repository contains experimental Python scripts used for the major-revision stage of the manuscript:

    Physics-Informed Support Vector Kernels via Green-Function Analogies and Jackson–Chebyshev Spectral Design

Authors:
    Nan-Hong Kuo
    Renata Wong

Repository purpose
------------------

This repository is intended to provide the revised and reproducible experimental scripts associated with the revised manuscript. The scripts extend the original experimental code by adding:

    1. repeated cross-validation statistics;
    2. error bars and 95% confidence intervals;
    3. explicit hyperparameter-tuning records;
    4. additional baseline models, including dummy-mean and random-forest regressors;
    5. computational-cost summaries;
    6. CSV/JSON outputs for benchmark reporting and supplementary tables.

Relationship to the original repository
---------------------------------------

The revised scripts in this repository are modified versions of the original experimental programs made available in the repository:

    https://github.com/renatawong/svm-kernels-as-quantum-propagators

Original repository:
    renatawong/svm-kernels-as-quantum-propagators

Revised repository:
    kuonanhong/SVM-kernels-as-quantum-propagators

The original repository contains the reference experimental notebooks for the paper. The present repository contains revised Python versions prepared for the major revision of the manuscript. The original scientific idea, paper context, and baseline experimental structure are retained. The revised scripts mainly add statistical validation, benchmarking, hyperparameter search, plotting updates, and computational-cost reporting.

Revised scripts included here
-----------------------------

The following revised Python scripts are included:

1. copper_conductivity.py

2. band_structure_graphene_revised.py

3. anharmonic_oscillator_energy_levels.py

4. photonic_crystals_revised.py 

5. quasicrystals_electronic_transmission.py 

6. svmprop_common.py  
------------------------------

## Materials Project API key

```bash
export MP_API_KEY="your-materials-project-api-key"

```

Then read it in Python using:

    import os
    API_KEY = os.environ.get("MP_API_KEY")
    
or an untracked `materials_project_api_key.txt` beside the graphene script.

## Environment check

```bash
python -m pip install -r requirements.txt
python -c "from pymatgen.electronic_structure.core import Spin; from mp_api.client import MPRester; print('Materials stack OK')"
```

## Full runs

```bash
python copper_conductivity.py --output-dir ../round2_full_results/copper
python band_structure_graphene.py --fetch --output-dir ../round2_full_results/graphene
python anharmonic_oscillator_energy_levels.py --output-dir ../round2_full_results/anharmonic
python photonic_crystals.py --output-dir ../round2_full_results/photonic
python quasicrystals_electronic_transmission.py --output-dir ../round2_full_results/quasicrystal
python generate_round2_latex.py --results-root ../round2_full_results
```

Use `--quick` / `--smoke-test` only for execution checks. Do not use quick/synthetic outputs as manuscript evidence.

Attribution
-----------

The original experimental code was authored by Renata Wong and released as supplementary material for the manuscript "Physics-Informed Support Vector Kernels via Green-Function Analogies and Jackson–Chebyshev Spectral Design."

The revised versions in this repository were prepared by Nan-Hong Kuo for the major revision of the manuscript, based on the original code structure and scientific examples.

Both the original code and the revised code are distributed under the MIT License. Original copyright notices should be retained in all derivative files.

Recommended citation
--------------------

If you use the original code, please cite:

    Renata Wong,
    Physics-Informed Support Vector Kernels via Green-Function Analogies and Jackson–Chebyshev Spectral Design,
    GitHub repository:
    https://github.com/renatawong/svm-kernels-as-quantum-propagators

If you use the revised scripts, please also cite:

    Nan-Hong Kuo,
    Physics-Informed Support Vector Kernels via Green-Function Analogies and Jackson–Chebyshev Spectral Design,
    GitHub repository:
    https://github.com/kuonanhong/SVM-kernels-as-quantum-propagators

License
-------

This repository is released under the MIT License. See the LICENSE file for details.

Because the revised scripts are derived from the original repository by Renata Wong, the original copyright and attribution notices are retained.
