SVM Kernels as Quantum Propagators: Revised Experimental Scripts
================================================================

This repository contains revised experimental Python scripts used for the major-revision stage of the manuscript:

    Support Vector Machine Kernels as Quantum Propagators

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

    1. copper_conductivity_revised.py
       Revised from the original copper-conductivity experiment.

    2. band_structure_graphene_revised.py
       Revised from the original graphene band-structure experiment.

    3. anharmonic_oscillator_energy_levels_revised.py
       Revised from the original anharmonic-oscillator experiment.

    4. photonic_crystals_revised.py
       Revised from the original photonic-crystal transmission experiment.

    5. quasicrystals_electronic_transmission_revised.py
       Revised from the original quasiperiodic electronic-transmission experiment.

Attribution
-----------

The original experimental code was authored by Renata Wong and released as supplementary material for the manuscript "Support Vector Machine Kernels as Quantum Propagators."

The revised versions in this repository were prepared by Nan-Hong Kuo for the major revision of the manuscript, based on the original code structure and scientific examples.

Both the original code and the revised code are distributed under the MIT License. Original copyright notices should be retained in all derivative files.

Recommended citation
--------------------

If you use the original code, please cite:

    Renata Wong,
    svm-kernels-as-quantum-propagators,
    GitHub repository:
    https://github.com/renatawong/svm-kernels-as-quantum-propagators

If you use the revised scripts, please also cite:

    Nan-Hong Kuo,
    SVM-kernels-as-quantum-propagators,
    GitHub repository:
    https://github.com/kuonanhong/SVM-kernels-as-quantum-propagators

Important security note
-----------------------

Some scripts may require access to external scientific databases such as the Materials Project. Do not commit private API keys directly into public source code. Use environment variables or a local configuration file excluded by .gitignore.

For example:

    export MP_API_KEY="your-materials-project-api-key"

Then read it in Python using:

    import os
    API_KEY = os.environ.get("MP_API_KEY")

License
-------

This repository is released under the MIT License. See the LICENSE file for details.

Because the revised scripts are derived from the original repository by Renata Wong, the original copyright and attribution notices are retained.