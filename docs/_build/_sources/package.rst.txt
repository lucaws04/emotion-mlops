Python Package Plan
===================

This project implements a modular and reproducible pipeline for Natural Language Processing (NLP) tasks, following clean software engineering practices.

Directory Overview
------------------

.. code-block:: text

    2024-25d-fai2-adsai-group-nlp9/
    ├── app/                    # Application interface or backend
    │   ├── README.md
    │   └── main.py
    ├── data/                   # Data storage
    │   ├── raw/                # Original data files
    │   │   └── README.md
    │   ├── processed/          # Cleaned and transformed data
    │   │   └── README.md
    │   └── external/           # Data from third-party sources
    │       └── README.md
    ├── docs/                   # Project documentation
    │   └── README.md
    ├── models/                 # Trained ML model weights, checkpoints
    │   └── README.md
    ├── notebooks/              # Jupyter notebooks for analysis and prototyping
    │   └── README.md
    ├── src/                    # Source code modules
    │   ├── __init__.py
    │   ├── README.md
    │   ├── data/               # Data loading and processing
    │   │   ├── __init__.py
    │   │   ├── data_ingestion.py
    │   │   ├── data_preprocessing.py
    │   │   ├── data_validation.py
    │   │   └── data_splitting.py
    │   ├── features/           # Feature engineering
    │   │   ├── __init__.py
    │   │   ├── feature_extraction.py
    │   │   ├── feature_scaling.py
    │   │   └── feature_selection.py
    │   ├── models/             # Model lifecycle code
    │   │   ├── __init__.py
    │   │   ├── model_definitions.py
    │   │   ├── model_training.py
    │   │   ├── model_evaluation.py
    │   │   ├── model_tuning.py
    │   │   └── model_saving.py
    │   └── utils/              # Helper functions, config, and visual tools
    │       ├── __init__.py
    │       ├── helpers.py
    │       ├── visualization.py
    │       └── configuration.py
    ├── tests/                  # Unit tests and integration tests
    │   ├── test_api.py
    │   ├── test_model.py
    │   ├── test_predict.py
    │   └── README.md
    ├── main.py                 # Main entrypoint to run the pipeline or app
    ├── .gitignore              # Git ignored files
    ├── LICENSE                 # Project license
    ├── README.md               # Project overview
    └── pyproject.toml          # Project dependencies and build config

How to Use
----------

**Run the main pipeline**::

    python main.py

**Run unit tests**::

    pytest tests/

**Build documentation** (from within the `docs/` directory)::

    sphinx-build -b html . _build/html

Recommended Workflow
--------------------

1. **Raw data** is placed in `data/raw/`.
2. **Preprocessing and cleaning** is done using scripts in `src/data/`.
3. **Features** are extracted and transformed using `src/features/`.
4. **Model training and evaluation** is performed via `src/models/`.
5. **Notebooks** can be used for prototyping or reporting.
6. **Final models** are saved in `models/`.
7. **Documentation** is written and maintained under `docs/`.

