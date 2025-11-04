Getting Started
===============

Welcome to the MLOps AzureML Pipeline Project!
This guide will help you set up your environment, install dependencies, and run your first pipeline step.

Project Overview
----------------

This repository contains a production-ready MLOps pipeline built on **Azure Machine Learning**, integrating:
- CI/CD automation with GitHub Actions
- Model training, evaluation, and deployment
- Monitoring and drift detection
- Python packaging with Poetry
- FastAPI-based inference service

Requirements
------------

Before getting started, ensure the following are installed:

- Python >= 3.8
- Git
- Poetry (for dependency management)
- Azure CLI
- Docker (optional but recommended)
- An Azure subscription with a configured workspace

Installation
------------

1. **Clone the repository**

.. code-block:: bash

    git clone https://github.com/BredaUniversityADSAI/2024-25d-fai2-adsai-group-nlp9.git
    cd 2024-25d-fai2-adsai-group-nlp9

2. **Install dependencies using Poetry**

.. code-block:: bash

    poetry install

3. **Activate the virtual environment**

.. code-block:: bash

    poetry shell

4. **Set up pre-commit hooks**

.. code-block:: bash

    pre-commit install

5. **Configure Azure workspace**

Ensure you have access to an AzureML workspace. Then, log in and configure the workspace.

.. code-block:: bash

    az login
    az account set --subscription "<your-subscription-id>"

Update the `azureml/config/config.json` with your workspace details.

Running Tests
-------------

To run all unit tests:

.. code-block:: bash

    poetry run pytest

To run a specific test file:

.. code-block:: bash

    poetry run pytest tests/test_model_saving.py

To view test coverage:

.. code-block:: bash

    coverage run -m pytest
    coverage report -m
    # Or generate an HTML report
    coverage html

Linting and Formatting
----------------------

Linting is automated via pre-commit hooks. You can also run it manually:

.. code-block:: bash

    pre-commit run --all-files

Tools used:
- Black (code formatting)
- Flake8 (style guide enforcement)
- isort (import sorting)
- YAML, whitespace, and large file checks

First Pipeline Execution
------------------------

To test AzureML pipeline submission:

.. code-block:: bash

    python azureml/jobs/emotion-classifier_pipeline.py

Documentation
-------------

Project documentation is located in the `docs/` folder. For detailed pipeline breakdowns, refer to:

- `azureml/components/README.md`
- `src/README.md`

Troubleshooting
---------------

- If you encounter missing packages, ensure Poetry dependencies are synced.
- Azure authentication errors? Make sure you're logged in with `az login`.
- Always check `.pre-commit-config.yaml` and `.github/workflows/deploy.yml` for automation issues.

Need Help?
----------

Please refer to the main `README.md` or contact the maintainers listed there.

