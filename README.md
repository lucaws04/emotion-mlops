[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/i4PR14xw)

This repository contains a complete MLOps pipeline built on Azure Machine Learning, following modern engineering practices including CI/CD, automated testing, monitoring, and cloud deployment.

## MLOps Engineer Project Roadmap

### Timeline Overview (8 Weeks)

| Week | Tasks |
|------|-------|
| **Week 1** | - Plan and divide tasks<br> - Create Python package plan<br> - Set up environments |
| **Week 2** | - Produce a production-ready code package<br> - Produce documentation for the package |
| **Week 3** | - Unit test the package<br> - Basic deployment |
| **Week 4** | - Create Data Store, Assets, and Pipelines<br> - Create environment assets |
| **Week 5** | - Create training pipelines<br> - CI/CD and automation |
| **Week 6** | - Deploy API endpoints in the cloud<br> - Create monitoring tools and pipelines |
| **Week 7** | - Implement CI/CT/CD<br> - Wrap-up, testing, and demo |
| **Week 8** | - Final wrap-up, testing, and demo |

---

### Workstreams

#### Scoping & Planning
- Plan and divide tasks
- Python package plan
- Environment setup

#### MLOps Level 0
- MVP Application: Inference
- Production-ready packaging and documentation

#### MLOps Level 1/2
- Training in the cloud
- Unit testing
- Basic deployment
- Create pipelines and environment assets

#### MLOps Level 3/4
- CI/CD automation
- API deployment
- Monitoring and retraining

#### Final Phase
- CI/CT/CD implementation
- Testing
- Project demo and delivery

---

### Deadlines
- Week 2: MVP Package Complete
- Week 5: CI/CD Setup Complete
- Week 7: Final System Ready
- Week 8: Project Demo & Delivery

---

## Python Package

### Installing the `src/` Package

Many scripts depend on the `src/` package, which is a local Python package living inside the project. To use these scripts properly, **you must build and install this package as a wheel before running any component or pipeline scripts.**

#### **Build the Wheel**

Make sure you have `build` installed:

```bash
pip install build
```

From the root of the project, run:

```bash
python setup.py bdist_wheel
```

This will create a `.whl` file inside the `dist/` folder, for example: `setup-1.0.9-py3-none-any.whl`.

#### **Install the Wheel**

Install the newly built wheel file:

```bash
pip install dist/setup-1.0.9-py3-none-any.whl
```

**Note**: If you make any changes to the code inside the `src/` folder, you must rebuild the wheel and reinstall it to ensure those changes are reflected.

#### **Setting the Python Path**

The `src/` package lives locally in the project directory and is not installed in the global Python environment by default. To allow Python to correctly locate and import it when running scripts, you need to set the `PYTHONPATH` environment variable to the project root directory.

On Windows Powershell:

```bash
$env:PYTHONPATH = (Get-Location).Path
```

On macOS/Linux:

```bash
export PYTHONPATH=$(pwd)
```

##### **Important Notes:**

- This command needs to be run only **once** per shell session before running your Python scripts.
- It ensures all scripts can find and import the `src/` package without import errors.
- Failure to do this will result in import errors when scripts try to access `src` modules.
- Always rebuild and reinstall the wheel after making changes inside the `src/` directory to keep the installed package up to date.

---

## Project Setup & Development Guide

This project uses **Poetry** for dependency and environment management, **pre-commit** for linting, and **pytest** for testing. Below are detailed instructions on setting up, linting, testing, and maintaining the environment.

### Poetry Setup

We use [**Poetry**](https://python-poetry.org/) to manage dependencies, packaging, and virtual environments for this project.

#### **Key Information**

- The main package is in the `src/` folder.
- Python version requirement: >=3.9, <4.0
- Formatting tools use consistent settings in `pyproject.toml` (Black, Isort, Flake8).

#### **Installing Dependencies**

To install all dependencies (including dev packages  (testing, linting, formatting)):

```bash
poetry install
```

This will create and activate a virtual environment automatically.

#### **Running Inside Poetry Environment**

To run commands inside the Poetry-managed environment:

```bash
poetry run python path/to/script.py
```

### Installing from `requirements.txt`

Installing all project requirements can be done by installing `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Linting Guide

We use [pre-commit](https://pre-commit.com/) to manage linting and formatting hooks, ensuring code quality before every commit.

#### **Tools Used (configured in `.pre-commit-config.yaml`)**

- Trailing Whitespace Remover: removes unnecessary trailing whitespaces
- End of File Fixer: ensures files end with a newline
- Check YAML: validates YAML syntax
- Check Added Large Files: blocks committing large files
- Flake8: Python style guide (PEP8) checks
- Black: automatic code formatter
- Isort: automatic import sorter

#### **Install pre-commit:**

```bash
pip install pre-commit
```

#### **Install Git hooks:**

```bash
pre-commit install
```

This will install hooks `.git/hooks` to run automatically on commits.

#### **Run linting manually (optional):**

Run on all files:

```bash
pre-commit run --all files
```

Run on changed files only:

```bash
pre-commit run
```

#### **Update hooks when needed:**

```bash
pre-commit autoupdate
```

### Testing Guide

We use `pytest` for unit testing. test files are named using the pattern `test_*.py`. The tests are located in the `tests/` directory.

#### **Requirements**

- Python 3.8+
- Poetry (recommended) or ensure dependencies are installed

#### **Running Tests**

Run all tests with coverage:

```bash
coverage run -m pytest tests/
```

Run specific test file:

```bash
coverage run -m pytest tests/test_*.py
```

View coverage report in terminal:

```bash
coverage report -m
```

Or generate an HTML report:

```bash
coverage html
```

Run tests with Poetry environment:

```bash
poetry run pytest tests/test_model_saving.py
poetry run pytest --cov=src
```

---

## Docker Deployment

This project is containerized using Docker for easy deployment and consistent environments across different platforms. The Docker setup includes the complete MLOps pipeline with the emotion classification model, FastAPI backend, and all necessary dependencies.

### Prerequisites
- Docker (version 20.10 or higher)
- Docker Compose (optional, for multi-service setup)

### Project Structure for Docker
The Docker container includes:
- FastAPI application (`app/`)
- Trained emotion classification models
- Source package (`src/`)
- All Python dependencies and requirements

### Build and Run
```bash
# Build the Docker image
docker build -t emotion-classifier .

# Run the container with API exposed on port 8080
docker run -p 8080:8080 emotion-classifier
```

### Access Points
After running the container, you can access:
- **API Documentation**: http://localhost:8080/docs (Swagger UI)
- **Health Check**: http://localhost:8080/health

### Environment Variables
The container supports the following environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8080 | Port for the FastAPI application |
| `MODEL_PATH` | `/app/models` | Path to the trained models |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

Example with custom environment variables:
```bash
docker run -p 3000:3000 -e PORT=3000 -e LOG_LEVEL=DEBUG emotion-classifier
```

### Docker Image Details
- **Base Image**: Python 3.9-slim
- **Working Directory**: `/app`
- **Exposed Port**: 8080
- **Entry Point**: FastAPI application with Uvicorn server

### Troubleshooting

**Common Issues and Solutions:**

1. **Port already in use**:
   ```bash
   # Use a different port
   docker run -p 8081:8080 emotion-classifier
   ```

2. **Memory issues**:
   ```bash
   # Increase memory limit
   docker run --memory=2g -p 8080:8080 emotion-classifier
   ```

3. **Model loading errors**:
   - Ensure your trained models are included in the Docker image
   - Check that model files are in the correct directory structure

4. **Package import errors**:
   - The `src/` package is automatically installed during the Docker build process
   - PYTHONPATH is set correctly in the container

### Integration with Azure ML
The Docker container can be used alongside the Azure ML pipeline:
- For local testing of models trained in Azure ML
- As a deployment target for Azure Container Instances
- For edge deployment scenarios

---

## Azure ML Pipeline

This project provides a modular Azure Machine Learning (AzureML) pipeline for training and evaluating an emotion classification model using reusable components, environments, and job orchestration.

### Overview

- The pipeline is defined in [`azureml/pipeline/pipeline.py`](https://github.com/BredaUniversityADSAI/2024-25d-fai2-adsai-group-nlp9/blob/main/azureml/pipeline/pipeline.py) and is designed to run training and evaluation steps on GPU compute.
- Key pipeline components such as training and evaluation are registered in the AzureML workspace.
- Job submission and scheduling scripts allow running the pipeline on-demand or triggered by new data arrivals.
- Environment setup and compute resource management are handled via scripts in the [`azureml/environments/`](https://github.com/BredaUniversityADSAI/2024-25d-fai2-adsai-group-nlp9/tree/main/azureml/environments) folder.

To create or update the environment manually:

```bash
conda env update -f environment.yaml
```

Or

```bash
conda env create -f environment.yaml
```

*This environment is specifically tailored for AzureML pipelines using SDK v2*


### Basic Usage Steps
#### **Set up your environment:**
- Install dependencies using poetry
- Make sure to install the local `src` package via the provided wheel file.
- Set the `PYTHONPATH` environment variable to the project root to allow local imports.

```bash
# Windows PowerShell
$env:PYTHONPATH = (Get-Location).Path

# Linux/macOS
export PYTHONPATH=$(pwd)
```

#### **Register pipeline components:**

Use the registration scripts in [`azureml/components/`](https://github.com/BredaUniversityADSAI/2024-25d-fai2-adsai-group-nlp9/tree/main/azureml/components) to register or update components like training and evaluation.

#### **Upload and manage data assets:**

Use scripts in [`azureml/data_assets/`](https://github.com/BredaUniversityADSAI/2024-25d-fai2-adsai-group-nlp9/tree/main/azureml/data_assets) to register datasets with versioning support.

#### **Submit pipeline jobs:**

Run the pipeline by submitting jobs through scripts in ['azureml/jobs/'](https://github.com/BredaUniversityADSAI/2024-25d-fai2-adsai-group-nlp9/tree/main/azureml/jobs). For example:

```bash
python azureml/jobs/submit_pipeline.py
```

#### **Manage environments and compute:**

Configure and inspect AzureML environments and compute targets with scripts in [`azureml/environments/`](https://github.com/BredaUniversityADSAI/2024-25d-fai2-adsai-group-nlp9/tree/main/azureml/environments).

### More Information
For detailed instructions, configuration options, and advanced usage, please refer to the documentation and scripts within the [`azureml/`](https://github.com/BredaUniversityADSAI/2024-25d-fai2-adsai-group-nlp9/tree/main/azureml) directory.

---

## Project Structure

```
2024-25d-fai2-adsai-group-nlp9/
│
├── app/                    # FastAPI app (unified interface)
│   ├── main.py             # API entry point
│   ├── routes.py           # Endpoints for /translate, /classify, etc.
│   └── schemas.py          # Pydantic models for request/response
│
├── azureml/
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_component.py
│   │   ├── evaluation_component.py
│   │   ├── train_component.py
│   │   └── README.md
│   │
│   ├── data_assets/
│   │   ├── __init__.py
│   │   ├── list_data_assets.py
│   │   ├── upload_data_asset.py
│   │   └── README.md
│   │
│   ├── datastore/
│   │   ├── __init__.py
│   │   ├── list_datastores.py
│   │   └── README.md
│   │
│   ├── environments/
│   │   ├── __init__.py
│   │   ├── compute_targets.py
│   │   ├── list_environments.py
│   │   ├── pipeline_environment.py
│   │   └── README.md
│   │
│   ├── jobs/
│   │   ├── __init__.py
│   │   ├── schedule_job.py
│   │   ├── sumbit_pipeline_v2.py
│   │   ├── submit_sweep.py
│   │   └── README.md
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── pipeline_v2.py
│   │   └── README.md
│   │
│   └── scripts/
│   │   ├── __init__.py
│   │   ├── connecting_workspace.py
│   │   ├── evaluate.py
│   │   ├── evaluate.yml
│   │   ├── setup-1.0.3-py3-none-any.whl
│   │   ├── train_v2.py
│   │   ├── train_v2.yml
│   │   └── README.md
│   │
│   └── config/             # Azure subscription and workspace details
│       └──config.json
│
│
├── dashboard               # Dashboard contents
│
├── data/                   # (Optional) Training or retraining data logic
│   ├── raw/
│   │   └──README.MD
│   ├── processed/
│   │   └──README.MD
│   └── external/
│       └──README.MD/
│
├── docs/                   # Documentation
│   └── README.md
│
├── models/
│   └── README.md
│
├── monitoring/             # Performance monitoring/logging
│   ├── metrics.py
│   ├── drift.py
│   └── logger.py
│
├── notebooks/
│   └── README.md
│
├── src/                    # Source code for the package
│   ├── __init__.py
│   ├── README.md
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_preprocessing.py
│   │   ├── data_validation.py
│   │   ├─ data_splitting.py
│   │   └── README.md
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── feature_extraction.py
│   │   ├── feature_scaling.py
│   │   ├─ feature_selection.py
│   │   └── README.md
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model_definitions.py
│   │   ├── model_training.py
│   │   ├── model_evaluation.py
│   │   ├── model_tuning.py
│   │   ├─ model_saving.py
│   │   └── README.md
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       └── README.md
│
├── tests/                  # Unit and integration tests
│       ├── test_api.py
│       ├── test_data_ingestion.py
│       ├── test_data_validation.py
│       ├── test_data_splitting.py
│       ├── test_feature_extraction.py
│       ├── test_model_definitions.py
│       ├── test_model_training.py
│       ├── test_model_tuning.py
│       ├── test_model_saving.py
│       └── README.md
│
│
├── .github/workflows/      # CI/CD pipelines
│   └── deploy.yml
│
├── .dockerignore           # Files to ignore in Docker builds
├── .gitignore              # Files to ignore in version control
├── .pre-commit-config.yaml # Pre-commit hooks configuration
├── cli.py                  # Command-line interface for the package
├── Dockerfile              # For containerization
├── main.py                 # Main entry point for source code
├── pyproject.toml          # Project metadata and dependencies
├── README.md               # Project overview and setup instructions
└── requirements.txt        # Dependencies
```
