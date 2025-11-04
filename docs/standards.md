# Project Standards: Naming Conventions & Folder structure
This document defines the naming conventions and folder structure standards to be followed across the project to ensure consistency, readability, and maintainalbility of the codebases.

## Naming Conventions
### General Rules
- Use English for all names (files, folders, variables, functions).
- Be descriptive and consistent.
- Avoid abbrevations unless widely recognized (e.g., `id`, `url`).
- Separate words using appropriate case formats depending on context.

### Snake Case: `snake_case`
Use snake_case for:
- File names: `data_loader.py`, `train_model.py`
- Folder names: `data_processing/,` `model_utils/`
- Function and variable names: `load_data()`, emotion_labe`ls

### Pascal Case: `PascalCase`
Use PascalCase for:
- Class names: `TextPreprocessor`, `EmotionClassifier`

### Camel Case: `camelCase`
Use camelCase only for:
- JSON keys (if interacting with external APIs that require it)

## Folder Structure Guidelines
```
project_root/
│
├── data/                  # Raw and processed data (never committed to version control)
│   ├── raw/
│   └── processed/
│
├── notebooks/             # Jupyter notebooks for exploration
│
├── src/                   # All source code
│   ├── data_processing/   # Data loading, cleaning, and transformation
│   ├── models/            # Model architecture and training scripts
│   ├── utils/             # Utility functions (e.g., logging, metrics)
│   └── cli/               # CLI interfaces and handlers
│
├── tests/                 # Unit and integration tests
│
├── configs/               # Configuration files (YAML, JSON, etc.)
│
├── scripts/               # Helper or automation scripts
│
├── outputs/               # Model outputs, logs, predictions (ignored in .gitignore)
│
├── README.md              # Project overview and instructions
├── standards.md           # This standards document
└── requirements.txt       # Python dependencies
```

## File Naming Rules
- Use `snake_case` for all file names.
- File names should be short and descriptive of their content or purpose.
- Avoid redundant prefixes/suffixes (e.g., model_model.py is incorrect).

| File Type      | Example                   |
| -------------- | ------------------------- |
| Python scripts | `train_model.py`          |
| Config files   | `model_config.yaml`       |
| Test files     | `test_data_processing.py` |
| Notebook files | `eda_emotions.ipynb`      |
