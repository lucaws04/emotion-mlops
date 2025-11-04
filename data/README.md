# Data Directory

This folder contains data used for the project, organized by processing stage.

- `raw/`: Untouched, original datasets
- `processed/`: Cleaned and formatted data ready for analysis
- `external/`: Data from third-party sources

Avoid manually modifying files in `processed/` — they should be generated via scripts in `src/data/`.
