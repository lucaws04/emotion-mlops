# Utilities Module

General utility functions used across the project.

- `loggers.py`: Sets up and configures project logging.

### `logger.py`
Sets up a project-wide logger that outputs logs to both the console and a log file. Useful for tracking the flow of training, evaluation, and inference processes.

**Key function**
- `setup_logger(name: str) -> logging.Logger`: Creates a logger with the specified name and attaches:
  - A console handler for real-time logging.
  - A file handler for persistent logs saved to `logs/app.log`.

  **Note:** These modules are designed to be used together as part of a larger machine learning pipeline for tasks such as emotion classification in NLP.
