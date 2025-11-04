# Tests

This directory contains unit and integration tests for the `nlp_project`. The tests are designed to verify the correctness of the components in the `src/` folder, ensuring that the data loading, model creation, training, evaluation, and prediction functions work as expected. The goal of this test suite is to ensure reliability, maintainability, and correctness of the core components, including data handling, preprocessing, model definition, evaluation, and persistence.

## Running the Tests

You can run all tests using `pytest` with or without coverage:

### Basic Test Run
```
pytest tests/
```

Run with Coverage Report
```
coverage run -m pytest tests/
coverage report -m
```

## Test Strategy

- **Mocking**: Transformer models and heavy components are mocked using `unittest.mock` to avoid loading real models. This makes the tests lightweight and fast.

- **Isolation**: Each test class or function is designed to test a single unit of behavior. No external dependencies or datasets are required.

- **Assertions**: Functions are validated by checking:

 - Method calls (e.g., `assert_called_once_with`)

 - Return values

 - Side effects (e.g., file save/load)

## Test Requirements
Ensure you have the following installed in your virtual environment:
```
pip install -r requirements.txt
# Or for just testing dependencies:
pip install pytest coverage transformers scikit-learn
```

## Notes
- When mocking model behavior, verify both the input arguments and output shape/type.
- If you modify logic in `src/`, corresponding test files must be updated.
- Warnings from `pytest-asyncio` can be silenced by configuring the event loop scope explicity in pytest.ini