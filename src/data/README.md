# Data Module

Scripts for handling the full data pipeline:

- `data_ingestion.py`: Load data from files or APIs
- `data_preprocessing.py`: Clean and normalize data
- `data_validation.py`: Check for missing values, schema conformity
- `data_splitting.py`: Train/test/validation set creation

### `data_ingestion.py`
This script handles the loading of data from various sources like files, APIs, or databases. It ensures that the data is properly loaded into the desired format (usually a Pandas DataFrame).

**Key Function**
- `load_dataset(filepath: str) -> pd.DataFrame`: Reads the .csv file and returns the Dataframe.

### `data_preprocessing.py`
This script contains the core functionality for cleaning and normalizing the data. It deals with handling missing values, encoding categorical variables, and scaling numerical features. It is an essential part of preparing data for machine learning models.

**Key Functions**
- `emotion_mapping(emotion_column: Union[pd.Series, list]) -> pd.Series`: Ensures that the emotion in the dataset alligns with the 6 core emotions.
- `map_label(label)`: Is a function within the `emotion_mapping` that labels the emotions accordingly. It also ensures that the string is in lowercase letters for readability.

### `data_validation.py`
This script checks for issues in the data such as missing values, invalid data types, or inconsistent formats. It ensures that the data adheres to a specific schema and is ready for processing.

**Key Function:**
- `handle_missing_values(dataframe: pd.DataFrame, method: str = 'drop') -> pd.DataFrame`: Handles missing values by removing the row.

### `data_splitting.py`
This script handles splitting the data into training, testing, and validation sets. It ensures that the data is appropriately partitioned for model training and evaluation.

**Key Function:**
- `split_dataset(dataframe, target_column, test_size=0.2, val_size=0.1, random_state=None)`: Splits the dataframe into training, testing and validation sets.

**Note:** These modules are designed to be used together as part of a larger machine learning pipeline for tasks such as emotion classification in NLP.
