import pandas as pd
import os
import pickle
import yaml
import logging
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)  # Fill any NaN values
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def apply_bow(train_data: pd.DataFrame, max_features: int, ngram_range: tuple, min_df: int, max_df: float) -> tuple:
    """Apply Bag of Words (CountVectorizer) with ngrams to the data."""
    try:
        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df
        )

        X_train = train_data['clean_comment'].values
        y_train = train_data['category'].values

        # Perform BoW transformation
        X_train_bow = vectorizer.fit_transform(X_train)

        logger.debug("BoW transformation complete. Train shape: %s", X_train_bow.shape)

        # Save the vectorizer in the root directory
        with open(os.path.join(get_root_directory(), 'bow_vectorizer.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f)

        logger.debug('BoW (CountVectorizer) applied and data transformed')
        return X_train_bow, y_train
    except Exception as e:
        logger.error('Error during BoW transformation: %s', e)
        raise


def train_logistic_regression(
    X_train,
    y_train,
    c_value: float,
    penalty: str,
    solver: str,
    class_weight: str,
    max_iter: int
) -> LogisticRegression:
    """Train a Logistic Regression model."""
    try:
        best_model = LogisticRegression(
            C=c_value,
            penalty=penalty,
            solver=solver,
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=42
        )
        best_model.fit(X_train, y_train)
        logger.debug('Logistic Regression model training completed')
        return best_model
    except Exception as e:
        logger.error('Error during Logistic Regression model training: %s', e)
        raise


def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise


def get_root_directory() -> str:
    """Get the root directory (two levels up from this script's location)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))


def main():
    try:
        # Get root directory and resolve the path for params.yaml
        root_dir = get_root_directory()

        # Load parameters from the root directory
        params = load_params(os.path.join(root_dir, 'params.yaml'))
        max_features = params['model_building']['max_features']
        # BoW with bigrams only
        ngram_range = (1, 2)
        min_df = params['model_building']['min_df']
        max_df = params['model_building']['max_df']
        c_value = params['model_building']['C']
        penalty = params['model_building']['penalty']
        solver = params['model_building']['solver']
        class_weight = params['model_building']['class_weight']
        max_iter = params['model_building']['max_iter']

        # Load the preprocessed training data from the interim directory
        train_data = load_data(os.path.join(root_dir, 'data/interim/train_processed.csv'))

        # Apply BoW feature engineering on training data
        X_train_bow, y_train = apply_bow(train_data, max_features, ngram_range, min_df, max_df)

        # Train the Logistic Regression model using hyperparameters from params.yaml
        best_model = train_logistic_regression(
            X_train_bow,
            y_train,
            c_value,
            penalty,
            solver,
            class_weight,
            max_iter
        )

        # Save the trained model in the root directory
        save_model(best_model, os.path.join(root_dir, 'logistic_regression_model.pkl'))

    except Exception as e:
        logger.error('Failed to complete the feature engineering and model building process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
