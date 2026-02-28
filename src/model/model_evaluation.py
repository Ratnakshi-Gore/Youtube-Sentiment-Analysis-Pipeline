import numpy as np
import pandas as pd
import pickle
import logging
import yaml
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
from mlflow.models import infer_signature
from typing import Any

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)  # Fill any NaN values
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except Exception as e:
        logger.error('Error loading data from %s: %s', file_path, e)
        raise


def load_model(model_path: str):
    """Load the trained model."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', model_path)
        return model
    except Exception as e:
        logger.error('Error loading model from %s: %s', model_path, e)
        raise


def load_vectorizer(vectorizer_path: str) -> CountVectorizer:
    """Load the saved BoW vectorizer."""
    try:
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        logger.debug('BoW vectorizer loaded from %s', vectorizer_path)
        return vectorizer
    except Exception as e:
        logger.error('Error loading vectorizer from %s: %s', vectorizer_path, e)
        raise


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters loaded from %s', params_path)
        return params
    except Exception as e:
        logger.error('Error loading parameters from %s: %s', params_path, e)
        raise


def validate_vectorizer_params(vectorizer: CountVectorizer, model_params: dict) -> None:
    """Validate loaded vectorizer against params.yaml model_building settings."""
    if vectorizer.__class__.__name__ != model_params['vectorizer_type']:
        raise ValueError(
            f"Vectorizer type mismatch: expected {model_params['vectorizer_type']}, got {vectorizer.__class__.__name__}"
        )

    if tuple(vectorizer.ngram_range) != tuple(model_params['ngram_range']):
        raise ValueError(
            f"ngram_range mismatch: expected {tuple(model_params['ngram_range'])}, got {vectorizer.ngram_range}"
        )

    if vectorizer.max_features != model_params['max_features']:
        raise ValueError(
            f"max_features mismatch: expected {model_params['max_features']}, got {vectorizer.max_features}"
        )

    if vectorizer.min_df != model_params['min_df']:
        raise ValueError(
            f"min_df mismatch: expected {model_params['min_df']}, got {vectorizer.min_df}"
        )

    if vectorizer.max_df != model_params['max_df']:
        raise ValueError(
            f"max_df mismatch: expected {model_params['max_df']}, got {vectorizer.max_df}"
        )


def validate_model_params(model: Any, model_params: dict) -> None:
    """Validate loaded model against params.yaml model_building settings."""
    if model.__class__.__name__ != model_params['model_type']:
        raise ValueError(
            f"Model type mismatch: expected {model_params['model_type']}, got {model.__class__.__name__}"
        )

    if float(model.C) != float(model_params['C']):
        raise ValueError(f"C mismatch: expected {model_params['C']}, got {model.C}")

    if model.penalty != model_params['penalty']:
        raise ValueError(
            f"penalty mismatch: expected {model_params['penalty']}, got {model.penalty}"
        )

    if model.solver != model_params['solver']:
        raise ValueError(
            f"solver mismatch: expected {model_params['solver']}, got {model.solver}"
        )

    if model.class_weight != model_params['class_weight']:
        raise ValueError(
            f"class_weight mismatch: expected {model_params['class_weight']}, got {model.class_weight}"
        )

    if int(model.max_iter) != int(model_params['max_iter']):
        raise ValueError(
            f"max_iter mismatch: expected {model_params['max_iter']}, got {model.max_iter}"
        )


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """Evaluate the model and log classification metrics and confusion matrix."""
    try:
        # Predict and calculate classification metrics
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        logger.debug('Model evaluation completed')

        return report, cm
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise


def log_confusion_matrix(cm, dataset_name):
    """Log confusion matrix as an artifact."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Save confusion matrix plot as a file and log it to MLflow
    cm_file_path = f'confusion_matrix_{dataset_name}.png'
    plt.savefig(cm_file_path)
    mlflow.log_artifact(cm_file_path)
    plt.close()

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        # Create a dictionary with the info we want to save
        model_info = {
            'run_id': run_id,
            'model_path': model_path
        }
        # Save the dictionary as a JSON file
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise


def main():
    mlflow.set_tracking_uri("http://100.54.42.252:5000")

    mlflow.set_experiment('dvc-pipeline-runs')
    
    with mlflow.start_run() as run:
        try:
            # Load parameters from YAML file
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
            params = load_params(os.path.join(root_dir, 'params.yaml'))

            # Log parameters
            for key, value in params['model_building'].items():
                mlflow.log_param(f"model_building.{key}", value)
            
            # Load model and vectorizer
            model = load_model(os.path.join(root_dir, 'logistic_regression_model.pkl'))
            vectorizer = load_vectorizer(os.path.join(root_dir, 'bow_vectorizer.pkl'))

            # Validate loaded artifacts against params.yaml
            validate_vectorizer_params(vectorizer, params['model_building'])
            validate_model_params(model, params['model_building'])

            # Load test data for signature inference
            test_data = load_data(os.path.join(root_dir, 'data/interim/test_processed.csv'))

            # Prepare test data
            X_test_bow = vectorizer.transform(test_data['clean_comment'].values)
            y_test = test_data['category'].values

            # Create a DataFrame for signature inference (using first few rows as an example)
            input_example = pd.DataFrame(X_test_bow.toarray()[:5], columns=vectorizer.get_feature_names_out())  

            # Infer the signature
            signature = infer_signature(input_example, model.predict(X_test_bow[:5]))  

            # Log model with signature
            mlflow.sklearn.log_model(
                model,
                "logistic_regression_model",
                signature=signature,  
                input_example=input_example  
            )

            # Save model info
            # artifact_uri = mlflow.get_artifact_uri()
            model_path = "logistic_regression_model"
            save_model_info(run.info.run_id, model_path, 'experiment_info.json')

            # Log the vectorizer as an artifact
            mlflow.log_artifact(os.path.join(root_dir, 'bow_vectorizer.pkl'))

            # Evaluate model and get metrics
            report, cm = evaluate_model(model, X_test_bow, y_test)

            # Log classification report metrics for the test data
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    mlflow.log_metrics({
                        f"test_{label}_precision": metrics['precision'],
                        f"test_{label}_recall": metrics['recall'],
                        f"test_{label}_f1-score": metrics['f1-score']
                    })

            # Log confusion matrix
            log_confusion_matrix(cm, "Test Data")

            # Add important tags
            mlflow.set_tag("model_type", params['model_building']['model_type'])
            mlflow.set_tag("vectorizer_type", params['model_building']['vectorizer_type'])
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "YouTube Comments")

        except Exception as e:
            logger.error(f"Failed to complete model evaluation: {e}")
            print(f"Error: {e}")

if __name__ == '__main__':
    main()
