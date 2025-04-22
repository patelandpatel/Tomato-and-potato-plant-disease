"""
Common utility functions for the Student Performance Predictor.
Contains functions for file operations, object serialization, etc.
"""

import os
import sys
import yaml
import json
import pickle
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union

from src.utils.logger import Log
from src.utils.exception_handler import CustomException


def read_yaml(file_path: str) -> Dict:
    """
    Read a YAML file and return its contents as a dictionary.
    
    Args:
        file_path: Path to the YAML file
        
    Returns:
        Dictionary containing YAML file contents
        
    Raises:
        CustomException: If the file cannot be read
    """
    try:
        with open(file_path, 'r') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        Log.error(f"Exception occurred while reading YAML file: {file_path}")
        raise CustomException(str(e), sys) from e


def write_yaml(file_path: str, data: Dict) -> None:
    """
    Write a dictionary to a YAML file.
    
    Args:
        file_path: Path to the YAML file
        data: Dictionary to write
        
    Raises:
        CustomException: If the file cannot be written
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as yaml_file:
            yaml.dump(data, yaml_file, default_flow_style=False)
    except Exception as e:
        Log.error(f"Exception occurred while writing YAML file: {file_path}")
        raise CustomException(str(e), sys) from e


def read_json(file_path: str) -> Dict:
    """
    Read a JSON file and return its contents as a dictionary.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary containing JSON file contents
        
    Raises:
        CustomException: If the file cannot be read
    """
    try:
        with open(file_path, 'r') as json_file:
            return json.load(json_file)
    except Exception as e:
        Log.error(f"Exception occurred while reading JSON file: {file_path}")
        raise CustomException(str(e), sys) from e


def write_json(file_path: str, data: Dict) -> None:
    """
    Write a dictionary to a JSON file.
    
    Args:
        file_path: Path to the JSON file
        data: Dictionary to write
        
    Raises:
        CustomException: If the file cannot be written
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
    except Exception as e:
        Log.error(f"Exception occurred while writing JSON file: {file_path}")
        raise CustomException(str(e), sys) from e


def save_object(file_path: str, obj: Any) -> None:
    """
    Save any Python object to a pickle file.
    
    Args:
        file_path: Path to save the object
        obj: The object to save
        
    Raises:
        CustomException: If the object cannot be saved
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        Log.error(f"Exception occurred while saving object to file: {file_path}")
        raise CustomException(str(e), sys) from e


def load_object(file_path: str) -> Any:
    """
    Load a Python object from a pickle file.
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        The loaded object
        
    Raises:
        CustomException: If the object cannot be loaded
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        Log.error(f"Exception occurred while loading object from file: {file_path}")
        raise CustomException(str(e), sys) from e


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Pandas DataFrame containing the data
        
    Raises:
        CustomException: If the data cannot be loaded
    """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        Log.error(f"Exception occurred while loading data from file: {file_path}")
        raise CustomException(str(e), sys) from e


def save_data(dataframe: pd.DataFrame, file_path: str) -> None:
    """
    Save a pandas DataFrame to a CSV file.
    
    Args:
        dataframe: The DataFrame to save
        file_path: Path to save the CSV file
        
    Raises:
        CustomException: If the data cannot be saved
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        dataframe.to_csv(file_path, index=False)
    except Exception as e:
        Log.error(f"Exception occurred while saving data to file: {file_path}")
        raise CustomException(str(e), sys) from e


def evaluate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    models: Dict[str, Any],
    params: Dict[str, Dict]
) -> Dict[str, float]:
    """
    Train and evaluate multiple models with hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        models: Dictionary of model name to model object
        params: Dictionary of model name to hyperparameters
        
    Returns:
        Dictionary of model name to performance metric
        
    Raises:
        CustomException: If model evaluation fails
    """
    try:
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import r2_score
        
        report = {}
        
        for model_name, model in models.items():
            Log.info(f"Training and evaluating {model_name}...")
            
            # Get parameters for the current model
            model_params = params.get(model_name, {})
            
            # If parameters exist, perform grid search
            if model_params:
                gs = GridSearchCV(
                    estimator=model,
                    param_grid=model_params,
                    cv=3,
                    n_jobs=-1,
                    verbose=2
                )
                gs.fit(X_train, y_train)
                
                # Set best parameters
                model.set_params(**gs.best_params_)
                
                Log.info(f"Best parameters for {model_name}: {gs.best_params_}")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            Log.info(f"{model_name} - Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}")
            
            # Store test metric in report
            report[model_name] = test_r2
        
        return report
        
    except Exception as e:
        Log.error
        