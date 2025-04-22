"""
Model development module for the Student Performance Predictor.
Handles model training, evaluation, and selection.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV

# Import models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.utils.logger import Log
from src.utils.common import save_object, write_json, evaluate_models
from src.utils.exception_handler import CustomException
from config.config import ModelTrainerConfig


class ModelTrainer:
    """
    Class to handle model training operations.
    Trains multiple models, evaluates them, and selects the best one.
    """
    
    def __init__(self, config: ModelTrainerConfig):
        """
        Initialize the ModelTrainer with configuration.
        
        Args:
            config: Configuration for model training
        """
        self.config = config
        
    def get_models(self) -> Dict[str, Any]:
        """
        Get dictionary of models to train.
        
        Returns:
            Dictionary of model name to model object
        """
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "XGBoost": XGBRegressor(),
            "CatBoost": CatBoostRegressor(verbose=False),
            "AdaBoost": AdaBoostRegressor()
        }
        
        return models
        
    def get_model_params(self) -> Dict[str, Dict]:
        """
        Get hyperparameters for each model.
        
        Returns:
            Dictionary of model name to hyperparameters
        """
        try:
            # Load params from YAML
            with open("config/params.yaml", 'r') as f:
                params = yaml.safe_load(f)
                
            model_params = {
                "Linear Regression": {},
                "Decision Tree": params["model_trainer"]["models"]["decision_tree"],
                "Random Forest": params["model_trainer"]["models"]["random_forest"],
                "Gradient Boosting": params["model_trainer"]["models"]["gradient_boosting"],
                "XGBoost": params["model_trainer"]["models"]["xgboost"],
                "CatBoost": params["model_trainer"]["models"]["catboost"],
                "AdaBoost": params["model_trainer"]["models"]["adaboost"]
            }
            
            return model_params
            
        except Exception as e:
            Log.error(f"Exception occurred while loading model parameters: {str(e)}")
            raise CustomException(str(e), sys) from e
            
    def train_and_evaluate_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Train and evaluate multiple models.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of model name to performance metric
        """
        try:
            Log.info("Starting model training and evaluation")
            
            # Get models and params
            models = self.get_models()
            params = self.get_model_params()
            
            # Evaluate models
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params
            )
            
            # Save model report
            os.makedirs(os.path.dirname(self.config.model_report_path), exist_ok=True)
            write_json(self.config.model_report_path, model_report)
            
            Log.info(f"Model evaluation report: {model_report}")
            
            return model_report
            
        except Exception as e:
            Log.error(f"Exception occurred during model training and evaluation: {str(e)}")
            raise CustomException(str(e), sys) from e
            
    def get_best_model(self, model_report: Dict[str, float]) -> Tuple[str, float]:
        """
        Get the best model name and score from the model report.
        
        Args:
            model_report: Dictionary of model name to performance metric
            
        Returns:
            Tuple of (best_model_name, best_model_score)
        """
        try:
            # Get best model
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            Log.info(f"Best model: {best_model_name} with score: {best_model_score}")
            
            return best_model_name, best_model_score
            
        except Exception as e:
            Log.error(f"Exception occurred while getting best model: {str(e)}")
            raise CustomException(str(e), sys) from e
            
    def save_model(self, model: Any) -> str:
        """
        Save the trained model to a file.
        
        Args:
            model: Trained model to save
            
        Returns:
            Path to the saved model
            
        Raises:
            CustomException: If saving fails
        """
        try:
            Log.info(f"Saving model to {self.config.model_path}")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
            
            # Save model
            save_object(self.config.model_path, model)
            
            Log.info("Model saved successfully")
            
            return self.config.model_path
            
        except Exception as e:
            Log.error(f"Exception occurred while saving model: {str(e)}")
            raise CustomException(str(e), sys) from e
            
    def save_evaluation_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str
    ) -> str:
        """
        Calculate and save evaluation metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            model_name: Name of the model
            
        Returns:
            Path to the saved evaluation metrics
            
        Raises:
            CustomException: If saving fails
        """
        try:
            Log.info("Calculating evaluation metrics")
            
            # Calculate metrics
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            
            # Create evaluation report
            evaluation = {
                "model_name": model_name,
                "mean_absolute_error": float(mae),
                "mean_squared_error": float(mse),
                "root_mean_squared_error": float(rmse),
                "r2_score": float(r2)
            }
            
            Log.info(f"Evaluation metrics: {evaluation}")
            
            # Save evaluation
            os.makedirs(os.path.dirname(self.config.evaluation_path), exist_ok=True)
            write_json(self.config.evaluation_path, evaluation)
            
            Log.info(f"Evaluation metrics saved to {self.config.evaluation_path}")
            
            return self.config.evaluation_path
            
        except Exception as e:
            Log.error(f"Exception occurred while saving evaluation metrics: {str(e)}")
            raise CustomException(str(e), sys) from e
            
    def initiate_model_training(
        self,
        train_array: np.ndarray,
        test_array: np.ndarray
    ) -> Tuple[str, str, float]:
        """
        Execute the complete model training pipeline.
        
        Args:
            train_array: Training data array
            test_array: Test data array
            
        Returns:
            Tuple of (model_path, evaluation_path, best_score)
            
        Raises:
            CustomException: If any step in the pipeline fails
        """
        try:
            Log.info("Starting model training pipeline")
            
            # Split arrays into features and target
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            Log.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            Log.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
            
            # Train and evaluate models
            model_report = self.train_and_evaluate_models(X_train, y_train, X_test, y_test)
            
            # Get best model
            best_model_name, best_model_score = self.get_best_model(model_report)
            
            # Check if best model meets threshold
            if best_model_score < self.config.r2_threshold:
                Log.warning(f"Best model score {best_model_score} is below threshold {self.config.r2_threshold}")
                raise CustomException("No model meets the performance threshold")
            
            # Get the best model
            models = self.get_models()
            best_model = models[best_model_name]
            
            # Train the best model with all data
            best_model.fit(X_train, y_train)
            
            # Make predictions with the best model
            y_pred = best_model.predict(X_test)
            
            # Save the best model
            model_path = self.save_model(best_model)
            
            # Save evaluation metrics
            evaluation_path = self.save_evaluation_metrics(y_test, y_pred, best_model_name)
            
            Log.info("Model training pipeline completed successfully")
            
            return model_path, evaluation_path, best_model_score
            
        except Exception as e:
            Log.error(f"Exception occurred during model training pipeline: {str(e)}")
            raise CustomException(str(e), sys) from e