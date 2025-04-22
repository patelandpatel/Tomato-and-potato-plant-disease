"""
Data preprocessing module for the Student Performance Predictor.
Handles feature engineering, transformation, and preprocessing.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.logger import Log
from src.utils.common import save_object
from src.utils.exception_handler import CustomException
from config.config import DataPreprocessingConfig


class DataPreprocessor:
    """
    Class to handle data preprocessing operations.
    Creates preprocessing pipeline for both numerical and categorical features.
    """
    
    def __init__(self, config: DataPreprocessingConfig):
        """
        Initialize the DataPreprocessor with configuration.
        
        Args:
            config: Configuration for data preprocessing
        """
        self.config = config
        
    def get_preprocessor(self) -> ColumnTransformer:
        """
        Create a preprocessing pipeline for both numerical and categorical features.
        
        Returns:
            ColumnTransformer object with preprocessing pipelines
            
        Raises:
            CustomException: If creating the preprocessor fails
        """
        try:
            Log.info("Creating data preprocessing pipeline")
            
            numerical_columns = self.config.numerical_columns
            categorical_columns = self.config.categorical_columns
            
            Log.info(f"Numerical columns: {numerical_columns}")
            Log.info(f"Categorical columns: {categorical_columns}")
            
            # Pipeline for numerical features
            numerical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            
            # Pipeline for categorical features
            categorical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            
            # Combine pipelines into a column transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', numerical_pipeline, numerical_columns),
                    ('cat_pipeline', categorical_pipeline, categorical_columns)
                ],
                remainder='drop'  # Drop columns not specified
            )
            
            Log.info("Preprocessing pipeline created successfully")
            
            return preprocessor
            
        except Exception as e:
            Log.error(f"Exception occurred while creating preprocessor: {str(e)}")
            raise CustomException(str(e), sys) from e
            
    def preprocess_data(
        self,
        train_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame] = None,
        fit_transform: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray], ColumnTransformer]:
        """
        Preprocess training and test data using the preprocessing pipeline.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame (optional)
            fit_transform: Whether to fit and transform (True) or just transform (False)
            
        Returns:
            Tuple of (train_array, test_array, preprocessor)
            
        Raises:
            CustomException: If preprocessing fails
        """
        try:
            Log.info("Starting data preprocessing")
            
            # Get target column
            target_column = self.config.target_column
            
            # Drop specified columns if any
            if self.config.drop_columns:
                Log.info(f"Dropping columns: {self.config.drop_columns}")
                train_df = train_df.drop(columns=self.config.drop_columns, errors='ignore')
                if test_df is not None:
                    test_df = test_df.drop(columns=self.config.drop_columns, errors='ignore')
            
            # Split features and target
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            
            X_test = None
            y_test = None
            
            if test_df is not None:
                X_test = test_df.drop(columns=[target_column])
                y_test = test_df[target_column]
            
            # Get preprocessor
            preprocessor = self.get_preprocessor()
            
            # Apply preprocessing
            if fit_transform:
                Log.info("Fitting and transforming training data")
                X_train_processed = preprocessor.fit_transform(X_train)
            else:
                Log.info("Transforming training data using existing preprocessor")
                X_train_processed = preprocessor.transform(X_train)
            
            X_test_processed = None
            if X_test is not None:
                Log.info("Transforming test data")
                X_test_processed = preprocessor.transform(X_test)
            
            # Combine features and target
            train_arr = np.c_[X_train_processed, np.array(y_train)]
            
            test_arr = None
            if X_test_processed is not None and y_test is not None:
                test_arr = np.c_[X_test_processed, np.array(y_test)]
            
            Log.info(f"Preprocessed train data shape: {train_arr.shape}")
            if test_arr is not None:
                Log.info(f"Preprocessed test data shape: {test_arr.shape}")
            
            return train_arr, test_arr, preprocessor
            
        except Exception as e:
            Log.error(f"Exception occurred during data preprocessing: {str(e)}")
            raise CustomException(str(e), sys) from e
            
    def save_preprocessor(self, preprocessor: ColumnTransformer) -> str:
        """
        Save the preprocessor object to a file.
        
        Args:
            preprocessor: ColumnTransformer object to save
            
        Returns:
            Path to the saved preprocessor
            
        Raises:
            CustomException: If saving fails
        """
        try:
            Log.info(f"Saving preprocessor to {self.config.preprocessor_path}")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config.preprocessor_path), exist_ok=True)
            
            # Save preprocessor
            save_object(self.config.preprocessor_path, preprocessor)
            
            Log.info("Preprocessor saved successfully")
            
            return self.config.preprocessor_path
            
        except Exception as e:
            Log.error(f"Exception occurred while saving preprocessor: {str(e)}")
            raise CustomException(str(e), sys) from e
            
    def initiate_data_preprocessing(
        self,
        train_path: str,
        test_path: str
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Execute the complete data preprocessing pipeline.
        
        Args:
            train_path: Path to the training data
            test_path: Path to the test data
            
        Returns:
            Tuple of (train_array, test_array, preprocessor_path)
            
        Raises:
            CustomException: If any step in the pipeline fails
        """
        try:
            Log.info("Starting data preprocessing pipeline")
            
            # Load datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            Log.info(f"Loaded train data with shape: {train_df.shape}")
            Log.info(f"Loaded test data with shape: {test_df.shape}")
            
            # Preprocess data
            train_arr, test_arr, preprocessor = self.preprocess_data(train_df, test_df)
            
            # Save preprocessor
            preprocessor_path = self.save_preprocessor(preprocessor)
            
            Log.info("Data preprocessing completed successfully")
            
            return train_arr, test_arr, preprocessor_path
            
        except Exception as e:
            Log.error(f"Exception occurred during data preprocessing pipeline: {str(e)}")
            raise CustomException(str(e), sys) from e