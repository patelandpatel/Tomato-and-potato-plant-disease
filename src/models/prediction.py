"""
Prediction module for the Student Performance Predictor.
Handles making predictions using the trained model.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union

from src.utils.logger import Log
from src.utils.common import load_object
from src.utils.exception_handler import CustomException
from config.config import PredictionPipelineConfig


class PredictionPipeline:
    """
    Class to handle prediction operations.
    Loads the model and preprocessor, and makes predictions on new data.
    """
    
    def __init__(self, config: PredictionPipelineConfig):
        """
        Initialize the PredictionPipeline with configuration.
        
        Args:
            config: Configuration for prediction pipeline
        """
        self.config = config
        self.model = None
        self.preprocessor = None
        
    def load_models(self) -> None:
        """
        Load the trained model and preprocessor.
        
        Raises:
            CustomException: If loading fails
        """
        try:
            Log.info("Loading model and preprocessor")
            
            # Load model
            self.model = load_object(self.config.model_path)
            
            # Load preprocessor
            self.preprocessor = load_object(self.config.preprocessor_path)
            
            Log.info("Model and preprocessor loaded successfully")
            
        except Exception as e:
            Log.error(f"Exception occurred while loading model and preprocessor: {str(e)}")
            raise CustomException(str(e), sys) from e
            
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the loaded model.
        
        Args:
            features: DataFrame containing input features
            
        Returns:
            NumPy array of predictions
            
        Raises:
            CustomException: If prediction fails
        """
        try:
            Log.info("Making predictions")
            
            # Load models if not already loaded
            if self.model is None or self.preprocessor is None:
                self.load_models()
                
            # Preprocess features
            features_processed = self.preprocessor.transform(features)
            
            # Make predictions
            predictions = self.model.predict(features_processed)
            
            Log.info(f"Predictions made successfully: {predictions}")
            
            return predictions
            
        except Exception as e:
            Log.error(f"Exception occurred during prediction: {str(e)}")
            raise CustomException(str(e), sys) from e


class CustomData:
    """
    Class to convert input data to a DataFrame for prediction.
    """
    
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: float,
        writing_score: float
    ):
        """
        Initialize the CustomData with input features.
        
        Args:
            gender: Gender of the student
            race_ethnicity: Race/ethnicity group
            parental_level_of_education: Highest education level of parents
            lunch: Type of lunch (standard or free/reduced)
            test_preparation_course: Test preparation course completion status
            reading_score: Reading test score
            writing_score: Writing test score
        """
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
        
    def get_data_as_dataframe(self) -> pd.DataFrame:
        """
        Convert input data to a pandas DataFrame.
        
        Returns:
            DataFrame containing input features
            
        Raises:
            CustomException: If conversion fails
        """
        try:
            Log.info("Converting input data to DataFrame")
            
            # Create a dictionary of input data
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }
            
            # Convert to DataFrame
            return pd.DataFrame(custom_data_input_dict)
            
        except Exception as e:
            Log.error(f"Exception occurred while converting input data to DataFrame: {str(e)}")
            raise CustomException(str(e), sys) from e