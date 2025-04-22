"""
Data ingestion module for the Student Performance Predictor.
Handles loading data from sources and splitting into train/test sets.
"""

import os
import sys
import pandas as pd
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split

from src.utils.logger import Log
from src.utils.common import save_data
from src.utils.exception_handler import CustomException
from config.config import DataIngestionConfig


class DataIngestion:
    """
    Class to handle data ingestion operations.
    Responsible for loading data and splitting into train/test sets.
    """
    
    def __init__(self, config: DataIngestionConfig):
        """
        Initialize the DataIngestion with configuration.
        
        Args:
            config: Configuration for data ingestion
        """
        self.config = config
        
    def download_data(self) -> str:
        """
        Download data from a remote source if provided.
        
        Returns:
            Path to the downloaded data file
            
        Raises:
            CustomException: If download fails
        """
        try:
            # Create directory for raw data if it doesn't exist
            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
            
            # If URL is provided, download data
            if self.config.raw_data_url:
                import requests
                
                Log.info(f"Downloading data from {self.config.raw_data_url}")
                response = requests.get(self.config.raw_data_url)
                response.raise_for_status()  # Raise exception for HTTP errors
                
                # Save downloaded data to raw data path
                with open(self.config.raw_data_path, 'wb') as f:
                    f.write(response.content)
                    
                Log.info(f"Data downloaded successfully to {self.config.raw_data_path}")
            else:
                Log.info(f"No download URL provided. Using existing data at {self.config.raw_data_path}")
                
            return self.config.raw_data_path
            
        except Exception as e:
            Log.error(f"Exception occurred during data download: {str(e)}")
            raise CustomException(str(e), sys) from e
            
    def load_data(self) -> pd.DataFrame:
        """
        Load data from local raw data path.
        
        Returns:
            Pandas DataFrame containing the loaded data
            
        Raises:
            CustomException: If data loading fails
        """
        try:
            Log.info(f"Loading data from {self.config.raw_data_path}")
            
            # Check if file exists
            if not os.path.exists(self.config.raw_data_path):
                Log.error(f"Data file not found at {self.config.raw_data_path}")
                raise FileNotFoundError(f"Data file not found at {self.config.raw_data_path}")
                
            # Load data
            df = pd.read_csv(self.config.raw_data_path)
            Log.info(f"Data loaded successfully with shape: {df.shape}")
            
            return df
            
        except Exception as e:
            Log.error(f"Exception occurred during data loading: {str(e)}")
            raise CustomException(str(e), sys) from e
            
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and test sets.
        
        Args:
            df: DataFrame to split
            
        Returns:
            Tuple of (train_df, test_df)
            
        Raises:
            CustomException: If data splitting fails
        """
        try:
            Log.info("Splitting data into train and test sets")
            
            # Create directories for processed data if they don't exist
            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.config.test_data_path), exist_ok=True)
            
            # Split data
            train_df, test_df = train_test_split(
                df,
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )
            
            Log.info(f"Train set shape: {train_df.shape}, Test set shape: {test_df.shape}")
            
            # Save the split datasets
            save_data(train_df, self.config.train_data_path)
            save_data(test_df, self.config.test_data_path)
            
            Log.info(f"Train data saved to: {self.config.train_data_path}")
            Log.info(f"Test data saved to: {self.config.test_data_path}")
            
            return train_df, test_df
            
        except Exception as e:
            Log.error(f"Exception occurred during data splitting: {str(e)}")
            raise CustomException(str(e), sys) from e
            
    def initiate_data_ingestion(self) -> Tuple[str, str]:
        """
        Execute the complete data ingestion pipeline.
        
        Returns:
            Tuple of (train_data_path, test_data_path)
            
        Raises:
            CustomException: If any step in the pipeline fails
        """
        try:
            Log.info("Starting data ingestion process")
            
            # Download data if URL is provided
            if self.config.raw_data_url:
                self.download_data()
                
            # Load data
            df = self.load_data()
            
            # Split data
            self.split_data(df)
            
            Log.info("Data ingestion completed successfully")
            
            return self.config.train_data_path, self.config.test_data_path
            
        except Exception as e:
            Log.error(f"Exception occurred during data ingestion: {str(e)}")
            raise CustomException(str(e), sys) from e