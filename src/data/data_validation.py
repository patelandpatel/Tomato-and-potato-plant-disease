"""
Data validation module for the Student Performance Predictor.
Ensures data quality and validates schema before processing.
"""

import os
import sys
import json
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

from src.utils.logger import Log
from src.utils.common import write_json
from src.utils.exception_handler import CustomException
from config.config import DataValidationConfig


class DataValidation:
    """
    Class to handle data validation operations.
    Validates data schema, checks for missing values, and ensures data quality.
    """
    
    def __init__(self, config: DataValidationConfig):
        """
        Initialize the DataValidation with configuration.
        
        Args:
            config: Configuration for data validation
        """
        self.config = config
        self.validation_report = {
            "schema_validation": False,
            "missing_values": {},
            "data_types": {},
            "value_counts": {},
            "data_drift": {},
            "validation_status": False
        }
        
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """
        Validate that the DataFrame contains all required columns.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if schema is valid, False otherwise
        """
        try:
            Log.info("Validating data schema")
            
            # Check if all required columns are present
            available_columns = set(df.columns)
            required_columns = set(self.config.required_columns)
            
            missing_columns = required_columns - available_columns
            
            if missing_columns:
                Log.warning(f"Schema validation failed. Missing columns: {missing_columns}")
                self.validation_report["schema_validation"] = False
                return False
                
            Log.info("Schema validation successful")
            self.validation_report["schema_validation"] = True
            return True
            
        except Exception as e:
            Log.error(f"Exception occurred during schema validation: {str(e)}")
            raise CustomException(str(e), sys) from e
            
    def check_missing_values(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Check for missing values in each column.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dictionary of column names to percentage of missing values
        """
        try:
            Log.info("Checking for missing values")
            
            # Calculate percentage of missing values for each column
            missing_values = (df.isnull().sum() / len(df)) * 100
            missing_values_dict = missing_values.to_dict()
            
            # Filter out columns with no missing values
            missing_values_dict = {k: v for k, v in missing_values_dict.items() if v > 0}
            
            if missing_values_dict:
                Log.warning(f"Missing values found: {missing_values_dict}")
            else:
                Log.info("No missing values found")
                
            self.validation_report["missing_values"] = missing_values_dict
            return missing_values_dict
            
        except Exception as e:
            Log.error(f"Exception occurred during missing values check: {str(e)}")
            raise CustomException(str(e), sys) from e
            
    def check_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Check data types of each column.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dictionary of column names to data types
        """
        try:
            Log.info("Checking data types")
            
            # Get data types of each column
            data_types = df.dtypes.astype(str).to_dict()
            
            Log.info(f"Data types: {data_types}")
            self.validation_report["data_types"] = data_types
            return data_types
            
        except Exception as e:
            Log.error(f"Exception occurred during data types check: {str(e)}")
            raise CustomException(str(e), sys) from e
            
    def check_value_counts(self, df: pd.DataFrame, categorical_columns: List[str]) -> Dict[str, Dict]:
        """
        Check value counts for categorical columns.
        
        Args:
            df: DataFrame to check
            categorical_columns: List of categorical column names
            
        Returns:
            Dictionary of column names to value counts
        """
        try:
            Log.info("Checking value counts for categorical columns")
            
            value_counts = {}
            
            for column in categorical_columns:
                if column in df.columns:
                    # Get value counts for the column
                    counts = df[column].value_counts().to_dict()
                    value_counts[column] = counts
                    
            Log.info(f"Value counts checked for {len(value_counts)} columns")
            self.validation_report["value_counts"] = value_counts
            return value_counts
            
        except Exception as e:
            Log.error(f"Exception occurred during value counts check: {str(e)}")
            raise CustomException(str(e), sys) from e
            
    def check_data_drift(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Check for data drift between train and test datasets.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            
        Returns:
            Dictionary of drift metrics for each column
        """
        try:
            from scipy.stats import ks_2samp
            
            Log.info("Checking for data drift between train and test sets")
            
            drift_report = {}
            
            # Check for columns present in both datasets
            common_columns = set(train_df.columns).intersection(set(test_df.columns))
            
            for column in common_columns:
                # Skip target column
                if column == self.config.target_column:
                    continue
                    
                # Perform KS test for numerical columns
                if train_df[column].dtype in ['int64', 'float64']:
                    statistic, p_value = ks_2samp(train_df[column], test_df[column])
                    drift_detected = p_value < 0.05
                    
                    drift_report[column] = {
                        "statistic": float(statistic),
                        "p_value": float(p_value),
                        "drift_detected": drift_detected
                    }
            
            Log.info(f"Data drift checked for {len(drift_report)} columns")
            self.validation_report["data_drift"] = drift_report
            return drift_report
            
        except Exception as e:
            Log.error(f"Exception occurred during data drift check: {str(e)}")
            raise CustomException(str(e), sys) from e
            
    def save_validation_report(self) -> str:
        """
        Save the validation report to a JSON file.
        
        Returns:
            Path to the saved validation report
        """
        try:
            Log.info(f"Saving validation report to {self.config.report_path}")
            
            # Create directory for report if it doesn't exist
            os.makedirs(os.path.dirname(self.config.report_path), exist_ok=True)
            
            # Convert numpy values to Python types for JSON serialization
            import numpy as np
            
            def convert_to_serializable(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            # Save report to JSON file
            write_json(self.config.report_path, self.validation_report)
            
            Log.info("Validation report saved successfully")
            return self.config.report_path
            
        except Exception as e:
            Log.error(f"Exception occurred while saving validation report: {str(e)}")
            raise CustomException(str(e), sys) from e
            
    def validate_data(self, train_path: str, test_path: str) -> Tuple[bool, str]:
        """
        Execute the complete data validation pipeline.
        
        Args:
            train_path: Path to the training data
            test_path: Path to the test data
            
        Returns:
            Tuple of (validation_status, report_path)
            
        Raises:
            CustomException: If any step in the pipeline fails
        """
        try:
            Log.info("Starting data validation process")
            
            # Load datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            Log.info(f"Loaded train data with shape: {train_df.shape}")
            Log.info(f"Loaded test data with shape: {test_df.shape}")
            
            # Validate schema
            schema_valid = self.validate_schema(train_df)
            
            # Only proceed with other validations if schema is valid
            if schema_valid:
                # Check for missing values
                self.check_missing_values(train_df)
                self.check_missing_values(test_df)
                
                # Check data types
                self.check_data_types(train_df)
                
                # Check value counts for categorical columns
                categorical_columns = [col for col in train_df.columns if train_df[col].dtype == 'object']
                self.check_value_counts(train_df, categorical_columns)
                
                # Check for data drift
                self.check_data_drift(train_df, test_df)
                
                # Set overall validation status
                self.validation_report["validation_status"] = True
                
            # Save validation report
            report_path = self.save_validation_report()
            
            Log.info(f"Data validation completed with status: {self.validation_report['validation_status']}")
            
            return self.validation_report["validation_status"], report_path
            
        except Exception as e:
            Log.error(f"Exception occurred during data validation: {str(e)}")
            raise CustomException(str(e), sys) from e