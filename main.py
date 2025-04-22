"""
Main entry point for the Student Performance Predictor application.
Orchestrates the data processing, model training, and prediction pipeline.
"""

import os
import sys
import argparse
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional, Any

from src.utils.logger import Log
from src.utils.exception_handler import CustomException
from config.config import ConfigurationManager
from src.data.data_ingestion import DataIngestion
from src.data.data_validation import DataValidation
from src.features.preprocessing import DataPreprocessor
from src.models.model_development import ModelTrainer
from web_app.app import run_web_app


def run_training_pipeline() -> None:
    """
    Run the complete model training pipeline.
    
    Raises:
        CustomException: If any step in the pipeline fails
    """
    try:
        Log.info("Starting training pipeline")
        
        # Initialize configuration manager
        config_manager = ConfigurationManager()
        
        # Data Ingestion
        Log.info("Starting data ingestion")
        data_ingestion_config = config_manager.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        train_path, test_path = data_ingestion.initiate_data_ingestion()
        
        # Data Validation
        Log.info("Starting data validation")
        data_validation_config = config_manager.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        validation_status, validation_report_path = data_validation.validate_data(train_path, test_path)
        
        if not validation_status:
            Log.error("Data validation failed")
            raise CustomException("Data validation failed. Check validation report for details.")
            
        # Data Preprocessing
        Log.info("Starting data preprocessing")
        data_preprocessing_config = config_manager.get_data_preprocessing_config()
        data_preprocessor = DataPreprocessor(config=data_preprocessing_config)
        train_arr, test_arr, preprocessor_path = data_preprocessor.initiate_data_preprocessing(train_path, test_path)
        
        # Model Training
        Log.info("Starting model training")
        model_trainer_config = config_manager.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_path, evaluation_path, best_score = model_trainer.initiate_model_training(train_arr, test_arr)
        
        Log.info(f"Training pipeline completed successfully with best model score: {best_score}")
        
        return model_path, evaluation_path
        
    except Exception as e:
        Log.error(f"Exception occurred in training pipeline: {str(e)}")
        raise CustomException(str(e), sys) from e


def main():
    """
    Main function to parse arguments and run appropriate pipeline.
    """
    parser = argparse.ArgumentParser(description="Student Performance Predictor")
    
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run the training pipeline"
    )
    
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run the web application"
    )
    
    args = parser.parse_args()
    
    try:
        if args.train:
            Log.info("Running training pipeline")
            model_path, evaluation_path = run_training_pipeline()
            Log.info(f"Model saved at: {model_path}")
            Log.info(f"Evaluation saved at: {evaluation_path}")
            
        if args.serve or not (args.train or args.serve):  # Default to serving if no args provided
            Log.info("Starting web application")
            run_web_app()
            
    except Exception as e:
        Log.error(f"Exception occurred in main: {str(e)}")
        raise CustomException(str(e), sys) from e


if __name__ == "__main__":
    main()