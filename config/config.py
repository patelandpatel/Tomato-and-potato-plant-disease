"""
Configuration module for the Student Performance Predictor.
Contains path configurations and parameter handling.
"""

import os
import yaml
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion component."""
    raw_data_path: str
    train_data_path: str
    test_data_path: str
    validation_data_path: str
    raw_data_url: str = None  # Optional URL if data is fetched from remote source
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class DataValidationConfig:
    """Configuration for data validation component."""
    schema_path: str
    report_path: str
    required_columns: List[str]
    target_column: str


@dataclass
class DataPreprocessingConfig:
    """Configuration for data preprocessing component."""
    preprocessor_path: str
    numerical_columns: List[str]
    categorical_columns: List[str]
    target_column: str
    drop_columns: List[str] = None


@dataclass
class ModelTrainerConfig:
    """Configuration for model training component."""
    model_path: str
    trained_model_path: str
    train_data_path: str
    test_data_path: str
    evaluation_path: str
    target_column: str
    model_report_path: str
    r2_threshold: float = 0.6


@dataclass
class PredictionPipelineConfig:
    """Configuration for prediction pipeline."""
    model_path: str
    preprocessor_path: str


class ConfigurationManager:
    """
    Configuration manager to handle all configurations for the project.
    Reads from params.yaml and provides configurations for different components.
    """
    def __init__(self, config_filepath: str = "config/params.yaml"):
        """
        Initialize the configuration manager.
        
        Args:
            config_filepath: Path to the configuration file
        """
        self.config = self._read_config(config_filepath)
        
        # Create necessary directories
        self._create_directories([
            self.config["data_paths"]["processed_data_dir"],
            self.config["model_paths"]["models_dir"],
            self.config["logs"]["logs_dir"],
            self.config["reports"]["reports_dir"]
        ])

    def _read_config(self, config_filepath: str) -> Dict:
        """
        Read configuration from YAML file.
        
        Args:
            config_filepath: Path to the configuration file
            
        Returns:
            Dict containing configuration parameters
        """
        with open(config_filepath) as config_file:
            config = yaml.safe_load(config_file)
        return config

    def _create_directories(self, dir_paths: List[str]) -> None:
        """
        Create directories if they don't exist.
        
        Args:
            dir_paths: List of directory paths to create
        """
        for dir_path in dir_paths:
            os.makedirs(dir_path, exist_ok=True)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Get configuration for data ingestion.
        
        Returns:
            DataIngestionConfig object
        """
        data_config = self.config["data_ingestion"]
        
        return DataIngestionConfig(
            raw_data_path=os.path.join(self.config["data_paths"]["raw_data_dir"], "student_data.csv"),
            train_data_path=os.path.join(self.config["data_paths"]["processed_data_dir"], "train.csv"),
            test_data_path=os.path.join(self.config["data_paths"]["processed_data_dir"], "test.csv"),
            validation_data_path=os.path.join(self.config["data_paths"]["processed_data_dir"], "validation.csv"),
            raw_data_url=data_config.get("raw_data_url", None),
            test_size=data_config.get("test_size", 0.2),
            random_state=data_config.get("random_state", 42)
        )

    def get_data_validation_config(self) -> DataValidationConfig:
        """
        Get configuration for data validation.
        
        Returns:
            DataValidationConfig object
        """
        data_config = self.config["data_validation"]
        
        return DataValidationConfig(
            schema_path=os.path.join(self.config["data_paths"]["schema_dir"], "schema.json"),
            report_path=os.path.join(self.config["reports"]["reports_dir"], "data_validation_report.json"),
            required_columns=data_config.get("required_columns", []),
            target_column=data_config.get("target_column", "math_score")
        )

    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        """
        Get configuration for data preprocessing.
        
        Returns:
            DataPreprocessingConfig object
        """
        data_config = self.config["data_preprocessing"]
        
        return DataPreprocessingConfig(
            preprocessor_path=os.path.join(self.config["model_paths"]["models_dir"], "preprocessor.pkl"),
            numerical_columns=data_config.get("numerical_columns", []),
            categorical_columns=data_config.get("categorical_columns", []),
            target_column=data_config.get("target_column", "math_score"),
            drop_columns=data_config.get("drop_columns", None)
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """
        Get configuration for model training.
        
        Returns:
            ModelTrainerConfig object
        """
        model_config = self.config["model_trainer"]
        
        return ModelTrainerConfig(
            model_path=os.path.join(self.config["model_paths"]["models_dir"], "model.pkl"),
            trained_model_path=os.path.join(self.config["model_paths"]["models_dir"], "trained_model.pkl"),
            train_data_path=os.path.join(self.config["data_paths"]["processed_data_dir"], "train.csv"),
            test_data_path=os.path.join(self.config["data_paths"]["processed_data_dir"], "test.csv"),
            evaluation_path=os.path.join(self.config["reports"]["reports_dir"], "model_evaluation.json"),
            target_column=model_config.get("target_column", "math_score"),
            model_report_path=os.path.join(self.config["reports"]["reports_dir"], "model_report.json"),
            r2_threshold=model_config.get("r2_threshold", 0.6)
        )

    def get_prediction_pipeline_config(self) -> PredictionPipelineConfig:
        """
        Get configuration for prediction pipeline.
        
        Returns:
            PredictionPipelineConfig object
        """
        return PredictionPipelineConfig(
            model_path=os.path.join(self.config["model_paths"]["models_dir"], "model.pkl"),
            preprocessor_path=os.path.join(self.config["model_paths"]["models_dir"], "preprocessor.pkl")
        )