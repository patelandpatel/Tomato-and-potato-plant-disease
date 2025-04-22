"""
Logging configuration for the Student Performance Predictor.
Provides a standardized logging mechanism across the application.
"""

import os
import logging
import datetime
from pathlib import Path
from typing import Optional, Dict, Any


def get_logger(
    logger_name: str = __name__,
    log_dir: str = "logs",
    log_level: int = logging.INFO,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """
    Configure and return a logger for the specified module.
    
    Args:
        logger_name: Name of the logger (typically __name__)
        log_dir: Directory to store log files
        log_level: Logging level (default: INFO)
        log_format: Custom log format (if None, uses default format)
        
    Returns:
        Configured logger
    """
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(exist_ok=True)
    
    # Generate log file name with timestamp
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_name = f"{current_time}.log"
    log_file_path = os.path.join(log_dir, log_file_name)
    
    # Set log format
    if log_format is None:
        log_format = "[ %(asctime)s ] %(levelname)s %(lineno)d %(name)s - %(message)s"
    
    # Configure logging
    logging.basicConfig(
        filename=log_file_path,
        level=log_level,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Set up console handler to display logs in console as well
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # Get logger and add console handler
    logger = logging.getLogger(logger_name)
    logger.addHandler(console_handler)
    
    return logger


# Create a default logger for the application
logger = get_logger()


class Log:
    """
    Log class that provides standardized logging methods.
    Can be used as a singleton across the application.
    """
    
    @staticmethod
    def info(message: str) -> None:
        """Log info level message."""
        logger.info(message)
    
    @staticmethod
    def debug(message: str) -> None:
        """Log debug level message."""
        logger.debug(message)
    
    @staticmethod
    def warning(message: str) -> None:
        """Log warning level message."""
        logger.warning(message)
    
    @staticmethod
    def error(message: str) -> None:
        """Log error level message."""
        logger.error(message)
    
    @staticmethod
    def critical(message: str) -> None:
        """Log critical level message."""
        logger.critical(message)
        
    @staticmethod
    def exception(message: str) -> None:
        """Log exception with traceback."""
        logger.exception(message)