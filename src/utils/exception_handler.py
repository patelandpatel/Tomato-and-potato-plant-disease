"""
Custom exception handler for the Student Performance Predictor.
Provides detailed error messages with file names and line numbers.
"""

import sys
from typing import Tuple


class CustomException(Exception):
    """
    Custom exception class that provides detailed error information.
    Includes error message, file name, and line number where the error occurred.
    """
    
    def __init__(self, error_message: str, error_detail: sys) -> None:
        """
        Initialize the CustomException with error details.
        
        Args:
            error_message: The error message
            error_detail: System error details
        """
        super().__init__(error_message)
        self.error_message = self._get_detailed_error_message(error_message, error_detail)
    
    def _get_detailed_error_message(self, error_message: str, error_detail: sys) -> str:
        """
        Get a detailed error message with file name and line number.
        
        Args:
            error_message: The original error message
            error_detail: System error details
            
        Returns:
            Detailed error message
        """
        _, _, exc_tb = error_detail.exc_info()
        
        # Get file name and line number where exception occurred
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        
        # Format the detailed error message
        return (f"Error occurred in Python script: [{file_name}] "
                f"at line number: [{line_number}] "
                f"with error message: [{error_message}]")
    
    def __str__(self) -> str:
        """
        String representation of the CustomException.
        
        Returns:
            The detailed error message
        """
        return self.error_message


def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Get detailed error message from an exception.
    Utility function that can be used without creating a CustomException.
    
    Args:
        error: The exception
        error_detail: System error details
        
    Returns:
        Detailed error message
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    
    return (f"Error occurred in Python script: [{file_name}] "
            f"at line number: [{line_number}] "
            f"with error message: [{str(error)}]")