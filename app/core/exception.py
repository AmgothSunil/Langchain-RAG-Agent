import sys
import logging

def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Extract detailed error information including file name, line number, and the error message.

    Args:
        error (Exception): The exception that occurred.
        error_detail (sys): The sys module to access traceback details.

    Returns:
        str: Formatted error message string.
    """
    # Get traceback details safely
    exc_type, exc_value, exc_tb = error_detail.exc_info()

    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        error_message = (
            f"Error occurred in file: [{file_name}] "
            f"at line number [{line_number}] "
            f"with error: {str(error)}"
        )
    else:
        error_message = f"Error occurred: {str(error)} (no traceback available)"

    logging.error(error_message)
    return error_message


class AppException(Exception):
    """
    Custom application-level exception for standardized error handling.
    """

    def __init__(self, error_message: str, error_detail: sys):
        """
        Initialize the AppException with a detailed error message.

        Args:
            error_message (str): A string describing the error.
            error_detail (sys): The sys module to access traceback details.
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self) -> str:
        """Return a clean, human-readable string representation."""
        return self.error_message

    def __repr__(self) -> str:
        """Return an unambiguous developer-friendly representation."""
        return f"{self.__class__.__name__}({self.error_message})"