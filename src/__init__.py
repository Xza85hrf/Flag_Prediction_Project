import os
import logging
from . import config

# List of functions and modules to be exported when using "from src import *"
__all__ = [
    "load_data",
    "get_data_loaders",
    "train_model",
    "create_model",
    "collect_data",
    "process_image",
    "process_images",
    "predict_country",
    "cli",
]


def setup_logging():
    """
    Set up logging configuration for the project.

    This function configures both file and console logging handlers with the following features:
    - Creates a log directory if it doesn't exist
    - Removes any existing handlers from the root logger
    - Sets up a file handler to write logs to a file
    - Sets up a console handler to display logs in the console
    - Configures log formatting and log levels based on the project's config
    - Performs a test write to ensure the log file is writable
    - Handles potential errors and provides fallback logging configuration

    The function uses settings from the config module, such as LOG_FILE and LOG_LEVEL.

    Raises:
        PermissionError: If there's no write permission for the log file
        Exception: For any other errors during logging setup

    Note:
        If an error occurs during setup, the function will fall back to a basic logging configuration.
    """
    try:
        # Ensure the log directory exists
        os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)

        # Remove all handlers associated with the root logger object
        # This prevents duplicate log entries if the function is called multiple times
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Set up file handler
        file_handler = logging.FileHandler(config.LOG_FILE, mode="a")
        file_handler.setLevel(config.LOG_LEVEL)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)

        # Set up console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(config.LOG_LEVEL)
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)

        # Get the root logger and add both handlers
        root_logger = logging.getLogger()
        root_logger.setLevel(config.LOG_LEVEL)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

        # Test write to log file
        # This helps catch permission issues early
        with open(config.LOG_FILE, "a") as f:
            f.write("Test write to log file\n")

        logging.info("Logging setup completed.")
    except PermissionError:
        # Handle the case where the script doesn't have permission to write to the log file
        print(f"Permission denied: Unable to write to log file at {config.LOG_FILE}")
    except Exception as e:
        # Catch any other exceptions that might occur during logging setup
        print(f"Error setting up logging: {str(e)}")
        # Fallback to default logging configuration if there's an error
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        logging.warning(
            "Using default logging configuration. Check config.py for missing attributes."
        )
