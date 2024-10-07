# Import necessary modules
from src.cli import cli  # Import the command-line interface function
import sys  # For system-specific parameters and functions
import logging  # For logging functionality
from src import setup_logging  # Import the logging setup function from src


def main():
    """
    Main function to run the Flag Prediction application.

    This function sets up logging, runs the CLI application, and handles any
    unhandled exceptions that may occur during execution. It also ensures
    proper exit of the application.

    The function performs the following steps:
    1. Set up logging for the application
    2. Attempt to run the CLI application
    3. Log and print the start and completion of the application
    4. Catch and log any unhandled exceptions
    5. Ensure the application exits properly

    Returns:
        None
    """
    # Set up logging for the application
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Log and print the start of the application
        logger.info("Starting the Flag Prediction application")
        print("Starting the Flag Prediction application")

        # Run the CLI application
        cli()

        # Log and print successful completion
        logger.info("Flag Prediction application completed successfully")
        print("Flag Prediction application completed successfully")

    except Exception as e:
        # Log and print any unhandled exceptions
        logger.error(f"An unhandled error occurred: {str(e)}")
        print(f"An error occurred: {str(e)}", file=sys.stderr)

    finally:
        # Ensure that we always log and print the exit message
        logger.info("Exiting the application")
        print("Exiting the application")
        # Exit the application with a status code of 0 (successful execution)
        sys.exit(0)


if __name__ == "__main__":
    main()
