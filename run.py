from src.cli import cli
import sys
import logging
from src import setup_logging


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    try:
        logger.info("Starting the Flag Prediction application")
        print("Starting the Flag Prediction application")
        cli()
        logger.info("Flag Prediction application completed successfully")
        print("Flag Prediction application completed successfully")
    except Exception as e:
        logger.error(f"An unhandled error occurred: {str(e)}")
        print(f"An error occurred: {str(e)}", file=sys.stderr)
    finally:
        logger.info("Exiting the application")
        print("Exiting the application")
        sys.exit(0)


if __name__ == "__main__":
    main()
