import os
import logging
import torch
from PIL import Image
from torchvision import transforms
from typing import List
from . import config
from .model import create_model
from .predict import load_model_and_encoder, predict_country

# Set up logging for the module
logger = logging.getLogger(__name__)


def main():
    """
    Main function to run the flag prediction process.

    This function sets up logging, loads a test image, and predicts the country
    or countries associated with the flag in the image.
    """
    # Configure logging settings
    logging.basicConfig(
        filename=config.LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.info("Starting prediction...")

    # Example usage of the prediction functionality
    test_image_path = "path_to_test_image.png"

    # Check if the test image file exists
    if os.path.exists(test_image_path):
        # Predict the country or countries for the flag in the image
        # The multi_label=True parameter allows for multiple country predictions
        countries = predict_country(test_image_path, multi_label=True)

        # Display the prediction results
        if countries:
            print(f"This flag belongs to: {', '.join(countries)}")
        else:
            print("Failed to predict the country.")
    else:
        print(f"Image file {test_image_path} does not exist.")


# Entry point of the script
if __name__ == "__main__":
    main()
