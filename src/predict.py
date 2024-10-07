import os
import torch
from PIL import Image
from torchvision import transforms
from typing import List, Tuple
from .config import (
    get_model_path,
    LABEL_ENCODER_FILE,
    IMAGE_SIZE,
    MULTI_LABEL_THRESHOLD,
    LOG_FILE,
    LOG_DIR,
)
import pickle
from .model import create_model
import logging

# Set up logging for this module
logger = logging.getLogger(__name__)


def load_model_and_encoder(
    multi_label: bool = False, model_type: str = "efficientnet_b0"
) -> Tuple[torch.nn.Module, object]:
    """
    Load the trained model and label encoder.

    Args:
        multi_label (bool): Whether the model is trained for multi-label classification.
        model_type (str): The type of model architecture to use.

    Returns:
        Tuple[torch.nn.Module, object]: The loaded model and label encoder.
        Returns (None, None) if loading fails.

    This function loads the pre-trained model and the label encoder used for
    converting between numeric labels and country names.
    """
    try:
        # Load the label encoder from the pickle file
        with open(LABEL_ENCODER_FILE, "rb") as f:
            label_encoder = pickle.load(f)

        # Get the number of classes from the label encoder
        num_classes = len(label_encoder.classes_)

        # Create the model with the specified architecture and number of classes
        model = create_model(num_classes, model_type, multi_label)

        # Get the path to the saved model weights
        model_path = get_model_path(model_type)

        # Load the model weights
        model.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
        )

        # Set the model to evaluation mode
        model.eval()

        logger.info("Model and label encoder loaded successfully.")
        return model, label_encoder
    except Exception as e:
        logger.error(f"Error loading model and encoder: {e}")
        return None, None


def predict_country(
    image_path: str, multi_label: bool = False, model_type: str = "efficientnet_b0"
) -> List[str]:
    """
    Predict the country (or countries) for a given flag image.

    Args:
        image_path (str): Path to the input image file.
        multi_label (bool): Whether to use multi-label prediction.
        model_type (str): The type of model architecture to use.

    Returns:
        List[str]: A list of predicted country names.

    This function loads an image, preprocesses it, runs it through the model,
    and returns the predicted country or countries.
    """
    try:
        # Attempt to open and convert the image to RGB
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise Exception(f"Invalid image file: {image_path}. Error: {str(e)}")

    # Load the model and label encoder
    model, label_encoder = load_model_and_encoder(multi_label, model_type)
    if model is None or label_encoder is None:
        return []

    # Define the image transformation pipeline
    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Open, convert, and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform the prediction
    with torch.no_grad():
        outputs = model(image)
        if multi_label:
            # For multi-label, use a threshold to determine positive classes
            predictions = (outputs > MULTI_LABEL_THRESHOLD).float()
            predicted_classes = predictions.nonzero().squeeze().tolist()
            if isinstance(predicted_classes, int):  # Handle single class case
                predicted_classes = [predicted_classes]
            countries = label_encoder.inverse_transform(predicted_classes)
        else:
            # For single-label, take the class with the highest probability
            _, preds = torch.max(outputs, 1)
            countries = [label_encoder.inverse_transform(preds.cpu().numpy())[0]]

    return countries


def main():
    """
    Main function to demonstrate the usage of the prediction functionality.

    This function sets up logging, attempts to predict the country for a test image,
    and logs the results.
    """
    # Set up basic logging configuration
    logging.basicConfig(
        filename=os.path.join(LOG_DIR, "flag_prediction.log"),
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.info("Starting prediction...")

    # Example usage
    test_image_path = "path_to_test_image.png"
    if os.path.exists(test_image_path):
        # Attempt to predict the country for the test image
        countries = predict_country(test_image_path, multi_label=True)
        if countries:
            result = f"This flag belongs to: {', '.join(countries)}"
            print(result)
            logger.info(result)
        else:
            print("Failed to predict the country.")
            logger.error("Failed to predict the country.")
    else:
        print(f"Image file {test_image_path} does not exist.")
        logger.error(f"Image file {test_image_path} does not exist.")


if __name__ == "__main__":
    main()
