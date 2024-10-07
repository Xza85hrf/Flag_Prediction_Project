import os
import numpy as np
import logging
from PIL import Image, ImageEnhance, ImageFilter
from . import config
import pandas as pd
from tqdm import tqdm
import gc
import signal
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# Set up logging for this module
logger = logging.getLogger(__name__)


def process_image(image_path: str) -> Optional[np.ndarray]:
    """
    Process a single image by opening, converting to RGB, resizing, and normalizing.

    Args:
        image_path (str): Path to the image file.

    Returns:
        Optional[np.ndarray]: Processed image as a numpy array, or None if processing fails.

    Raises:
        AttributeError: If FLAG_IMAGES_DIR is not defined in the config module.
    """
    if not hasattr(config, "FLAG_IMAGES_DIR") or not config.FLAG_IMAGES_DIR:
        raise AttributeError(
            "The 'FLAG_IMAGES_DIR' attribute is missing in the config module."
        )
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img = img.resize(config.IMAGE_SIZE)
            img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
        return img_array
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        return None


def apply_augmentation(img: Image.Image) -> Image.Image:
    """
    Apply various augmentation techniques to the image.

    This function applies random rotations, color jitter, contrast enhancement,
    and brightness changes to the input image.

    Args:
        img (Image.Image): Input PIL Image object.

    Returns:
        Image.Image: Augmented PIL Image object.
    """
    from PIL import ImageOps, ImageEnhance
    import random

    # Rotate the image by a random angle between -20 and 20 degrees
    angle = random.uniform(-20, 20)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=True)

    # Apply random color jitter
    color_factor = random.uniform(0.8, 1.2)
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(color_factor)

    # Apply random contrast enhancement
    contrast_factor = random.uniform(0.8, 1.2)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)

    # Apply random brightness changes
    brightness_factor = random.uniform(0.8, 1.2)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)

    return img


def process_and_augment_batch(
    batch: List[Tuple[str, str, int]]
) -> List[Tuple[str, str]]:
    """
    Process and augment a batch of images.

    This function takes a batch of image paths, processes each image,
    applies augmentation, and saves the augmented images.

    Args:
        batch (List[Tuple[str, str, int]]): A list of tuples containing
            (input_path, base_name, duplicates) for each image.

    Returns:
        List[Tuple[str, str]]: A list of tuples containing
            (output_path, base_name) for each processed and augmented image.
    """
    results = []
    for input_path, base_name, duplicates in batch:
        try:
            original_img = Image.open(input_path).convert("RGB")
            original_img = original_img.resize(config.IMAGE_SIZE)

            for i in range(duplicates):
                augmented_img = apply_augmentation(original_img)
                output_filename = f"{base_name}_{i+1}.png"
                output_path = os.path.join(config.PROCESSED_DIR, output_filename)
                augmented_img.save(output_path)
                results.append((output_path, base_name))

            logger.info(f"Processed and augmented {base_name} {duplicates} times")
        except Exception as e:
            logger.error(f"Error processing image {input_path}: {e}")

    return results


def process_images(duplicate_times: Optional[int] = None) -> None:
    """
    Process collected images using batch processing and multiprocessing.

    This function reads images from the FLAG_IMAGES_DIR, processes them in batches,
    applies augmentation, and saves the results. It also creates a CSV file with
    the processed image data.

    Args:
        duplicate_times (Optional[int]): Number of times to duplicate each image
            during augmentation. If None, uses the value from config.AUGMENTATION_FACTOR.

    Raises:
        KeyboardInterrupt: If the process is interrupted by the user.
    """
    logger.info("Starting image processing...")
    if duplicate_times is None:
        duplicate_times = config.AUGMENTATION_FACTOR

    # Set up a signal handler to catch keyboard interrupts
    def signal_handler(signum, frame):
        raise KeyboardInterrupt("Process interrupted by user")

    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Get a list of all image files in the FLAG_IMAGES_DIR
        input_files = [
            f
            for f in os.listdir(config.FLAG_IMAGES_DIR)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        total_images = len(input_files)
        print(f"Processing and augmenting {total_images} images...")

        # Prepare batches for processing
        batch_size = 10  # Adjust this based on your system's capabilities
        batches = [
            [
                (
                    os.path.join(config.FLAG_IMAGES_DIR, f),
                    os.path.splitext(f)[0],
                    duplicate_times,
                )
                for f in input_files[i : i + batch_size]
            ]
            for i in range(0, len(input_files), batch_size)
        ]

        # Process batches using multiprocessing
        all_results = []
        with ProcessPoolExecutor(max_workers=config.NUM_WORKERS) as executor:
            futures = [
                executor.submit(process_and_augment_batch, batch) for batch in batches
            ]
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing batches"
            ):
                all_results.extend(future.result())

        print("Image augmentation completed. Creating CSV file...")
        create_processed_data_csv(all_results, config.DATA_PATH)

        print(f"Image processing completed. CSV file created at {config.DATA_PATH}")
        logger.info(
            f"Image processing completed. CSV file created at {config.DATA_PATH}"
        )
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Cleaning up...")
        logger.warning("Process interrupted by user. Cleaning up...")
    except Exception as e:
        print(f"An error occurred during image processing: {str(e)}")
        logger.error(f"An error occurred during image processing: {str(e)}")
    finally:
        print("Image processing function completed.")
        logger.info("Image processing function completed.")
        gc.collect()  # Perform garbage collection to free up memory


def create_processed_data_csv(data: List[Tuple[str, str]], output_csv: str) -> None:
    """
    Create a CSV file with processed image data.

    This function takes the processed image data and creates a CSV file
    containing the image paths and their corresponding labels.

    Args:
        data (List[Tuple[str, str]]): A list of tuples containing
            (image_path, label) for each processed image.
        output_csv (str): Path to the output CSV file.

    Raises:
        ValueError: If no valid processed images are found.
    """
    print(f"Creating CSV file at {output_csv}")
    logger.info(f"Creating CSV file at {output_csv}")
    try:
        if not data:
            error_msg = "No valid processed images found to create CSV file."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Create a DataFrame from the processed image data
        df = pd.DataFrame(data, columns=["image_path", "label"])

        # Ensure the directory for the output CSV exists
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

        # Save the DataFrame to a CSV file
        df.to_csv(output_csv, index=False)
        logger.info(
            f"Processed data saved to {output_csv} with {len(df)} rows and {len(df.columns)} columns"
        )
        print(
            f"Processed data saved to {output_csv} with {len(df)} rows and {len(df.columns)} columns"
        )
    except Exception as e:
        error_msg = f"Error creating CSV file: {str(e)}"
        logger.error(error_msg)
        print(error_msg)
        raise


# Export the public functions
__all__ = [
    "process_image",
    "process_images",
    "create_processed_data_csv",
]