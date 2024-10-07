import pytest
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from PIL.Image import Resampling
from pytest_mock import mocker
from src.image_processing import (
    process_image,
    process_and_augment_batch,
    apply_augmentation,
    process_images,
    create_processed_data_csv,
)
from src import config
import logging
import warnings


@pytest.fixture(autouse=True)
def ignore_pillow_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="PIL")


@pytest.fixture
def sample_image(tmp_path):
    """
    Fixture to create a sample image for testing purposes.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.

    Returns:
        str: Path to the created sample image.
    """
    # Create a sample image for testing
    image = Image.new("RGB", (100, 100), color="red")
    image_path = tmp_path / "sample_image.png"
    image.save(image_path)
    return str(image_path)


@pytest.fixture
def corrupt_image(tmp_path):
    """
    Fixture to create a corrupt image file for negative testing.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.

    Returns:
        str: Path to the created corrupt image file.
    """
    # Create a corrupt image for negative testing
    corrupt_image_path = tmp_path / "corrupt_image.png"
    with open(corrupt_image_path, "wb") as f:
        f.write(b"Not a valid image file")
    return str(corrupt_image_path)


def test_process_image(sample_image):
    """
    Test the process_image function with a valid sample image.

    This test ensures that the function correctly processes the image and returns
    a numpy array with the expected shape, data type, and value range.
    """
    processed_image = process_image(sample_image)
    assert isinstance(processed_image, np.ndarray)
    assert processed_image.shape == (*config.IMAGE_SIZE, 3)
    assert processed_image.dtype == np.float64
    assert 0 <= processed_image.min() <= processed_image.max() <= 1


def test_process_image_negative(corrupt_image, caplog):
    """
    Test the process_image function with a corrupt image file.

    This negative test case ensures that the function handles errors gracefully
    when processing an invalid image file, returning None and logging an error.
    """
    with caplog.at_level(logging.ERROR):
        result = process_image(corrupt_image)
    assert result is None
    assert "Error processing image" in caplog.text


def test_apply_augmentation(sample_image):
    """
    Test the apply_augmentation function.

    This test verifies that the function applies augmentation to the input image,
    returning a new Image object that is different from the original.
    """
    original_img = Image.open(sample_image).convert("RGB")
    original_img = original_img.resize(
        config.IMAGE_SIZE, Resampling.BICUBIC
    )  # Update this line
    augmented_img = apply_augmentation(original_img)
    assert isinstance(augmented_img, Image.Image)
    assert augmented_img.mode == original_img.mode
    assert augmented_img != original_img


@pytest.mark.parametrize("batch_size", [1, 3, 5])
def test_process_and_augment_batch(tmp_path, sample_image, batch_size):
    """
    Test the process_and_augment_batch function with different batch sizes.

    This test ensures that the function correctly processes and augments a batch of images,
    creating the expected number of output files and returning the correct results.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
        sample_image: Fixture providing a sample image path.
        batch_size: Parameterized batch size for testing different scenarios.
    """
    output_dir = tmp_path / "processed"
    os.makedirs(output_dir, exist_ok=True)
    config.PROCESSED_DIR = str(output_dir)

    batch = [(sample_image, f"test_country_{i}", 2) for i in range(batch_size)]
    results = process_and_augment_batch(batch)

    assert len(results) == batch_size * 2
    for output_path, base_name in results:
        assert os.path.exists(output_path)
        assert base_name.startswith("test_country_")


def test_process_and_augment_batch_negative(tmp_path, corrupt_image, caplog):
    """
    Test the process_and_augment_batch function with a corrupt image.

    This negative test case ensures that the function handles errors gracefully
    when processing a batch containing an invalid image file.
    """
    output_dir = tmp_path / "processed"
    os.makedirs(output_dir, exist_ok=True)
    config.PROCESSED_DIR = str(output_dir)

    batch = [(corrupt_image, "corrupt_country", 2)]
    with caplog.at_level(logging.ERROR):
        results = process_and_augment_batch(batch)

    assert len(results) == 0
    assert "Error processing image" in caplog.text


@pytest.mark.parametrize("duplicate_times", [1, 3, 5])
def test_process_images(tmp_path, sample_image, duplicate_times, mocker):
    """
    Test the process_images function with different duplication factors.

    This test verifies that the function correctly processes all images in the input directory,
    applies augmentation, and creates a CSV file with the expected number of entries.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
        sample_image: Fixture providing a sample image path.
        duplicate_times: Parameterized duplication factor for testing different scenarios.
        mocker: Pytest fixture for mocking.
    """
    mocker.patch("src.config.FLAG_IMAGES_DIR", str(tmp_path))
    processed_dir = tmp_path / "processed"
    mocker.patch("src.config.PROCESSED_DIR", str(processed_dir))
    mocker.patch("src.config.DATA_PATH", str(tmp_path / "processed_data.csv"))
    mocker.patch("src.config.NUM_WORKERS", 1)

    os.makedirs(processed_dir, exist_ok=True)

    for i in range(3):
        Image.new("RGB", (100, 100), color="red").save(tmp_path / f"flag_{i}.png")

    Image.new("RGB", (100, 100), color="blue").save(tmp_path / "sample_image.png")

    process_images(duplicate_times)

    assert os.path.exists(tmp_path / "processed_data.csv")

    df = pd.read_csv(tmp_path / "processed_data.csv")
    assert len(df) == 4 * duplicate_times
    assert all(df.columns == ["image_path", "label"])

    unique_image_paths = df["image_path"].nunique()
    assert unique_image_paths == 4 * duplicate_times


def test_process_images_no_images(tmp_path, caplog, mocker):
    """
    Test the process_images function when no valid images are present.

    This test ensures that the function handles the case of an empty input directory
    correctly, logging an error and not creating a CSV file.
    """
    mocker.patch("src.config.FLAG_IMAGES_DIR", str(tmp_path))
    mocker.patch("src.config.PROCESSED_DIR", str(tmp_path / "processed"))
    mocker.patch("src.config.DATA_PATH", str(tmp_path / "processed_data.csv"))
    mocker.patch("src.config.NUM_WORKERS", 1)

    with caplog.at_level(logging.ERROR):
        process_images()

    assert "No valid processed images found to create CSV file" in caplog.text
    assert not os.path.exists(tmp_path / "processed_data.csv")


def test_create_processed_data_csv(tmp_path):
    """
    Test the create_processed_data_csv function with valid input data.

    This test verifies that the function correctly creates a CSV file with the
    provided image paths and labels.
    """
    data = [
        (str(tmp_path / "image1.png"), "country1"),
        (str(tmp_path / "image2.png"), "country2"),
    ]
    output_csv = str(tmp_path / "output.csv")

    create_processed_data_csv(data, output_csv)

    assert os.path.exists(output_csv)
    df = pd.read_csv(output_csv)
    assert len(df) == 2
    assert all(df.columns == ["image_path", "label"])


def test_create_processed_data_csv_negative(tmp_path, caplog):
    """
    Test the create_processed_data_csv function with empty input data.

    This negative test case ensures that the function raises a ValueError and
    logs an error when attempting to create a CSV file with no valid data.
    """
    data = []  # Empty data
    output_csv = str(tmp_path / "output.csv")

    with pytest.raises(ValueError):
        with caplog.at_level(logging.ERROR):
            create_processed_data_csv(data, output_csv)

    assert "No valid processed images found to create CSV file" in caplog.text
    assert not os.path.exists(output_csv)
