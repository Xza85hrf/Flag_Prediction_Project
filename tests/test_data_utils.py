import pytest
from pytest_mock import mocker
import torch
import numpy as np
import random
from src.data_utils import FlagDataset, load_data, get_data_loaders
from src import config
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
import logging


@pytest.fixture
def sample_data(tmp_path):
    """
    Fixture to create sample data for testing.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.

    Returns:
        str: Path to the created sample data CSV file.
    """
    # Create a sample DataFrame with image paths and labels
    data = pd.DataFrame(
        {
            "image_path": [str(tmp_path / f"flag_{i}.png") for i in range(10)],
            "label": ["country_" + str(i % 3) for i in range(10)],
        }
    )
    data_path = tmp_path / "processed_data.csv"
    data.to_csv(data_path, index=False)

    # Create dummy image files
    for i in range(10):
        image = Image.new("RGB", (224, 224), color="red")
        image.save(tmp_path / f"flag_{i}.png")

    return str(data_path)


@pytest.fixture
def mock_config(mocker, sample_data):
    """
    Fixture to mock configuration settings for testing.

    Args:
        mocker: Pytest mocker object.
        sample_data: Path to sample data from the sample_data fixture.
    """
    mocker.patch("src.config.PROCESSED_DIR", os.path.dirname(sample_data))
    mocker.patch("src.config.DATA_PATH", sample_data)
    mocker.patch("src.config.BATCH_SIZE", 2)
    mocker.patch("src.config.TEST_SIZE", 0.2)
    mocker.patch("src.config.RANDOM_STATE", 42)
    mocker.patch("src.config.PIN_MEMORY", True)


def test_load_data(sample_data, mock_config):
    """
    Test the load_data function to ensure it correctly loads image paths and labels.
    """
    image_paths, labels = load_data()
    assert len(image_paths) > 0
    assert len(labels) > 0
    assert all(os.path.exists(path) for path in image_paths)


def test_flag_dataset(sample_data):
    """
    Test the FlagDataset class to ensure it correctly handles data and transformations.
    """
    data = pd.read_csv(sample_data)
    labels = pd.factorize(data["label"])[0]
    dataset = FlagDataset(data["image_path"].tolist(), labels)

    assert len(dataset) == 10
    image, label = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, *config.IMAGE_SIZE)
    assert isinstance(label, torch.Tensor)

    # Test with custom transform
    custom_transform = transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor()]
    )
    dataset_with_transform = FlagDataset(
        data["image_path"].tolist(), labels, transform=custom_transform
    )
    image, _ = dataset_with_transform[0]
    assert image.shape == (3, 32, 32)

    # Test multi-label
    multi_label_dataset = FlagDataset(
        data["image_path"].tolist(), np.random.rand(10, 3), multi_label=True
    )
    _, label = multi_label_dataset[0]
    assert label.dtype == torch.float32


def test_get_data_loaders(sample_data, mock_config, caplog):
    """
    Test the get_data_loaders function to ensure it correctly creates data loaders.

    Args:
        caplog: Pytest fixture for capturing log output.
    """
    caplog.set_level(logging.INFO)

    train_loader, val_loader, num_classes, label_encoder = get_data_loaders(
        multi_label=False, num_workers=0
    )

    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert isinstance(val_loader, torch.utils.data.DataLoader)
    assert num_classes > 0
    assert hasattr(label_encoder, "classes_")

    # Check batch size
    assert train_loader.batch_size == config.BATCH_SIZE
    assert val_loader.batch_size == config.BATCH_SIZE

    # Check data split
    total_samples = len(train_loader.dataset) + len(val_loader.dataset)
    assert len(val_loader.dataset) / total_samples == pytest.approx(
        config.TEST_SIZE, abs=0.05
    )

    # Test multi-label
    train_loader, val_loader, num_classes, label_encoder = get_data_loaders(
        multi_label=True, num_workers=0
    )
    assert num_classes == 1
    assert hasattr(label_encoder, "classes_")


def test_get_data_loaders_invalid_data(mock_config, tmp_path, caplog, mocker):
    """
    Test get_data_loaders function with invalid data to ensure proper error handling.
    """
    caplog.set_level(logging.ERROR)
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    mocker.patch("src.config.PROCESSED_DIR", str(empty_dir))

    with pytest.raises(ValueError, match="With n_samples=0"):
        get_data_loaders(multi_label=False, num_workers=0)


def test_flag_dataset_invalid_image(sample_data):
    """
    Test FlagDataset with an invalid image path to ensure proper error handling.
    """
    data = pd.read_csv(sample_data)
    data.loc[0, "image_path"] = "invalid_path.png"
    labels = pd.factorize(data["label"])[0]
    dataset = FlagDataset(data["image_path"].tolist(), labels)

    with pytest.raises(FileNotFoundError):
        _ = dataset[0]


def test_load_data_empty_directory(tmp_path, mocker):
    """
    Test load_data function with an empty directory to ensure proper handling.
    """
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    mocker.patch("src.config.PROCESSED_DIR", str(empty_dir))

    image_paths, labels = load_data()
    assert len(image_paths) == 0
    assert len(labels) == 0


def test_get_data_loaders_single_class(sample_data, mock_config):
    """
    Test get_data_loaders function with a single class to ensure proper handling.
    """
    data = pd.read_csv(sample_data)
    data["label"] = "single_country"
    data.to_csv(sample_data, index=False)

    train_loader, val_loader, num_classes, label_encoder = get_data_loaders(
        multi_label=False, num_workers=0
    )

    assert num_classes == 1
    assert len(label_encoder.classes_) == 1


def test_get_data_loaders_imbalanced_data(sample_data, mock_config, mocker):
    """
    Test get_data_loaders function with imbalanced data to ensure proper handling.
    """
    data = pd.read_csv(sample_data)
    # Create imbalanced data
    data["label"] = ["country_1"] * 8 + ["country_2"] * 2
    data.to_csv(sample_data, index=False)

    for i in range(10):
        image = Image.new("RGB", (224, 224), color="red")
        label = "country_1" if i < 8 else "country_2"
        image.save(os.path.join(os.path.dirname(sample_data), f"{label}_flag_{i}.png"))

    mocker.patch("src.config.PROCESSED_DIR", os.path.dirname(sample_data))

    def mock_load_data():
        image_paths = [
            os.path.join(os.path.dirname(sample_data), f"{label}_flag_{i}.png")
            for i, label in enumerate(data["label"])
        ]
        return image_paths, data["label"].tolist()

    mocker.patch("src.data_utils.load_data", side_effect=mock_load_data)

    train_loader, val_loader, num_classes, label_encoder = get_data_loaders(
        multi_label=False, num_workers=0
    )

    assert num_classes == 2

    train_labels = [y.item() for _, y in train_loader.dataset]
    val_labels = [y.item() for _, y in val_loader.dataset]

    train_class_counts = np.bincount(train_labels)
    val_class_counts = np.bincount(val_labels)

    assert train_class_counts[0] > train_class_counts[1]

    if len(val_class_counts) > 1:
        assert val_class_counts[0] >= val_class_counts[1]
    else:
        assert val_class_counts[0] == len(val_labels)
        assert label_encoder.inverse_transform([0])[0] == "country_1"

    assert len(train_labels) + len(val_labels) == 10
    assert sum(train_class_counts) + sum(val_class_counts) == 10


def test_flag_dataset_multi_label_threshold(sample_data):
    """
    Test FlagDataset with multi-label data to ensure proper thresholding.
    """
    data = pd.read_csv(sample_data)
    multi_label_data = np.random.rand(10, 3)
    dataset = FlagDataset(
        data["image_path"].tolist(), multi_label_data, multi_label=True
    )

    image, label = dataset[0]
    assert label.shape == (3,)
    assert torch.all((label >= 0) & (label <= 1))


def test_get_data_loaders_reproducibility(sample_data, mock_config, mocker):
    """
    Test get_data_loaders function for reproducibility with fixed random seeds.
    """

    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Mock the transforms to remove randomness
    mock_transform = transforms.Compose(
        [
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    mocker.patch("torchvision.transforms.Compose", return_value=mock_transform)

    # Set seed and create first data loader
    set_seed()
    train_loader1, val_loader1, _, _ = get_data_loaders(
        multi_label=False, num_workers=0
    )

    # Set seed again and create second data loader
    set_seed()
    train_loader2, val_loader2, _, _ = get_data_loaders(
        multi_label=False, num_workers=0
    )

    # Check if the data splits are the same
    for (img1, label1), (img2, label2) in zip(train_loader1, train_loader2):
        assert torch.allclose(img1, img2)
        assert torch.all(label1.eq(label2))


def test_flag_dataset_augmentation(sample_data):
    """
    Test FlagDataset with data augmentation to ensure it produces different results.
    """
    data = pd.read_csv(sample_data)
    labels = pd.factorize(data["label"])[0]

    augmentation_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ]
    )

    dataset = FlagDataset(
        data["image_path"].tolist(), labels, transform=augmentation_transform
    )

    # Check if augmentation produces different results
    image1, _ = dataset[0]
    image2, _ = dataset[0]
    assert not torch.all(image1.eq(image2))


def test_get_data_loaders_pin_memory(sample_data, mock_config, mocker):
    """
    Test get_data_loaders function with different pin_memory settings.
    """
    mocker.patch("src.config.PIN_MEMORY", True)
    train_loader, val_loader, _, _ = get_data_loaders(multi_label=False, num_workers=0)

    assert train_loader.pin_memory
    assert val_loader.pin_memory

    mocker.patch("src.config.PIN_MEMORY", False)
    train_loader, val_loader, _, _ = get_data_loaders(multi_label=False, num_workers=0)

    assert not train_loader.pin_memory
    assert not val_loader.pin_memory
