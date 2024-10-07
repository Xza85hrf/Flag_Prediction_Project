import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from torchvision import transforms
from typing import Tuple, List
from . import config


# FlagDataset class: A custom PyTorch Dataset for handling flag images and labels
class FlagDataset(Dataset):
    """
    A custom Dataset class for loading and preprocessing flag images and their corresponding labels.

    This class is designed to work with both single-label and multi-label classification tasks.
    It loads images from file paths, applies transformations, and returns tensor representations
    of images along with their labels.

    Attributes:
        image_paths (List[str]): List of file paths to the flag images.
        labels (np.ndarray): Array of labels corresponding to the images.
        transform (callable, optional): A function/transform to apply to the image.
        multi_label (bool): Whether the task is multi-label classification.
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: np.ndarray,
        transform=None,
        multi_label: bool = False,
    ):
        """
        Initialize the FlagDataset.

        Args:
            image_paths (List[str]): List of file paths to the flag images.
            labels (np.ndarray): Array of labels corresponding to the images.
            transform (callable, optional): A function/transform to apply to the image.
            multi_label (bool): Whether the task is multi-label classification.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform if transform else transforms.ToTensor()
        self.multi_label = multi_label

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fetch and preprocess a single sample from the dataset.

        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            tuple: A tuple containing the preprocessed image tensor and its corresponding label tensor.
        """
        # Load the image from file and convert to RGB
        image = Image.open(self.image_paths[idx]).convert("RGB")
        # Resize the image to the specified size in config
        image = image.resize(config.IMAGE_SIZE)
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        # Convert label to tensor, using appropriate dtype based on multi-label flag
        label = torch.tensor(
            self.labels[idx], dtype=torch.float32 if self.multi_label else torch.long
        )
        return image, label


def load_data() -> Tuple[List[str], List[str]]:
    """
    Load image file paths and their corresponding labels from the processed data directory.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing two lists:
            1. List of image file paths
            2. List of corresponding labels
    """
    data_dir = config.PROCESSED_DIR
    # Get all image files from the data directory
    image_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    # Extract labels from file names (assuming format: label_*.ext)
    labels = [
        os.path.splitext(os.path.basename(f))[0].rsplit("_", 1)[0] for f in image_files
    ]
    return image_files, labels


def get_data_loaders(
    multi_label: bool = False,
    num_workers: int = config.NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, int, object]:
    """
    Prepare and return data loaders for training and validation sets.

    This function handles data loading, preprocessing, and splitting into train and validation sets.
    It supports both single-label and multi-label classification tasks.

    Args:
        multi_label (bool): Whether the task is multi-label classification.
        num_workers (int): Number of worker processes for data loading.

    Returns:
        Tuple[DataLoader, DataLoader, int, object]: A tuple containing:
            1. Training data loader
            2. Validation data loader
            3. Number of classes
            4. Label encoder object (LabelEncoder or MultiLabelBinarizer)
    """
    # Load image paths and labels
    image_paths, labels = load_data()

    if multi_label:
        # Multi-label binarization for multi-label classification
        label_encoder = MultiLabelBinarizer()
        encoded_labels = label_encoder.fit_transform(
            [label.split(",") for label in labels]
        )
    else:
        # Label encoding for single-label classification
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        # Note: One-hot encoding is not used for CrossEntropyLoss

    # Get the number of unique classes
    num_classes = len(label_encoder.classes_)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        image_paths,
        encoded_labels,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=None if multi_label else encoded_labels,
    )

    # Define data augmentation and normalization for training set
    train_transform = transforms.Compose(
        [
            transforms.Resize(config.IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Define normalization for validation set (no augmentation)
    val_transform = transforms.Compose(
        [
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create dataset objects
    train_dataset = FlagDataset(
        X_train, y_train, transform=train_transform, multi_label=multi_label
    )
    val_dataset = FlagDataset(
        X_val, y_val, transform=val_transform, multi_label=multi_label
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
    )

    return train_loader, val_loader, num_classes, label_encoder
