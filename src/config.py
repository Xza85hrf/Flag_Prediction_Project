import os
import logging
from typing import Tuple, List, Dict, Any

# Paths
# Define the root directory of the project
PROJECT_ROOT: str = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
# Define directories for data storage and processing
DATA_DIR: str = os.path.join(PROJECT_ROOT, "data")
FLAG_IMAGES_DIR: str = os.path.join(DATA_DIR, "flag_images")
PROCESSED_DIR: str = os.path.join(DATA_DIR, "processed")
MODELS_DIR: str = os.path.join(PROJECT_ROOT, "models")
LOG_DIR: str = os.path.join(PROJECT_ROOT, "logs")

# Model filenames
# Specify the filenames for the trained classifier model and label encoder
CLASSIFIER_MODEL_FILE: str = os.path.join(MODELS_DIR, "flag_classifier.pth")
LABEL_ENCODER_FILE: str = os.path.join(MODELS_DIR, "label_encoder.pkl")

# Image processing
# Define the target size for input images (width, height)
IMAGE_SIZE: Tuple[int, int] = (224, 224)
# Number of bins for color histogram feature extraction
COLOR_HISTOGRAM_BINS: int = 64

# Multiprocessing & Data loading
# Number of worker processes for data loading
NUM_WORKERS: int = 16
# Enable pinned memory for faster data transfer to GPU
PIN_MEMORY: bool = True

# Training
# Proportion of data to use for testing
TEST_SIZE: float = 0.2
# Minimum size of the test set
MIN_TEST_SIZE: float = 0.1
# Random seed for reproducibility
RANDOM_STATE: int = 42
# Number of training epochs
EPOCHS: int = 10
# Number of epochs for fine-tuning
FINE_TUNE_EPOCHS: int = 5
# Number of samples per batch
BATCH_SIZE: int = 32

# Learning rates
INITIAL_LEARNING_RATE: float = 1e-4
FINE_TUNE_LEARNING_RATE: float = 1e-5

# Early stopping
# Number of epochs with no improvement after which training will be stopped
EARLY_STOPPING_PATIENCE: int = 5

# Learning rate schedule
# Factor by which the learning rate will be reduced
LR_DECAY_RATE: float = 0.1
# Number of epochs between learning rate decay
LR_DECAY_STEPS: int = 10
# Number of epochs for learning rate warm-up
WARMUP_EPOCHS: int = 5

# Data augmentation
# Factor by which to increase the dataset size through augmentation
AUGMENTATION_FACTOR: int = 50
# Flag to enable/disable data augmentation
DATA_AUGMENTATION: bool = True

# Cross-validation
# Number of folds for k-fold cross-validation
N_SPLITS: int = 5
# Flag to enable/disable cross-validation
USE_CROSS_VALIDATION: bool = False

# Mixup data augmentation
# Flag to enable/disable mixup augmentation
USE_MIXUP: bool = False
# Alpha parameter for mixup augmentation
MIXUP_ALPHA: float = 0.2

# Logging configuration
LOG_LEVEL: int = logging.INFO
LOG_FILE: str = os.path.join(LOG_DIR, "flag_prediction.log")

# URLs
# URL for the gallery of sovereign state flags
FLAG_GALLERY_URL: str = "https://en.wikipedia.org/wiki/Gallery_of_sovereign_state_flags"

# Model architecture
# Number of units in the dense layer
DENSE_UNITS: int = 256
# Dropout rate for regularization
DROPOUT_RATE: float = 0.3

# Data augmentation parameters
# Range of random rotation
ROTATION_RANGE: int = 20
# Range for random horizontal and vertical shifts
WIDTH_SHIFT_RANGE: float = 0.2
HEIGHT_SHIFT_RANGE: float = 0.2
# Range for random shear transformation
SHEAR_RANGE: float = 0.2
# Range for random zoom
ZOOM_RANGE: float = 0.2
# Enable/disable random horizontal flip
HORIZONTAL_FLIP: bool = True
# Enable/disable random vertical flip
VERTICAL_FLIP: bool = False
# Method for filling in newly created pixels
FILL_MODE: str = "reflect"
# Range for random brightness adjustment
BRIGHTNESS_RANGE: List[float] = [0.9, 1.1]
# Range for random channel shifts
CHANNEL_SHIFT_RANGE: float = 20.0

# Transfer learning
# Number of layers to make trainable (-1 means all layers)
TRAINABLE_LAYERS: int = -1

# Model selection
# List of model architectures to train
MODELS_TO_TRAIN: List[str] = ["resnet50", "mobilenet_v2", "efficientnet_b0"]

# Learning rate schedule parameters
# Factor by which the learning rate will be reduced
LR_SCHEDULE_FACTOR: float = 0.1
# Number of epochs with no improvement after which learning rate will be reduced
LR_SCHEDULE_PATIENCE: int = 5
# Minimum learning rate
LR_SCHEDULE_MIN_LR: float = 1e-6

# Validation split
# Proportion of training data to use for validation
VALIDATION_SPLIT: float = 0.2

# Save best model
# Flag to enable/disable saving the best model during training
SAVE_BEST_MODEL: bool = True
# Filename for the best model
BEST_MODEL_FILENAME: str = "best_model.pth"

# Data preprocessing
# Flag to enable/disable image normalization
NORMALIZE_IMAGES: bool = True
# Factor to rescale pixel values
RESCALE_FACTOR: float = 1.0 / 255.0

# Class balancing
# Flag to enable/disable class balancing
USE_CLASS_BALANCING: bool = True
# Strategy for computing class weights
CLASS_WEIGHT_STRATEGY: str = "balanced"
# Flag to enable/disable class weighting
CLASS_WEIGHTING: bool = True

# Fine-tuning
# Flag to enable/disable fine-tuning
USE_FINE_TUNING: bool = True

# Visualization
# Flag to enable/disable plotting of training history
PLOT_TRAINING_HISTORY: bool = True
# Flag to enable/disable plotting of confusion matrix
PLOT_CONFUSION_MATRIX: bool = True

# Hardware acceleration
# Flag to enable/disable mixed precision training
USE_MIXED_PRECISION: bool = True

# Custom metrics
# List of custom metrics to compute during training
CUSTOM_METRICS: List[str] = ["accuracy", "top_k_accuracy"]

# L2 regularization
# Lambda parameter for L2 regularization
L2_LAMBDA: float = 1e-4

# Confusion matrix
# Filename for saving the confusion matrix plot
CONFUSION_MATRIX_FILENAME: str = "confusion_matrix.png"

# Stratified sampling
# Flag to enable/disable stratified sampling
USE_STRATIFIED_SAMPLING: bool = True

# Multi-label classification
# Threshold for multi-label classification
MULTI_LABEL_THRESHOLD: float = 0.5
# Flag to enable/disable multi-label classification
USE_MULTI_LABEL: bool = False


def ensure_directories() -> None:
    """
    Ensure that all necessary directories exist.
    If a directory doesn't exist, it will be created.
    """
    directories: List[str] = [
        DATA_DIR,
        FLAG_IMAGES_DIR,
        PROCESSED_DIR,
        MODELS_DIR,
        LOG_DIR,
    ]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


# Data file path
# Path to the processed data CSV file
DATA_PATH: str = os.path.join(PROCESSED_DIR, "processed_data.csv")


def get_model_path(model_type: str) -> str:
    """
    Get the file path for a specific model type.

    Args:
        model_type (str): The type of the model (e.g., 'resnet50', 'mobilenet_v2').

    Returns:
        str: The full file path for the model.
    """
    return os.path.join(MODELS_DIR, f"flag_classifier_{model_type}.pth")


def get_history_path(model_type: str) -> str:
    """
    Get the file path for the training history of a specific model type.

    Args:
        model_type (str): The type of the model (e.g., 'resnet50', 'mobilenet_v2').

    Returns:
        str: The full file path for the training history CSV.
    """
    return os.path.join(LOG_DIR, f"training_history_{model_type}.csv")
