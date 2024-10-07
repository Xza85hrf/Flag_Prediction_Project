import pandas as pd
import pytest
import torch
import numpy as np
from src.train import (
    train_epoch,
    validate_epoch,
    train_single_model,
    cross_validate_model,
    analyze_cv_results,
    compare_models,
    train_models,
)
from src.model import create_model, get_optimizer, get_criterion, get_scheduler
from src.data_utils import get_data_loaders, FlagDataset
from PIL import Image
import os
import logging
import torch.nn as nn
from unittest.mock import MagicMock
import matplotlib

matplotlib.use("Agg")
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="User provided device_type of 'cuda', but CUDA is not available. Disabling",
)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="matplotlib")

# Fixtures


@pytest.fixture
def sample_model():
    """
    Fixture to create a sample model for testing.

    Returns:
        A PyTorch model instance with 3 output classes using ResNet50 architecture.
    """
    return create_model(num_classes=3, model_type="resnet50", multi_label=False)


@pytest.fixture
def sample_data_loaders(sample_data, mocker):
    """
    Fixture to create sample data loaders for testing.

    Args:
        sample_data: Path to sample data CSV file.
        mocker: pytest-mock fixture for mocking.

    Returns:
        Tuple containing train_loader, val_loader, num_classes, and label_encoder.
    """
    mocker.patch("src.config.PROCESSED_DIR", os.path.dirname(sample_data))
    mocker.patch("src.config.DATA_PATH", sample_data)
    return get_data_loaders(multi_label=False, num_workers=0)


@pytest.fixture
def sample_data(tmp_path):
    """
    Fixture to create sample data for testing.

    Args:
        tmp_path: pytest fixture for temporary directory path.

    Returns:
        Path to the created sample data CSV file.
    """
    # Create a DataFrame with sample data
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
def mock_logger(mocker):
    """
    Fixture to mock the logger for testing.

    Args:
        mocker: pytest-mock fixture for mocking.

    Returns:
        Mocked logger object.
    """
    return mocker.patch("src.train.logger")


@pytest.fixture
def mock_grad_scaler(mocker):
    """
    Fixture to mock the GradScaler for testing mixed precision training.
    """
    mock_scaler = mocker.Mock()
    mock_scaler.scale.return_value.backward = mocker.Mock()
    mock_scaler.step = mocker.Mock()
    mock_scaler.update = mocker.Mock()
    return mock_scaler


# Test functions


def test_train_epoch(
    sample_model, sample_data_loaders, mocker, mock_logger, mock_grad_scaler
):
    """
    Test the train_epoch function.

    This test ensures that the train_epoch function runs without errors and
    returns expected output types for loss and accuracy.
    """
    train_loader, _, _, _ = sample_data_loaders
    criterion = get_criterion(multi_label=False)
    optimizer = get_optimizer(sample_model)
    device = torch.device("cpu")

    # Mock tqdm to avoid progress bar output during testing
    mocker.patch("src.train.tqdm", lambda x, *args, **kwargs: x)
    mocker.patch(
        "src.config.USE_MIXED_PRECISION", False
    )  # Disable mixed precision for CPU testing
    mocker.patch("torch.cuda.amp.GradScaler", return_value=mock_grad_scaler)

    epoch_loss, epoch_acc = train_epoch(
        sample_model,
        train_loader,
        criterion,
        optimizer,
        device,
        mock_grad_scaler,
        multi_label=False,
    )

    # Assert that the function returns expected output types
    assert isinstance(epoch_loss, float)
    assert isinstance(epoch_acc, float)
    assert 0 <= epoch_acc <= 1
    mock_logger.info.assert_called_with(
        f"Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.4f}"
    )


def test_validate_epoch(sample_model, sample_data_loaders, mocker, mock_logger):
    """
    Test the validate_epoch function.

    This test ensures that the validate_epoch function runs without errors and
    returns expected output types for loss and accuracy.
    """
    _, val_loader, _, _ = sample_data_loaders
    criterion = get_criterion(multi_label=False)
    device = torch.device("cpu")

    # Mock tqdm to avoid progress bar output during testing
    mocker.patch("src.train.tqdm", lambda x, *args, **kwargs: x)

    epoch_loss, epoch_acc = validate_epoch(
        sample_model, val_loader, criterion, device, multi_label=False
    )

    # Assert that the function returns expected output types
    assert isinstance(epoch_loss, float)
    assert isinstance(epoch_acc, float)
    assert 0 <= epoch_acc <= 1
    mock_logger.info.assert_called_with(
        f"Validation Loss: {epoch_loss:.4f}, Validation Accuracy: {epoch_acc:.4f}"
    )


def test_train_single_model(
    sample_model, sample_data_loaders, tmp_path, mocker, mock_logger, mock_grad_scaler
):
    """
    Test the train_single_model function.

    This test ensures that the train_single_model function runs without errors,
    saves the model, and returns expected output.
    """
    train_loader, val_loader, _, _ = sample_data_loaders
    criterion = get_criterion(multi_label=False)
    optimizer = get_optimizer(sample_model)
    scheduler = get_scheduler(optimizer)
    device = torch.device("cpu")

    # Mock various functions and configurations
    mocker.patch("src.train.tqdm", lambda x, *args, **kwargs: x)
    mocker.patch("src.config.EPOCHS", 2)
    mocker.patch("src.config.EARLY_STOPPING_PATIENCE", 1)
    model_path = str(tmp_path / "test_model.pth")
    mocker.patch("src.config.get_model_path", return_value=model_path)
    mocker.patch("src.train.plot_training_history")
    mocker.patch("src.config.USE_MIXED_PRECISION", True)
    mocker.patch("torch.cuda.amp.GradScaler", return_value=mock_grad_scaler)

    # patch the train_epoch function
    mocker.patch("src.train.train_epoch", return_value=(0.5, 0.6))

    # Mock torch.save to actually save the model
    mock_torch_save = mocker.patch("torch.save")

    # Mock validate_epoch to return improving accuracy
    mocker.patch("src.train.validate_epoch", side_effect=[(0.5, 0.6), (0.4, 0.7)])

    result = train_single_model(
        sample_model,
        train_loader,
        val_loader,
        device,
        False,
        criterion,
        optimizer,
        scheduler,
        "test_model",
    )

    # Assert that the function returns expected output
    assert isinstance(result, dict)
    assert all(
        key in result
        for key in [
            "best_accuracy",
            "train_losses",
            "val_losses",
            "train_accuracies",
            "val_accuracies",
        ]
    )

    # Check if torch.save was called
    assert mock_torch_save.called, "torch.save was not called"

    # Check the arguments of the last call to torch.save
    args, kwargs = mock_torch_save.call_args
    assert len(args) == 2, "torch.save should be called with 2 arguments"
    assert isinstance(
        args[0], dict
    ), "First argument to torch.save should be a dict (state_dict)"
    assert (
        args[1] == model_path
    ), f"Second argument to torch.save should be {model_path}"

    mock_logger.info.assert_any_call(f"Model saved with accuracy: {0.7:.4f}")


def test_cross_validate_model(sample_data_loaders, mocker, mock_logger):
    """
    Test the cross_validate_model function.

    This test ensures that the cross_validate_model function runs without errors
    and returns expected output.
    """
    train_loader, _, num_classes, _ = sample_data_loaders
    device = torch.device("cpu")

    # Mock train_single_model to return a fixed accuracy
    mocker.patch("src.train.train_single_model", return_value={"best_accuracy": 0.8})
    mocker.patch("src.config.N_SPLITS", 2)
    mocker.patch(
        "src.train.analyze_cv_results",
        return_value={"mean_accuracy": 0.8, "std_accuracy": 0.1},
    )

    result = cross_validate_model(
        train_loader.dataset, num_classes, device, False, "resnet50", 0
    )

    # Assert that the function returns expected output
    assert isinstance(result, dict)
    assert "best_accuracy" in result
    assert result["best_accuracy"] == 0.8
    mock_logger.info.assert_any_call("Training fold 1/2")


def test_analyze_cv_results(mocker, mock_logger):
    """
    Test the analyze_cv_results function.

    This test ensures that the analyze_cv_results function runs without errors
    and returns expected output.
    """
    results = [{"best_accuracy": 0.8}, {"best_accuracy": 0.9}]
    mocker.patch("matplotlib.pyplot.savefig")

    result = analyze_cv_results(results, "test_model")

    # Assert that the function returns expected output
    assert isinstance(result, dict)
    assert all(
        key in result for key in ["mean_accuracy", "std_accuracy", "fold_accuracies"]
    )
    mock_logger.info.assert_any_call("Cross-validation results for test_model:")


def test_compare_models(mocker, mock_logger):
    """
    Test the compare_models function.

    This test ensures that the compare_models function runs without errors
    and logs expected information.
    """
    results = {
        "model1": {"best_accuracy": 0.8},
        "model2": {"mean_accuracy": 0.9, "std_accuracy": 0.05},
    }
    mocker.patch("matplotlib.pyplot.savefig")

    compare_models(results)

    mock_logger.info.assert_any_call("Model Comparison Results:")
    mock_logger.info.assert_any_call("Best performing model: model2")


@pytest.mark.parametrize("multi_label", [True, False])
def test_train_epoch_multi_label(
    sample_model,
    sample_data_loaders,
    mocker,
    mock_logger,
    mock_grad_scaler,
    multi_label,
):
    """
    Test the train_epoch function with both single-label and multi-label scenarios.

    This test ensures that the train_epoch function handles both single-label and
    multi-label cases correctly.
    """
    train_loader, _, _, _ = sample_data_loaders
    criterion = get_criterion(multi_label=multi_label)
    optimizer = get_optimizer(sample_model)
    device = torch.device("cpu")

    # Mock various functions and configurations
    mocker.patch("src.train.tqdm", lambda x, *args, **kwargs: x)
    mocker.patch("src.config.MULTI_LABEL_THRESHOLD", 0.5)
    mocker.patch(
        "src.config.USE_MIXED_PRECISION", False
    )  # Disable mixed precision for CPU testing
    mocker.patch("torch.cuda.amp.GradScaler", return_value=mock_grad_scaler)

    # Adjust mock data for multi-label case
    if multi_label:
        num_samples = 32  # Use a small number of samples for testing
        num_classes = 3  # Assuming 3 classes for this test
        mock_images = torch.rand(num_samples, 3, 224, 224)  # Random image tensors
        mock_labels = torch.randint(0, 2, (num_samples, num_classes)).float()
        mock_dataset = torch.utils.data.TensorDataset(mock_images, mock_labels)
        train_loader = torch.utils.data.DataLoader(
            mock_dataset, batch_size=16, shuffle=True
        )

    epoch_loss, epoch_acc = train_epoch(
        sample_model,
        train_loader,
        criterion,
        optimizer,
        device,
        mock_grad_scaler,
        multi_label=multi_label,
    )

    # Assert that the function returns expected output types
    assert isinstance(epoch_loss, float)
    assert isinstance(epoch_acc, float)
    assert 0 <= epoch_acc <= 1


def test_train_single_model_early_stopping(
    sample_model, sample_data_loaders, tmp_path, mocker, mock_logger, mock_grad_scaler
):
    """
    Test the train_single_model function with early stopping.

    This test ensures that the early stopping mechanism in train_single_model
    works correctly.
    """
    train_loader, val_loader, _, _ = sample_data_loaders
    criterion = get_criterion(multi_label=False)
    optimizer = get_optimizer(sample_model)
    scheduler = get_scheduler(optimizer)
    device = torch.device("cpu")

    # Mock various functions and configurations
    mocker.patch("src.train.tqdm", lambda x, *args, **kwargs: x)
    mocker.patch("src.config.EPOCHS", 10)
    mocker.patch("src.config.EARLY_STOPPING_PATIENCE", 2)
    mocker.patch(
        "src.config.get_model_path", return_value=str(tmp_path / "test_model.pth")
    )
    mocker.patch("src.train.plot_training_history")
    mocker.patch("src.config.USE_MIXED_PRECISION", True)
    mocker.patch("torch.cuda.amp.GradScaler", return_value=mock_grad_scaler)

    # patch the train_epoch function
    mocker.patch("src.train.train_epoch", return_value=(0.5, 0.6))

    # Mock validate_epoch to always return decreasing accuracy
    mocker.patch(
        "src.train.validate_epoch", side_effect=[(0.5, 0.8), (0.6, 0.7), (0.7, 0.6)]
    )

    result = train_single_model(
        sample_model,
        train_loader,
        val_loader,
        device,
        False,
        criterion,
        optimizer,
        scheduler,
        "test_model",
    )

    # Assert that early stopping was triggered
    assert (
        len(result["val_accuracies"]) == 3
    )  # Should stop after 3 epochs due to early stopping
    mock_logger.info.assert_any_call("Early stopping triggered.")


def test_train_single_model_exception_handling(
    sample_model, sample_data_loaders, tmp_path, mocker, mock_logger
):
    """
    Test exception handling in the train_single_model function.

    This test ensures that exceptions during training are properly caught and logged.
    """
    train_loader, val_loader, _, _ = sample_data_loaders
    criterion = get_criterion(multi_label=False)
    optimizer = get_optimizer(sample_model)
    scheduler = get_scheduler(optimizer)
    device = torch.device("cpu")

    # Mock various functions and configurations
    mocker.patch("src.train.tqdm", lambda x, *args, **kwargs: x)
    mocker.patch("src.config.EPOCHS", 2)
    mocker.patch(
        "src.config.get_model_path", return_value=str(tmp_path / "test_model.pth")
    )
    mocker.patch("src.train.plot_training_history")

    # Simulate an exception during training
    mocker.patch("src.train.train_epoch", side_effect=RuntimeError("Simulated error"))

    with pytest.raises(RuntimeError):
        train_single_model(
            sample_model,
            train_loader,
            val_loader,
            device,
            False,
            criterion,
            optimizer,
            scheduler,
            "test_model",
        )

    mock_logger.error.assert_called_with(
        "An error occurred during model training: Simulated error"
    )


def test_train_models(mocker, mock_logger):
    """
    Test the train_models function for single model training.

    This test ensures that the train_models function correctly handles
    training a single model without cross-validation.
    """
    mock_data_loaders = (
        mocker.Mock(),  # train_loader
        mocker.Mock(),  # val_loader
        3,  # num_classes
        mocker.Mock(),  # label_encoder
    )
    mocker.patch("src.train.get_data_loaders", return_value=mock_data_loaders)
    mocker.patch("src.train.create_model")
    mocker.patch("src.train.get_criterion")
    mocker.patch("src.train.get_optimizer")
    mocker.patch("src.train.get_scheduler")
    mocker.patch("src.train.train_single_model", return_value={"best_accuracy": 0.9})
    mocker.patch("src.train.compare_models")
    mocker.patch("src.config.MODELS_TO_TRAIN", ["resnet50"])
    mocker.patch("pickle.dump")

    train_models(multi_label=False, cross_validate=False)

    mock_logger.info.assert_any_call("Training model: resnet50")
    mock_logger.info.assert_any_call("Starting single model training...")


def test_train_models_cross_validation(mocker, mock_logger):
    """
    Test the train_models function with cross-validation.

    This test ensures that the train_models function correctly handles
    training models using cross-validation.
    """
    mock_data_loaders = (
        mocker.Mock(),  # train_loader
        mocker.Mock(),  # val_loader
        3,  # num_classes
        mocker.Mock(),  # label_encoder
    )
    mocker.patch("src.train.get_data_loaders", return_value=mock_data_loaders)
    mocker.patch(
        "src.train.cross_validate_model",
        return_value={"mean_accuracy": 0.85, "std_accuracy": 0.05},
    )
    mocker.patch("src.train.compare_models")
    mocker.patch("src.config.MODELS_TO_TRAIN", ["resnet50"])
    mocker.patch("pickle.dump")

    train_models(multi_label=False, cross_validate=True)

    mock_logger.info.assert_any_call("Training model: resnet50")
    mock_logger.info.assert_any_call("Starting cross-validation...")


def test_train_models_exception_handling(mocker, mock_logger):
    """
    Test exception handling in the train_models function.

    This test ensures that exceptions during the train_models function
    are properly caught and logged.
    """
    mocker.patch(
        "src.train.get_data_loaders", side_effect=RuntimeError("Simulated error")
    )

    with pytest.raises(RuntimeError):
        train_models(multi_label=False, cross_validate=False)

    mock_logger.error.assert_called_with(
        "An error occurred during model training: Simulated error"
    )
