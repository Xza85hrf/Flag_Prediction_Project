import os
import numpy as np
import logging
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.amp import autocast
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Any
from . import config
from .data_utils import get_data_loaders, FlagDataset
from .model import create_model, get_optimizer, get_criterion, get_scheduler

# Set up logging
logger = logging.getLogger(__name__)


def train_models(
    multi_label: bool = config.USE_MULTI_LABEL,
    cross_validate: bool = config.USE_CROSS_VALIDATION,
) -> None:
    """
    Train one or more models based on the configuration settings.

    Args:
        multi_label (bool): Whether to use multi-label classification.
        cross_validate (bool): Whether to use cross-validation.

    This function orchestrates the entire training process, including data loading,
    model creation, training, and evaluation.
    """
    try:
        # Set up device (CPU or GPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Set number of workers for data loading
        num_workers = 0 if os.name == "nt" else config.NUM_WORKERS

        # Load data
        train_loader, val_loader, num_classes, label_encoder = get_data_loaders(
            multi_label=multi_label,
            num_workers=num_workers,
        )

        # Save label encoder
        with open(config.LABEL_ENCODER_FILE, "wb") as f:
            pickle.dump(label_encoder, f)

        results = {}

        # Train each model specified in the configuration
        for model_type in config.MODELS_TO_TRAIN:
            logger.info(f"Training model: {model_type}")

            if cross_validate:
                # Perform cross-validation
                logger.info("Starting cross-validation...")
                result = cross_validate_model(
                    train_loader.dataset,
                    num_classes,
                    device,
                    multi_label,
                    model_type,
                    num_workers,
                )
            else:
                # Train a single model
                logger.info("Starting single model training...")
                model = create_model(num_classes, model_type, multi_label)
                model = model.to(device)
                criterion = get_criterion(multi_label)
                optimizer = get_optimizer(model)
                scheduler = get_scheduler(optimizer)
                result = train_single_model(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    multi_label,
                    criterion,
                    optimizer,
                    scheduler,
                    model_type,
                )

            results[model_type] = result

        # Compare model results
        if results:
            compare_models(results)
        else:
            logger.warning("No models were successfully trained. Skipping comparison.")

    except Exception as e:
        logger.error(f"An error occurred during model training: {str(e)}")
        raise


def cross_validate_model(
    dataset: FlagDataset,
    num_classes: int,
    device: torch.device,
    multi_label: bool,
    model_type: str,
    num_workers: int,
) -> Dict[str, Any]:
    """
    Perform cross-validation for a given model type.

    Args:
        dataset (FlagDataset): The dataset to use for cross-validation.
        num_classes (int): Number of classes in the classification task.
        device (torch.device): The device to use for training (CPU or GPU).
        multi_label (bool): Whether to use multi-label classification.
        model_type (str): The type of model to train.
        num_workers (int): Number of workers for data loading.

    Returns:
        Dict[str, Any]: Results of cross-validation, including mean accuracy and fold accuracies.
    """
    # Set up K-Fold cross-validation
    kfold = KFold(
        n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE
    )

    fold_results = []
    num_samples = len(dataset)
    indices = np.arange(num_samples)

    # Perform K-Fold cross-validation
    for fold, (train_ids, val_ids) in enumerate(kfold.split(indices)):
        logger.info(f"Training fold {fold + 1}/{config.N_SPLITS}")

        # Create data loaders for this fold
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            sampler=train_subsampler,
            num_workers=num_workers,
            pin_memory=config.PIN_MEMORY,
        )
        val_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            sampler=val_subsampler,
            num_workers=num_workers,
            pin_memory=config.PIN_MEMORY,
        )

        # Create and train model for this fold
        model = create_model(num_classes, model_type, multi_label)
        model = model.to(device)
        criterion = get_criterion(multi_label)
        optimizer = get_optimizer(model)
        scheduler = get_scheduler(optimizer)

        result = train_single_model(
            model,
            train_loader,
            val_loader,
            device,
            multi_label,
            criterion,
            optimizer,
            scheduler,
            model_type,
        )
        fold_results.append(result)

    # Analyze cross-validation results
    cv_results = analyze_cv_results(fold_results, model_type)
    cv_results["best_accuracy"] = cv_results["mean_accuracy"]
    return cv_results


def train_single_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    multi_label: bool,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    model_type: str,
) -> Dict[str, Any]:
    """
    Train a single model.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        device (torch.device): The device to use for training.
        multi_label (bool): Whether to use multi-label classification.
        criterion (nn.Module): The loss function.
        optimizer (Optimizer): The optimizer.
        scheduler (_LRScheduler): The learning rate scheduler.
        model_type (str): The type of model being trained.

    Returns:
        Dict[str, Any]: Training results, including best accuracy and loss/accuracy history.
    """
    # Set up mixed precision training if available
    scaler = (
        GradScaler()
        if config.USE_MIXED_PRECISION and torch.cuda.is_available()
        else None
    )

    best_accuracy = 0.0
    early_stopping_counter = 0
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    try:
        # Training loop
        for epoch in range(config.EPOCHS):
            logger.info(f"Epoch {epoch+1}/{config.EPOCHS}")

            # Train for one epoch
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, scaler, multi_label
            )

            # Validate the model
            val_loss, val_acc = validate_epoch(
                model, val_loader, criterion, device, multi_label
            )

            # Record losses and accuracies
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

            # Update learning rate
            scheduler.step(val_loss)

            # Check for improvement and save model if better
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                early_stopping_counter = 0
                torch.save(model.state_dict(), config.get_model_path(model_type))
                logger.info(f"Model saved with accuracy: {best_accuracy:.4f}")
            else:
                early_stopping_counter += 1

            # Early stopping check
            if early_stopping_counter >= config.EARLY_STOPPING_PATIENCE:
                logger.info("Early stopping triggered.")
                break

        # Plot training history
        plot_training_history(
            train_losses, val_losses, train_accuracies, val_accuracies, model_type
        )

        return {
            "best_accuracy": best_accuracy,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies,
        }
    except Exception as e:
        logger.error(f"An error occurred during model training: {str(e)}")
        raise


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    multi_label: bool,
) -> Tuple[float, float]:
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        criterion (nn.Module): The loss function.
        optimizer (Optimizer): The optimizer.
        device (torch.device): The device to use for training.
        scaler (GradScaler): Gradient scaler for mixed precision training.
        multi_label (bool): Whether to use multi-label classification.

    Returns:
        Tuple[float, float]: The average loss and accuracy for this epoch.
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for images, labels in tqdm(train_loader, desc="Training"):
        # Skip batches with only one sample (causes issues with BatchNorm)
        if images.size(0) == 1:
            continue

        # Move data to device and zero gradients
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        # Forward pass and loss calculation
        if config.USE_MIXED_PRECISION and scaler is not None:
            with autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels.float() if multi_label else labels)
            # Backward pass with scaled gradients
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels.float() if multi_label else labels)
            loss.backward()
            optimizer.step()

        # Update running loss
        running_loss += loss.item() * images.size(0)

        # Calculate accuracy
        if multi_label:
            predictions = (outputs > config.MULTI_LABEL_THRESHOLD).float()
            correct_predictions += torch.all(predictions == labels, dim=1).sum().item()
        else:
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels).item()

        total_predictions += labels.size(0)

    # Check for valid batches
    if total_predictions == 0:
        logger.warning("No valid batches found during training epoch")
        return 0.0, 0.0

    # Calculate epoch statistics
    epoch_loss = running_loss / total_predictions
    epoch_acc = correct_predictions / total_predictions
    logger.info(f"Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.4f}")
    return epoch_loss, epoch_acc


def validate_epoch(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    multi_label: bool,
) -> Tuple[float, float]:
    """
    Validate the model for one epoch.

    Args:
        model (nn.Module): The model to validate.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): The loss function.
        device (torch.device): The device to use for validation.
        multi_label (bool): Whether to use multi-label classification.

    Returns:
        Tuple[float, float]: The average loss and accuracy for this validation epoch.
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            # Skip batches with only one sample (causes issues with BatchNorm)
            if images.size(0) == 1:
                continue

            # Move data to device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass and loss calculation
            outputs = model(images)
            loss = criterion(outputs, labels.float() if multi_label else labels)

            # Update running loss
            running_loss += loss.item() * images.size(0)

            # Calculate accuracy
            if multi_label:
                predictions = (outputs > config.MULTI_LABEL_THRESHOLD).float()
                correct_predictions += (
                    torch.all(predictions == labels, dim=1).sum().item()
                )
            else:
                _, preds = torch.max(outputs, 1)
                correct_predictions += torch.sum(preds == labels).item()

            total_predictions += labels.size(0)

    # Check for valid batches
    if total_predictions == 0:
        logger.warning("No valid batches found during validation epoch")
        return 0.0, 0.0

    # Calculate epoch statistics
    epoch_loss = running_loss / total_predictions
    epoch_acc = correct_predictions / total_predictions
    logger.info(
        f"Validation Loss: {epoch_loss:.4f}, Validation Accuracy: {epoch_acc:.4f}"
    )
    return epoch_loss, epoch_acc


def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    train_accuracies: List[float],
    val_accuracies: List[float],
    model_type: str,
) -> None:
    """
    Plot the training history (loss and accuracy) for a model.

    Args:
        train_losses (List[float]): List of training losses for each epoch.
        val_losses (List[float]): List of validation losses for each epoch.
        train_accuracies (List[float]): List of training accuracies for each epoch.
        val_accuracies (List[float]): List of validation accuracies for each epoch.
        model_type (str): The type of model being trained.

    This function creates two subplots:
    1. Training and Validation Loss
    2. Training and Validation Accuracy

    The plots are saved as an image file in the LOG_DIR specified in the config.
    """
    # Create a new figure with two subplots
    plt.figure(figsize=(12, 5))

    # Plot the losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title(f"Training and Validation Loss - {model_type}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot the accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.title(f"Training and Validation Accuracy - {model_type}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Adjust the layout and save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(config.LOG_DIR, f"training_history_{model_type}.png"))
    plt.close()


def analyze_cv_results(
    results: List[Dict[str, Any]], model_type: str
) -> Dict[str, Any]:
    """
    Analyze the results of cross-validation.

    Args:
        results (List[Dict[str, Any]]): A list of dictionaries containing the results for each fold.
        model_type (str): The type of model being analyzed.

    Returns:
        Dict[str, Any]: A dictionary containing the analysis results, including mean accuracy,
                        standard deviation of accuracy, and individual fold accuracies.

    This function calculates statistics on the cross-validation results, logs the information,
    and creates a bar plot of the accuracies for each fold.
    """
    # Extract the best accuracies from each fold
    best_accuracies = [result["best_accuracy"] for result in results]
    mean_accuracy = np.mean(best_accuracies)
    std_accuracy = np.std(best_accuracies)

    # Log the cross-validation results
    logger.info(f"Cross-validation results for {model_type}:")
    logger.info(f"Mean accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")
    logger.info(f"Individual fold accuracies: {best_accuracies}")

    # Plot cross-validation results
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(best_accuracies) + 1), best_accuracies)
    plt.axhline(y=mean_accuracy, color="r", linestyle="--", label="Mean Accuracy")
    plt.title(f"Cross-validation Results - {model_type}")
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(config.LOG_DIR, f"cv_results_{model_type}.png"))
    plt.close()

    # Return the analysis results
    return {
        "mean_accuracy": mean_accuracy,
        "std_accuracy": std_accuracy,
        "fold_accuracies": best_accuracies,
    }


def compare_models(results: Dict[str, Dict[str, Any]]) -> None:
    """
    Compare the performance of different models.

    Args:
        results (Dict[str, Dict[str, Any]]): A dictionary containing the results for each model.
            The keys are model names, and the values are dictionaries containing the model's results.

    This function creates a bar plot comparing the accuracies of different models,
    logs the comparison results, and identifies the best performing model.
    """
    model_names = list(results.keys())
    accuracies = []

    # Extract the accuracies for each model
    for result in results.values():
        if "best_accuracy" in result:
            accuracies.append(result["best_accuracy"])
        elif "mean_accuracy" in result:
            accuracies.append(result["mean_accuracy"])
        else:
            logger.warning(f"No accuracy found for model {result}")
            accuracies.append(0)  # or some default value

    # Create a bar plot of model accuracies
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, accuracies)
    plt.title("Model Comparison")
    plt.xlabel("Model")
    plt.ylabel("Best Accuracy / Mean Accuracy (CV)")
    plt.savefig(os.path.join(config.LOG_DIR, "model_comparison.png"))
    plt.close()

    # Log the comparison results
    logger.info("Model Comparison Results:")
    for model_name, result in results.items():
        if "best_accuracy" in result:
            logger.info(f"{model_name}: Best Accuracy = {result['best_accuracy']:.4f}")
        elif "mean_accuracy" in result:
            logger.info(
                f"{model_name}: Mean Accuracy (CV) = {result['mean_accuracy']:.4f} (+/- {result['std_accuracy']:.4f})"
            )
        else:
            logger.warning(f"No accuracy found for model {model_name}")

    # Identify the best performing model
    best_model = max(results, key=lambda x: accuracies[model_names.index(x)])
    logger.info(f"Best performing model: {best_model}")
