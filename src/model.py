import torch
import torch.nn as nn
import torchvision.models as models
from . import config
from typing import Union


# FlagClassifier: A neural network model for classifying flags
class FlagClassifier(nn.Module):
    """
    A neural network model for classifying flags using transfer learning.

    This class creates a classifier based on pre-trained models (ResNet50, MobileNetV2, or EfficientNetB0)
    and adapts them for flag classification tasks. It supports both multi-label and single-label classification.

    Attributes:
        multi_label (bool): Whether the model is used for multi-label classification.
        model_type (str): The type of base model used ('resnet50', 'mobilenet_v2', or 'efficientnet_b0').
        base_model (nn.Module): The adapted pre-trained model.
    """

    def __init__(
        self,
        num_classes: int,
        model_type: str,
        multi_label: bool = config.USE_MULTI_LABEL,
    ):
        """
        Initialize the FlagClassifier.

        Args:
            num_classes (int): Number of classes (flags) to classify.
            model_type (str): Type of the base model to use.
            multi_label (bool): Whether to use multi-label classification.
        """
        super(FlagClassifier, self).__init__()
        self.multi_label = multi_label
        self.model_type = model_type

        # Initialize the base model based on the specified model_type
        if model_type == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            self.base_model = models.resnet50(weights=weights)
            num_ftrs = self.base_model.fc.in_features
            self.base_model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_type == "mobilenet_v2":
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
            self.base_model = models.mobilenet_v2(weights=weights)
            num_ftrs = self.base_model.classifier[1].in_features
            self.base_model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        elif model_type == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            self.base_model = models.efficientnet_b0(weights=weights)
            num_ftrs = self.base_model.classifier[1].in_features
            self.base_model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Layer freezing logic based on config.TRAINABLE_LAYERS
        if config.TRAINABLE_LAYERS == 0:
            # Freeze all layers
            for param in self.base_model.parameters():
                param.requires_grad = False
        elif config.TRAINABLE_LAYERS > 0:
            # Freeze all layers first
            for param in self.base_model.parameters():
                param.requires_grad = False
            # Then unfreeze the last few layers
            params_to_unfreeze = list(self.base_model.parameters())[
                -config.TRAINABLE_LAYERS :
            ]
            for param in params_to_unfreeze:
                param.requires_grad = True
        else:
            # Unfreeze all layers (fine-tune the entire model)
            for param in self.base_model.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        output = self.base_model(x)
        if self.multi_label:
            # Apply sigmoid for multi-label classification
            output = torch.sigmoid(output)
        return output


def create_model(
    num_classes: int, model_type: str, multi_label: bool = config.USE_MULTI_LABEL
) -> FlagClassifier:
    """
    Create and return a FlagClassifier instance.

    Args:
        num_classes (int): Number of classes to classify.
        model_type (str): Type of the base model to use.
        multi_label (bool): Whether to use multi-label classification.

    Returns:
        FlagClassifier: An instance of the FlagClassifier.
    """
    return FlagClassifier(num_classes, model_type, multi_label)


def get_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """
    Create and return an Adam optimizer for the model.

    Args:
        model (nn.Module): The model to optimize.

    Returns:
        torch.optim.Optimizer: Adam optimizer instance.
    """
    return torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.INITIAL_LEARNING_RATE,
        weight_decay=config.L2_LAMBDA,
    )


def get_criterion(
    multi_label: bool = False,
) -> Union[nn.BCEWithLogitsLoss, nn.CrossEntropyLoss]:
    """
    Get the appropriate loss function based on the classification type.

    Args:
        multi_label (bool): Whether the task is multi-label classification.

    Returns:
        Union[nn.BCEWithLogitsLoss, nn.CrossEntropyLoss]: The loss function.
    """
    if multi_label:
        return nn.BCEWithLogitsLoss()
    else:
        return nn.CrossEntropyLoss()


def get_scheduler(
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """
    Create and return a learning rate scheduler.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to schedule.

    Returns:
        torch.optim.lr_scheduler.ReduceLROnPlateau: The learning rate scheduler.
    """
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.LR_SCHEDULE_FACTOR,
        patience=config.LR_SCHEDULE_PATIENCE,
        min_lr=config.LR_SCHEDULE_MIN_LR,
    )
