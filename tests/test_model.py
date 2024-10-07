import pytest
import torch
import torch.nn as nn
from src.model import (
    create_model,
    get_optimizer,
    get_criterion,
    get_scheduler,
    FlagClassifier,
)
from src import config


@pytest.mark.parametrize("num_classes", [1, 10, 100])
@pytest.mark.parametrize("model_type", ["resnet50", "mobilenet_v2", "efficientnet_b0"])
@pytest.mark.parametrize("multi_label", [True, False])
def test_create_model(num_classes, model_type, multi_label):
    """
    Test the create_model function with various configurations.

    This test ensures that:
    1. The created model is an instance of FlagClassifier
    2. The model's multi_label and model_type attributes are set correctly
    3. The model can perform a forward pass with the expected output shape
    4. For multi-label classification, the output is between 0 and 1
    5. For single-label classification, the output dtype is float32

    Args:
        num_classes (int): Number of classes for the model to predict
        model_type (str): Type of the base model architecture
        multi_label (bool): Whether the model is for multi-label classification
    """
    model = create_model(
        num_classes=num_classes, model_type=model_type, multi_label=multi_label
    )
    assert isinstance(model, FlagClassifier)
    assert model.multi_label == multi_label
    assert model.model_type == model_type

    # Test forward pass
    input_tensor = torch.randn(2, 3, 224, 224)
    output = model(input_tensor)
    assert output.shape == (2, num_classes)

    if multi_label:
        assert torch.all((output >= 0) & (output <= 1))
    else:
        assert output.dtype == torch.float32


def test_create_model_invalid_type():
    """
    Test that create_model raises a ValueError when given an invalid model type.
    """
    with pytest.raises(ValueError, match="Unsupported model type"):
        create_model(num_classes=10, model_type="invalid_model", multi_label=False)


def test_get_optimizer():
    """
    Test the get_optimizer function.

    This test ensures that:
    1. The returned optimizer is an instance of Adam
    2. The learning rate and weight decay are set according to the config
    """
    model = create_model(num_classes=10, model_type="resnet50", multi_label=False)
    optimizer = get_optimizer(model)
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.defaults["lr"] == config.INITIAL_LEARNING_RATE
    assert optimizer.defaults["weight_decay"] == config.L2_LAMBDA


@pytest.mark.parametrize("multi_label", [True, False])
def test_get_criterion(multi_label):
    """
    Test the get_criterion function for both multi-label and single-label scenarios.

    Args:
        multi_label (bool): Whether the criterion is for multi-label classification
    """
    criterion = get_criterion(multi_label)
    if multi_label:
        assert isinstance(criterion, nn.BCEWithLogitsLoss)
    else:
        assert isinstance(criterion, nn.CrossEntropyLoss)


def test_get_scheduler():
    """
    Test the get_scheduler function.

    This test ensures that:
    1. The returned scheduler is an instance of ReduceLROnPlateau
    2. The scheduler has the min_lrs attribute
    3. The minimum learning rate is set according to the config
    """
    model = create_model(num_classes=10, model_type="resnet50", multi_label=False)
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert hasattr(scheduler, "min_lrs")
    assert scheduler.min_lrs[0] == config.LR_SCHEDULE_MIN_LR


def test_flag_classifier_freeze_unfreeze():
    """
    Test the freezing and unfreezing of layers in the FlagClassifier.

    This test checks if the number of trainable parameters is correct based on the
    TRAINABLE_LAYERS configuration:
    - If 0, only the final fully connected layer should be trainable
    - If > 0, more parameters than just the final layer should be trainable
    - If < 0, all parameters should be trainable
    """
    model = FlagClassifier(num_classes=10, model_type="resnet50", multi_label=False)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    if config.TRAINABLE_LAYERS == 0:
        assert (
            trainable_params
            == model.base_model.fc.weight.numel() + model.base_model.fc.bias.numel()
        )
    elif config.TRAINABLE_LAYERS > 0:
        assert (
            trainable_params
            > model.base_model.fc.weight.numel() + model.base_model.fc.bias.numel()
        )
    else:
        assert trainable_params == total_params


@pytest.mark.parametrize(
    "input_shape", [(1, 3, 224, 224), (4, 3, 224, 224), (16, 3, 224, 224)]
)
def test_flag_classifier_forward(input_shape):
    """
    Test the forward pass of the FlagClassifier with different batch sizes.

    This test ensures that the model can handle inputs of various batch sizes
    and produces outputs of the expected shape.

    Args:
        input_shape (tuple): Shape of the input tensor (batch_size, channels, height, width)
    """
    model = FlagClassifier(num_classes=10, model_type="resnet50", multi_label=False)
    input_tensor = torch.randn(*input_shape)
    output = model(input_tensor)
    assert output.shape == (input_shape[0], 10)


def test_flag_classifier_multi_label():
    """
    Test the FlagClassifier in multi-label mode.

    This test ensures that when the model is in multi-label mode,
    all output values are between 0 and 1 (inclusive).
    """
    model = FlagClassifier(num_classes=10, model_type="resnet50", multi_label=True)
    input_tensor = torch.randn(2, 3, 224, 224)
    output = model(input_tensor)
    assert torch.all((output >= 0) & (output <= 1))
