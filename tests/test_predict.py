import pytest
import torch
from src.predict import load_model_and_encoder, predict_country
from src import config
from src.model import FlagClassifier, create_model
import os
import pickle
from PIL import Image
import logging
from sklearn.preprocessing import LabelEncoder
import numpy as np
from unittest.mock import MagicMock
from torchvision import models

# Ignore warnings related to unpickling LabelEncoder
pytestmark = pytest.mark.filterwarnings(
    "ignore:Trying to unpickle estimator LabelEncoder:UserWarning"
)


@pytest.fixture
def sample_model_and_encoder(tmp_path):
    """
    Create a sample FlagClassifier model and LabelEncoder for testing purposes.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.

    Returns:
        str: Path to the temporary directory containing the model and encoder files.
    """
    # Create a sample FlagClassifier model
    model = FlagClassifier(num_classes=209, model_type="resnet50", multi_label=False)
    # Save only the state dict, not the entire model
    torch.save(model.state_dict(), tmp_path / "flag_classifier_resnet50.pth")

    # Create a sample LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array([f"country_{i}" for i in range(209)])
    with open(tmp_path / "label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    return str(tmp_path)


@pytest.fixture
def sample_image(tmp_path):
    """
    Create a sample image file for testing.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.

    Returns:
        str: Path to the created sample image file.
    """
    image = Image.new("RGB", (224, 224), color="red")
    image_path = tmp_path / "test_flag.png"
    image.save(image_path)
    return str(image_path)


def test_load_model_and_encoder_success(sample_model_and_encoder, mocker):
    """
    Test the load_model_and_encoder function for successful loading of model and encoder.

    Args:
        sample_model_and_encoder: Fixture providing sample model and encoder.
        mocker: Pytest mocker object for mocking dependencies.
    """
    # Mock the config paths to use the sample model and encoder
    mocker.patch("src.config.MODELS_DIR", sample_model_and_encoder)
    mocker.patch(
        "src.config.LABEL_ENCODER_FILE",
        os.path.join(sample_model_and_encoder, "label_encoder.pkl"),
    )

    # Create a dummy label encoder
    dummy_label_encoder = LabelEncoder()
    dummy_label_encoder.classes_ = np.array([f"country_{i}" for i in range(209)])
    mocker.patch("pickle.load", return_value=dummy_label_encoder)

    # Create a mock ResNet50 model with the expected 'fc' attribute
    class MockResNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(2048, 1000)  # Mimicking the final layer of ResNet50

    dummy_resnet = MockResNet()
    mocker.patch("torchvision.models.resnet50", return_value=dummy_resnet)

    # Mock create_model to return a real FlagClassifier instance
    mocker.patch("src.model.create_model", side_effect=lambda *args, **kwargs: FlagClassifier(*args, **kwargs))

    # Create a dummy state dict that matches the expected structure
    dummy_state_dict = {
        "base_model.fc.weight": torch.randn(209, 2048),
        "base_model.fc.bias": torch.randn(209),
    }

    # Mock torch.load to return the dummy state dict
    mocker.patch("torch.load", return_value=dummy_state_dict)

    # Mock get_model_path to return a valid path
    mocker.patch("src.config.get_model_path", return_value=os.path.join(sample_model_and_encoder, "dummy_model.pth"))

    # Load the model and encoder
    model, encoder = load_model_and_encoder(multi_label=False, model_type="resnet50")

    # Assert that the loaded objects are of the correct type
    assert isinstance(model, torch.nn.Module)
    assert isinstance(encoder, LabelEncoder)


def test_load_model_and_encoder_failure(mocker):
    """
    Test the load_model_and_encoder function for failure case when files are not found.

    Args:
        mocker: Pytest mocker object for mocking dependencies.
    """
    # Mock non-existent paths
    mocker.patch("src.config.MODELS_DIR", "/non_existent_path")
    mocker.patch("src.config.LABEL_ENCODER_FILE", "/non_existent_file.pkl")

    # Attempt to load the model and encoder
    model, encoder = load_model_and_encoder(multi_label=False, model_type="resnet50")

    # Assert that both model and encoder are None when loading fails
    assert model is None
    assert encoder is None


@pytest.mark.parametrize("multi_label", [False, True])
@pytest.mark.parametrize("model_type", ["resnet50", "mobilenet_v2", "efficientnet_b0"])
def test_predict_country(
    sample_model_and_encoder, sample_image, mocker, multi_label, model_type
):
    """
    Test the predict_country function with different combinations of multi_label and model_type.

    Args:
        sample_model_and_encoder: Fixture providing sample model and encoder.
        sample_image: Fixture providing a sample image path.
        mocker: Pytest mocker object for mocking dependencies.
        multi_label: Boolean indicating whether prediction is multi-label or not.
        model_type: String specifying the type of model to use.
    """
    # Mock config paths and dependencies
    mocker.patch("src.config.MODELS_DIR", sample_model_and_encoder)
    mocker.patch(
        "src.config.LABEL_ENCODER_FILE",
        os.path.join(sample_model_and_encoder, "label_encoder.pkl"),
    )

    # Create a dummy label encoder
    dummy_label_encoder = LabelEncoder()
    dummy_label_encoder.classes_ = np.array([f"country_{i}" for i in range(209)])
    mocker.patch("pickle.load", return_value=dummy_label_encoder)

    # Create a mock model
    mock_model = MagicMock()
    mock_model.eval.return_value = None
    mock_model.return_value = torch.rand(1, 209)

    # Mock various functions and methods
    mocker.patch("src.model.create_model", return_value=mock_model)
    mocker.patch("torch.load", return_value={"dummy": "state_dict"})
    mocker.patch.object(FlagClassifier, "load_state_dict")
    mocker.patch(
        "src.predict.load_model_and_encoder",
        return_value=(mock_model, dummy_label_encoder),
    )

    # Mock country predictions
    mock_countries = ["United States", "Canada", "Mexico"]
    mocker.patch(
        "sklearn.preprocessing.LabelEncoder.inverse_transform",
        return_value=mock_countries[:1] if not multi_label else mock_countries,
    )

    # Perform country prediction
    countries = predict_country(
        sample_image, multi_label=multi_label, model_type=model_type
    )

    # Assert the prediction results
    assert isinstance(countries, list)
    if multi_label:
        assert len(countries) <= 3  # We're mocking 3 countries max
    else:
        assert len(countries) == 1
    for country in countries:
        assert isinstance(country, str)
        assert len(country) > 0


def test_predict_country_invalid_image(tmp_path, mocker):
    """
    Test the predict_country function with an invalid image file.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
        mocker: Pytest mocker object for mocking dependencies.
    """
    mocker.patch("src.config.MODELS_DIR", "/dummy_path")
    mocker.patch("src.config.LABEL_ENCODER_FILE", "/dummy_file.pkl")

    # Create an invalid image file (text file)
    invalid_image_path = tmp_path / "invalid_image.txt"
    invalid_image_path.write_text("This is not an image")

    # Assert that an exception is raised when trying to predict with an invalid image
    with pytest.raises(Exception, match="Invalid image file"):
        predict_country(str(invalid_image_path))


def test_predict_country_missing_model(sample_image, mocker):
    """
    Test the predict_country function when the model file is missing.

    Args:
        sample_image: Fixture providing a sample image path.
        mocker: Pytest mocker object for mocking dependencies.
    """
    mocker.patch("src.config.MODELS_DIR", "/non_existent_path")
    mocker.patch("src.config.LABEL_ENCODER_FILE", "/non_existent_file.pkl")

    # Perform prediction with missing model
    countries = predict_country(sample_image)

    # Assert that an empty list is returned when the model is missing
    assert countries == []


@pytest.fixture(autouse=True)
def setup_logging(caplog):
    """
    Set up logging for tests and reset after each test.

    Args:
        caplog: Pytest fixture for capturing log output.
    """
    caplog.set_level(logging.INFO)
    yield
    caplog.set_level(logging.NOTSET)


def test_logging(sample_model_and_encoder, sample_image, mocker, tmp_path, caplog):
    """
    Test the logging functionality of the predict module.

    Args:
        sample_model_and_encoder: Fixture providing sample model and encoder.
        sample_image: Fixture providing a sample image path.
        mocker: Pytest mocker object for mocking dependencies.
        tmp_path: Pytest fixture providing a temporary directory path.
        caplog: Pytest fixture for capturing log output.
    """
    # Mock config paths and dependencies
    mocker.patch("src.config.LOG_DIR", str(tmp_path))
    mocker.patch("src.config.MODELS_DIR", sample_model_and_encoder)
    mocker.patch(
        "src.config.LABEL_ENCODER_FILE",
        os.path.join(sample_model_and_encoder, "label_encoder.pkl"),
    )

    # Mock pickle.load to return a dummy label encoder
    dummy_label_encoder = LabelEncoder()
    dummy_label_encoder.classes_ = np.array([f"country_{i}" for i in range(209)])
    mocker.patch("pickle.load", return_value=dummy_label_encoder)

    # Create a mock model
    mock_model = MagicMock()
    mock_model.eval.return_value = None
    mock_model.return_value = torch.rand(1, 209)

    # Mock load_model_and_encoder to log a message
    def mock_load_model_and_encoder(*args, **kwargs):
        logger = logging.getLogger("src.predict")
        logger.info("Model and label encoder loaded successfully.")
        return mock_model, dummy_label_encoder

    mocker.patch(
        "src.predict.load_model_and_encoder", side_effect=mock_load_model_and_encoder
    )

    # Perform prediction
    predict_country(sample_image)

    # Assert that the expected log message is present
    assert "Model and label encoder loaded successfully" in caplog.text


def test_prediction_results_logging(
    sample_model_and_encoder, sample_image, mocker, tmp_path, caplog
):
    """
    Test the logging of prediction results in the predict module.

    Args:
        sample_model_and_encoder: Fixture providing sample model and encoder.
        sample_image: Fixture providing a sample image path.
        mocker: Pytest mocker object for mocking dependencies.
        tmp_path: Pytest fixture providing a temporary directory path.
        caplog: Pytest fixture for capturing log output.
    """
    # Mock config paths and dependencies
    mocker.patch("src.config.LOG_DIR", str(tmp_path))
    mocker.patch("src.config.MODELS_DIR", sample_model_and_encoder)
    mocker.patch(
        "src.config.LABEL_ENCODER_FILE",
        os.path.join(sample_model_and_encoder, "label_encoder.pkl"),
    )

    # Mock pickle.load to return a dummy label encoder
    dummy_label_encoder = LabelEncoder()
    dummy_label_encoder.classes_ = np.array([f"country_{i}" for i in range(209)])
    mocker.patch("pickle.load", return_value=dummy_label_encoder)

    # Create a mock model
    mock_model = MagicMock()
    mock_model.eval.return_value = None
    mock_model.return_value = torch.rand(1, 209)

    # Mock load_model_and_encoder to log a message
    def mock_load_model_and_encoder(*args, **kwargs):
        logger = logging.getLogger("src.predict")
        logger.info("Model and label encoder loaded successfully.")
        return mock_model, dummy_label_encoder

    mocker.patch(
        "src.predict.load_model_and_encoder", side_effect=mock_load_model_and_encoder
    )

    # Mock the label_encoder.inverse_transform to return a country name
    mock_country = ["United States"]
    mocker.patch(
        "sklearn.preprocessing.LabelEncoder.inverse_transform",
        return_value=mock_country,
    )

    # Perform prediction
    predicted_countries = predict_country(sample_image)

    # Assert that the expected log messages are present and the prediction is correct
    assert "Model and label encoder loaded successfully" in caplog.text
    assert predicted_countries == ["United States"]