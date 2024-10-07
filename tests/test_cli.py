import pytest
from click.testing import CliRunner
from src.cli import cli, download, process, train, predict, full_pipeline
import os
import logging


@pytest.fixture
def runner():
    """
    Pytest fixture to create a CliRunner instance.
    This runner can be used to invoke CLI commands in tests.
    """
    return CliRunner()


def test_download_command(runner, mocker):
    """
    Test the 'download' command of the CLI.

    This test checks if the download command correctly calls the collect_data function
    and prints the expected output.

    Args:
        runner (CliRunner): Pytest fixture for running CLI commands
        mocker (MockFixture): Pytest fixture for mocking
    """
    mock_collect_data = mocker.patch("src.cli.collect_data")
    result = runner.invoke(download)
    assert result.exit_code == 0
    assert "Starting flag image download..." in result.output
    mock_collect_data.assert_called_once()


def test_process_command(runner, mocker):
    """
    Test the 'process' command of the CLI with valid input.

    This test checks if the process command correctly calls the process_images function
    with the specified number of duplications.

    Args:
        runner (CliRunner): Pytest fixture for running CLI commands
        mocker (MockFixture): Pytest fixture for mocking
    """
    mock_process_images = mocker.patch("src.cli.process_images")
    result = runner.invoke(process, ["--duplicate-times", "2"])
    assert result.exit_code == 0
    assert "Starting image processing with 2 duplications..." in result.output
    mock_process_images.assert_called_once_with(2)


def test_process_command_invalid_input(runner):
    """
    Test the 'process' command of the CLI with invalid input (negative number).

    This test ensures that the CLI properly handles and rejects invalid input.

    Args:
        runner (CliRunner): Pytest fixture for running CLI commands
    """
    result = runner.invoke(process, ["--duplicate-times", "-1"])
    assert result.exit_code != 0
    assert "Invalid value for '--duplicate-times'" in result.output


def test_process_command_invalid_duplicate_times(runner):
    """
    Test the 'process' command of the CLI with invalid input (zero duplications).

    This test ensures that the CLI properly handles and rejects invalid input.

    Args:
        runner (CliRunner): Pytest fixture for running CLI commands
    """
    result = runner.invoke(process, ["--duplicate-times", "0"])
    assert result.exit_code != 0
    assert "Invalid value for '--duplicate-times'" in result.output


def test_train_command(runner, mocker):
    """
    Test the 'train' command of the CLI with various options.

    This test checks if the train command correctly calls the train_models function
    with the specified options (multi-label, cross-validate, and model types).

    Args:
        runner (CliRunner): Pytest fixture for running CLI commands
        mocker (MockFixture): Pytest fixture for mocking
    """
    mock_train = mocker.patch("src.cli.train_models", return_value=None)
    result = runner.invoke(
        train,
        [
            "--multi-label",
            "--cross-validate",
            "--models",
            "resnet50",
            "--models",
            "mobilenet_v2",
        ],
    )
    assert result.exit_code == 0
    assert "Starting model training..." in result.output
    assert "Multi-label: True" in result.output
    assert "Cross-validate: True" in result.output
    assert "Training models: resnet50, mobilenet_v2" in result.output
    mock_train.assert_called_once_with(multi_label=True, cross_validate=True)


def test_train_command_invalid_model(runner):
    """
    Test the 'train' command of the CLI with an invalid model name.

    This test ensures that the CLI properly handles and rejects invalid model names.

    Args:
        runner (CliRunner): Pytest fixture for running CLI commands
    """
    result = runner.invoke(train, ["--models", "invalid_model"])
    assert result.exit_code != 0
    assert "Invalid value for '--models'" in result.output


def test_predict_command(runner, mocker, tmp_path):
    """
    Test the 'predict' command of the CLI.

    This test checks if the predict command correctly calls the predict_country function
    with the specified options and handles the output properly.

    Args:
        runner (CliRunner): Pytest fixture for running CLI commands
        mocker (MockFixture): Pytest fixture for mocking
        tmp_path (Path): Pytest fixture for creating temporary directories
    """
    mock_predict_country = mocker.patch(
        "src.cli.predict_country", return_value=["Test Country"]
    )
    test_image = tmp_path / "test_image.png"
    test_image.touch()
    result = runner.invoke(
        predict, [str(test_image), "--multi-label", "--model", "resnet50"]
    )
    assert result.exit_code == 0
    assert (
        f"Predicted countries for image {test_image} using resnet50: Test Country"
        in result.output
    )
    mock_predict_country.assert_called_once_with(
        str(test_image), multi_label=True, model_type="resnet50"
    )


def test_predict_command_nonexistent_image(runner):
    """
    Test the 'predict' command of the CLI with a nonexistent image file.

    This test ensures that the CLI properly handles and rejects invalid image paths.

    Args:
        runner (CliRunner): Pytest fixture for running CLI commands
    """
    result = runner.invoke(predict, ["nonexistent_image.png"])
    assert result.exit_code != 0
    assert "Error: Invalid value for 'IMAGE_PATH'" in result.output


def test_full_pipeline_command(runner, mocker):
    """
    Test the 'full_pipeline' command of the CLI.

    This test checks if the full_pipeline command correctly calls all the necessary
    functions (collect_data, process_images, train_models) with the specified options.

    Args:
        runner (CliRunner): Pytest fixture for running CLI commands
        mocker (MockFixture): Pytest fixture for mocking
    """
    mock_collect_data = mocker.patch("src.cli.collect_data")
    mock_process_images = mocker.patch("src.cli.process_images")
    mock_train_models = mocker.patch("src.cli.train_models")
    result = runner.invoke(
        full_pipeline,
        [
            "--duplicate-times",
            "2",
            "--multi-label",
            "--cross-validate",
            "--models",
            "resnet50",
        ],
    )
    assert result.exit_code == 0
    assert "Starting full pipeline..." in result.output
    assert "Training models: resnet50" in result.output
    mock_collect_data.assert_called_once()
    mock_process_images.assert_called_once_with(2)
    mock_train_models.assert_called_once_with(multi_label=True, cross_validate=True)


def test_full_pipeline_command_exception(runner, mocker):
    """
    Test the 'full_pipeline' command of the CLI when an exception occurs.

    This test ensures that the CLI properly handles and reports exceptions
    that occur during the full pipeline execution.

    Args:
        runner (CliRunner): Pytest fixture for running CLI commands
        mocker (MockFixture): Pytest fixture for mocking
    """
    mocker.patch("src.cli.collect_data", side_effect=Exception("Test error"))
    result = runner.invoke(full_pipeline)
    assert "An error occurred during the full pipeline: Test error" in result.output


def test_logging_setup(runner, mocker, tmp_path):
    """
    Test the logging setup functionality.

    This test checks if the logging is properly configured with the correct
    file path, log level, and format.

    Args:
        runner (CliRunner): Pytest fixture for running CLI commands
        mocker (MockFixture): Pytest fixture for mocking
        tmp_path (Path): Pytest fixture for creating temporary directories
    """
    mock_logging = mocker.patch("src.cli.logging")
    mock_config = mocker.patch("src.cli.config")
    mock_config.LOG_DIR = str(tmp_path)
    mock_config.LOG_LEVEL = logging.INFO

    # Call setup_logging directly
    from src.cli import setup_logging

    setup_logging()

    assert mock_logging.basicConfig.called
    assert mock_logging.StreamHandler.called
    assert mock_logging.Formatter.called
    assert mock_logging.getLogger.called

    mock_root_logger = mock_logging.getLogger.return_value
    assert mock_root_logger.addHandler.called

    # Check if basicConfig was called with the correct arguments
    mock_logging.basicConfig.assert_called_once_with(
        filename=os.path.join(str(tmp_path), "flag_prediction.log"),
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


@pytest.mark.parametrize(
    "multi_label,cross_validate,models",
    [
        (True, True, ["resnet50"]),
        (False, False, ["mobilenet_v2"]),
        (True, False, ["resnet50", "mobilenet_v2"]),
    ],
)
def test_train_command_combinations(
    runner, mocker, multi_label, cross_validate, models
):
    """
    Test various combinations of options for the 'train' command.

    This parametrized test checks if the train command correctly handles
    different combinations of multi-label, cross-validate, and model options.

    Args:
        runner (CliRunner): Pytest fixture for running CLI commands
        mocker (MockFixture): Pytest fixture for mocking
        multi_label (bool): Whether multi-label classification is enabled
        cross_validate (bool): Whether cross-validation is enabled
        models (list): List of model names to train
    """
    mock_train = mocker.patch("src.cli.train_models", return_value=None)
    args = ["--multi-label"] if multi_label else []
    args += ["--cross-validate"] if cross_validate else []
    for model in models:
        args += ["--models", model]
    result = runner.invoke(train, args)
    assert result.exit_code == 0
    mock_train.assert_called_once_with(
        multi_label=multi_label, cross_validate=cross_validate
    )


def test_predict_command_logging(runner, mocker, tmp_path):
    """
    Test the logging functionality of the 'predict' command.

    This test checks if the predict command correctly logs its output
    using the logger.

    Args:
        runner (CliRunner): Pytest fixture for running CLI commands
        mocker (MockFixture): Pytest fixture for mocking
        tmp_path (Path): Pytest fixture for creating temporary directories
    """
    mock_predict_country = mocker.patch(
        "src.cli.predict_country", return_value=["Test Country"]
    )
    mock_logger = mocker.patch("src.cli.logger")
    test_image = tmp_path / "test_image.png"
    test_image.touch()
    result = runner.invoke(predict, [str(test_image)])
    assert result.exit_code == 0
    mock_logger.info.assert_called_with(
        f"Predicted country for image {test_image} using efficientnet_b0: Test Country"
    )


def test_predict_command_file_writing(runner, mocker, tmp_path):
    """
    Test the file writing functionality of the 'predict' command.

    This test checks if the predict command correctly writes its output
    to a log file.

    Args:
        runner (CliRunner): Pytest fixture for running CLI commands
        mocker (MockFixture): Pytest fixture for mocking
        tmp_path (Path): Pytest fixture for creating temporary directories
    """
    mock_predict_country = mocker.patch(
        "src.cli.predict_country", return_value=["Test Country"]
    )
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    mocker.patch("src.cli.config.LOG_DIR", str(log_dir))
    test_image = tmp_path / "test_image.png"
    test_image.touch()
    result = runner.invoke(predict, [str(test_image)])
    assert result.exit_code == 0
    prediction_log = log_dir / "prediction_results.log"
    assert prediction_log.exists()
    with open(prediction_log, "r") as f:
        assert (
            f"Predicted country for image {test_image} using efficientnet_b0: Test Country"
            in f.read()
        )


def test_cli_with_invalid_command(runner):
    """
    Test the CLI's response to an invalid command.

    This test ensures that the CLI properly handles and rejects invalid commands.

    Args:
        runner (CliRunner): Pytest fixture for running CLI commands
    """
    result = runner.invoke(cli, ["invalid_command"])
    assert result.exit_code != 0
    assert "No such command 'invalid_command'" in result.output


def test_train_command_with_conflicting_options(runner):
    """
    Test the 'train' command with conflicting options.

    This test checks if the CLI properly handles and rejects conflicting or invalid
    combinations of options for the train command.

    Args:
        runner (CliRunner): Pytest fixture for running CLI commands
    """
    result = runner.invoke(
        train, ["--multi-label", "--models", "resnet50", "--models", "invalid_model"]
    )
    assert result.exit_code != 0
    assert "Invalid value for '--models'" in result.output


def test_predict_command_with_nonexistent_model(runner, tmp_path):
    """
    Test the 'predict' command with a nonexistent model.

    This test ensures that the CLI properly handles and rejects invalid model names
    for the predict command.

    Args:
        runner (CliRunner): Pytest fixture for running CLI commands
        tmp_path (Path): Pytest fixture for creating temporary directories
    """
    test_image = tmp_path / "test_image.png"
    test_image.touch()
    result = runner.invoke(predict, [str(test_image), "--model", "nonexistent_model"])
    assert result.exit_code != 0
    assert "Invalid value for '--model'" in result.output
