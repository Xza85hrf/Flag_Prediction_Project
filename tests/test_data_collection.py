import pytest
import os
import json
import logging
import requests
from bs4 import BeautifulSoup
from src.data_collection import (
    download_flag_images,
    sanitize_country_name,
    save_metadata,
    download_flag,
    get_flag_image_url,
    download_flags,
    retry_with_exponential_backoff,
)
from src import config
from unittest.mock import MagicMock


@pytest.fixture
def mock_session(mocker):
    """
    Fixture to create a mock session for testing HTTP requests.

    Args:
        mocker: pytest-mock fixture for creating mocks

    Returns:
        A mock session object with predefined response attributes
    """
    mock_session = mocker.Mock()
    mock_response = mocker.Mock()
    mock_response.content = b"Test flag content"
    mock_response.headers = {"Content-Type": "image/png"}
    mock_response.status_code = 200
    mock_session.get.return_value = mock_response
    return mock_session


@pytest.mark.slow
def test_download_flag_images(tmp_path, mocker, caplog):
    """
    Test the download_flag_images function to ensure it correctly processes flag downloads and creates metadata.

    Args:
        tmp_path: pytest fixture for creating temporary directories
        mocker: pytest-mock fixture for creating mocks
        caplog: pytest fixture for capturing log output
    """
    test_output_dir = tmp_path / "test_flags"
    os.makedirs(test_output_dir, exist_ok=True)

    # Mock the download_flags and download_flag functions
    mocker.patch(
        "src.data_collection.download_flags",
        return_value=[("test_country", "https://example.com/flag.png")],
    )
    mocker.patch("src.data_collection.download_flag", return_value=None)

    with caplog.at_level(logging.INFO):
        download_flag_images(
            output_dir=str(test_output_dir),
            max_retries=1,
            timeout=5,
            max_workers=2,
            skip_small_flags=True,
        )

    # Assert that the metadata file was created and the expected log messages were produced
    assert os.path.exists(
        os.path.join(test_output_dir, "metadata.json")
    ), "Metadata file was not created"
    assert "Downloading flags to:" in caplog.text
    assert "Total flags processed: 1" in caplog.text


@pytest.mark.parametrize(
    "flag_info, content_type, expected_filename",
    [
        (
            ("test_country", "https://example.com/flag.png"),
            "image/png",
            "test_country.png",
        ),
        (
            ("svg_country", "https://example.com/flag.svg"),
            "image/svg+xml",
            "svg_country.png",
        ),
    ],
)
def test_download_flag(
    tmp_path, flag_info, content_type, expected_filename, mock_session, mocker
):
    """
    Test the download_flag function with different file types (PNG and SVG).

    Args:
        tmp_path: pytest fixture for creating temporary directories
        flag_info: tuple containing country name and flag URL
        content_type: MIME type of the flag image
        expected_filename: expected filename of the downloaded flag
        mock_session: mock session fixture
        mocker: pytest-mock fixture for creating mocks
    """
    test_output_dir = tmp_path / "test_flags"
    os.makedirs(test_output_dir, exist_ok=True)

    mock_session.get.return_value.headers = {"Content-Type": content_type}

    # Mock SVG to PNG conversion if the content type is SVG
    if content_type == "image/svg+xml":
        mocker.patch("cairosvg.svg2png", return_value=b"Converted PNG content")

    download_flag(flag_info, mock_session, str(test_output_dir), False)

    # Assert that the flag file was created with the expected filename
    assert os.path.exists(
        os.path.join(test_output_dir, expected_filename)
    ), f"{expected_filename} was not created"


def test_download_flag_small_file(tmp_path, mock_session):
    """
    Test that small flag files are not downloaded when skip_small_flags is True.

    Args:
        tmp_path: pytest fixture for creating temporary directories
        mock_session: mock session fixture
    """
    test_output_dir = tmp_path / "test_flags"
    os.makedirs(test_output_dir, exist_ok=True)

    mock_session.get.return_value.content = b"Small"

    download_flag(
        ("small_country", "https://example.com/small_flag.png"),
        mock_session,
        str(test_output_dir),
        True,
    )

    # Assert that the small flag file was not created
    assert not os.path.exists(
        os.path.join(test_output_dir, "small_country.png")
    ), "Small flag file should not be created"


def test_download_flag_network_error(tmp_path, mocker, caplog):
    """
    Test error handling when a network error occurs during flag download.

    Args:
        tmp_path: pytest fixture for creating temporary directories
        mocker: pytest-mock fixture for creating mocks
        caplog: pytest fixture for capturing log output
    """
    test_output_dir = tmp_path / "test_flags"
    os.makedirs(test_output_dir, exist_ok=True)

    mock_session = mocker.Mock()
    mock_session.get.side_effect = Exception("Network error")

    with caplog.at_level(logging.WARNING):
        download_flag(
            ("error_country", "https://example.com/error_flag.png"),
            mock_session,
            str(test_output_dir),
            False,
        )

    # Assert that the expected error message was logged
    assert (
        "Unexpected error downloading flag for error_country: Network error"
        in caplog.text
    )


def test_download_flag_images_network_error(tmp_path, mocker, caplog):
    """
    Test error handling when a network error occurs during the entire flag download process.

    Args:
        tmp_path: pytest fixture for creating temporary directories
        mocker: pytest-mock fixture for creating mocks
        caplog: pytest fixture for capturing log output
    """
    mocker.patch(
        "src.data_collection.download_flags",
        side_effect=requests.RequestException("Network error"),
    )
    with caplog.at_level(logging.ERROR):
        download_flag_images(output_dir=str(tmp_path))
    assert "Network error during flag download: Network error" in caplog.text


def test_sanitize_country_name():
    """
    Test the sanitize_country_name function with various input strings.
    """
    assert (
        sanitize_country_name("United States of America (USA)")
        == "united_states_of_america"
    )
    assert sanitize_country_name("Côte d'Ivoire") == "cote_d_ivoire"
    assert sanitize_country_name("  spaces  ") == "spaces"


def test_save_metadata(tmp_path):
    """
    Test the save_metadata function to ensure it correctly saves metadata to a JSON file.

    Args:
        tmp_path: pytest fixture for creating temporary directories
    """
    test_output_dir = tmp_path / "test_flags"
    os.makedirs(test_output_dir, exist_ok=True)

    save_metadata("test_country", "https://example.com/flag.png", str(test_output_dir))

    metadata_file = os.path.join(test_output_dir, "metadata.json")
    assert os.path.exists(metadata_file), "Metadata file was not created"

    with open(metadata_file, "r") as f:
        data = json.load(f)

    # Assert that the metadata was correctly saved
    assert len(data) == 1
    assert data[0]["country"] == "test_country"
    assert data[0]["url"] == "https://example.com/flag.png"


def test_download_flag_images_no_flags(tmp_path, mocker, caplog):
    """
    Test the download_flag_images function when no flags are found.

    Args:
        tmp_path: pytest fixture for creating temporary directories
        mocker: pytest-mock fixture for creating mocks
        caplog: pytest fixture for capturing log output
    """
    test_output_dir = tmp_path / "test_flags"
    os.makedirs(test_output_dir, exist_ok=True)

    mocker.patch("src.data_collection.download_flags", return_value=[])

    with caplog.at_level(logging.WARNING):
        download_flag_images(output_dir=str(test_output_dir))

    assert "No valid flag links found on the page." in caplog.text


@pytest.mark.parametrize(
    "max_retries, timeout, max_workers, expected_exception",
    [
        (0, 5, 1, ValueError),
        (3, 0, 1, ValueError),
        (3, 5, 0, ValueError),
    ],
)
def test_download_flag_images_invalid_params(
    tmp_path, max_retries, timeout, max_workers, expected_exception
):
    """
    Test the download_flag_images function with invalid parameters.

    Args:
        tmp_path: pytest fixture for creating temporary directories
        max_retries: number of retries for failed downloads
        timeout: timeout for HTTP requests
        max_workers: number of concurrent workers
        expected_exception: expected exception to be raised
    """
    test_output_dir = tmp_path / "test_flags"
    os.makedirs(test_output_dir, exist_ok=True)

    with pytest.raises(expected_exception):
        download_flag_images(
            output_dir=str(test_output_dir),
            max_retries=max_retries,
            timeout=timeout,
            max_workers=max_workers,
        )


def test_get_flag_image_url(mocker):
    """
    Test the get_flag_image_url function to ensure it correctly extracts the flag image URL.

    Args:
        mocker: pytest-mock fixture for creating mocks
    """
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.content = '<html><a class="internal" href="//upload.wikimedia.org/wikipedia/commons/test_flag.png"></a></html>'
    mocker.patch("requests.get", return_value=mock_response)

    a_tag = MagicMock()
    a_tag.__getitem__.return_value = "/wiki/File:Test_Flag.png"
    a_tag.get.return_value = "/wiki/File:Test_Flag.png"

    url = get_flag_image_url(a_tag)
    assert url == "https://upload.wikimedia.org/wikipedia/commons/test_flag.png"


def test_get_flag_image_url_error(mocker):
    """
    Test error handling in the get_flag_image_url function when a network error occurs.

    Args:
        mocker: pytest-mock fixture for creating mocks
    """
    mocker.patch("requests.get", side_effect=requests.RequestException("Network error"))

    a_tag = MagicMock()
    a_tag.__getitem__.return_value = "/wiki/File:Test_Flag.png"
    a_tag.get.return_value = "/wiki/File:Test_Flag.png"

    url = get_flag_image_url(a_tag)
    assert url is None


def test_download_flags(mocker):
    """
    Test the download_flags function to ensure it correctly parses the HTML and extracts flag information.

    Args:
        mocker: pytest-mock fixture for creating mocks
    """
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.content = """
    <html>
        <head><title>Test Title</title></head>
        <body>
            <table class="wikitable">
                <tr><th>Country</th><th>Flag</th></tr>
                <tr><td>Test Country</td><td><a href="/wiki/File:Flag.png">Flag</a></td></tr>
            </table>
        </body>
    </html>
    """
    mocker.patch("requests.Session.get", return_value=mock_response)
    mocker.patch(
        "src.data_collection.get_flag_image_url",
        return_value="https://example.com/flag.png",
    )
    mocker.patch(
        "src.data_collection.sanitize_country_name", return_value="test_country"
    )

    mock_session = mocker.Mock()
    mock_session.get.return_value = mock_response

    result = download_flags(mock_session)

    assert len(result) == 1
    assert result[0] == ("test_country", "https://example.com/flag.png")


def test_retry_with_exponential_backoff(mocker):
    """
    Test the retry_with_exponential_backoff decorator to ensure it retries the function on failure.

    Args:
        mocker: pytest-mock fixture for creating mocks
    """
    mock_func = mocker.Mock(side_effect=[ValueError, ValueError, "success"])
    mock_func.__name__ = "mock_func"
    decorated_func = retry_with_exponential_backoff()(mock_func)

    result = decorated_func()

    assert result == "success"
    assert mock_func.call_count == 3


def test_retry_with_exponential_backoff_all_fail(mocker, caplog):
    """
    Test the retry_with_exponential_backoff decorator when all retries fail.

    Args:
        mocker: pytest-mock fixture for creating mocks
        caplog: pytest fixture for capturing log output
    """
    mock_func = mocker.Mock(side_effect=ValueError("Test error"))
    mock_func.__name__ = "mock_func"
    decorated_func = retry_with_exponential_backoff(max_retries=2)(mock_func)

    caplog.set_level(logging.WARNING)

    with pytest.raises(ValueError) as excinfo:
        decorated_func()

    assert str(excinfo.value) == "Test error"
    assert mock_func.call_count == 2  # Initial call + 1 retry

    # Check that the correct log messages were produced
    assert "Retrying after error: Test error. Attempt 1" in caplog.text
    assert "Max retries reached. Failed to execute mock_func" in caplog.text


def test_retry_with_exponential_backoff_logging(mocker, caplog):
    """
    Test the logging behavior of the retry_with_exponential_backoff decorator.

    Args:
        mocker: pytest-mock fixture for creating mocks
        caplog: pytest fixture for capturing log output
    """
    mock_func = mocker.Mock(
        side_effect=[ValueError("Test error"), ValueError("Test error"), "success"]
    )
    mock_func.__name__ = "mock_func"
    decorated_func = retry_with_exponential_backoff(max_retries=3)(mock_func)

    with caplog.at_level(logging.WARNING):
        result = decorated_func()

    assert result == "success"
    assert mock_func.call_count == 3
    assert "Retrying after error: Test error. Attempt 1" in caplog.text
    assert "Retrying after error: Test error. Attempt 2" in caplog.text


def test_retry_with_exponential_backoff_no_retry(mocker, caplog):
    """
    Test the retry_with_exponential_backoff decorator when no retries are allowed.

    Args:
        mocker: pytest-mock fixture for creating mocks
        caplog: pytest fixture for capturing log output
    """
    mock_func = mocker.Mock(side_effect=ValueError("Test error"))
    mock_func.__name__ = "mock_func"
    decorated_func = retry_with_exponential_backoff(max_retries=1)(mock_func)

    caplog.set_level(logging.ERROR)

    with pytest.raises(ValueError) as excinfo:
        decorated_func()

    assert str(excinfo.value) == "Test error"
    assert mock_func.call_count == 1
    assert "Retrying after error" not in caplog.text
    assert "Max retries reached. Failed to execute mock_func" in caplog.text


def test_retry_with_exponential_backoff_increasing_delay(mocker):
    """
    Test that the retry_with_exponential_backoff decorator increases the delay between retries.

    Args:
        mocker: pytest-mock fixture for creating mocks
    """
    mock_sleep = mocker.patch("time.sleep")
    mock_func = mocker.Mock(side_effect=[ValueError, ValueError, ValueError, "success"])
    mock_func.__name__ = "mock_func"
    decorated_func = retry_with_exponential_backoff(
        max_retries=4, initial_delay=1, backoff_factor=2
    )(mock_func)

    decorated_func()

    assert mock_sleep.call_count == 3
    assert mock_sleep.call_args_list == [mocker.call(1), mocker.call(2), mocker.call(4)]
