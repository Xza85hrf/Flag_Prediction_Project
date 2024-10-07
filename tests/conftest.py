import pytest
import os
import sys
import logging
from datetime import datetime

# Add the project root directory to the Python path
# This allows importing modules from the project root in tests
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Define the path for the test log file
test_log_path = os.path.join(os.path.dirname(__file__), "test_log.log")


# Register the custom marker "slow" to avoid warnings
def pytest_configure(config):
    """
    Pytest hook to configure custom markers.
    This function is called once at the beginning of a test run.

    Args:
        config: The pytest config object

    Purpose:
    - Register the custom marker "slow" to avoid warnings when using this marker in tests
    - Allows users to deselect slow tests using the '-m "not slow"' option
    """
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


# Setup session-wide logging configuration
@pytest.fixture(scope="session", autouse=True)
def setup_logging():
    """
    Pytest fixture to set up logging for the entire test session.

    Scope: session (runs once for the entire test session)
    Autouse: True (automatically used without explicit declaration in test functions)

    This fixture:
    1. Ensures the log directory exists
    2. Removes existing log handlers to avoid duplication
    3. Configures logging to write to both a file and stdout
    4. Logs the start time of the test session
    """
    # Ensure the directory for the log file exists
    os.makedirs(os.path.dirname(test_log_path), exist_ok=True)

    # Remove existing handlers to avoid duplication
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(test_log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info(f"Test session started at {datetime.now()}")


# Common fixture for mocking network session in tests
@pytest.fixture
def mock_session(mocker):
    """
    Pytest fixture to create a mock session for network requests.

    Args:
        mocker: The pytest-mock fixture for creating mock objects

    Returns:
        A mock session object with predefined response attributes

    Purpose:
    - Provides a consistent mock object for tests that involve network requests
    - Allows tests to run without actual network calls, improving speed and reliability
    """
    mock_session = mocker.Mock()
    mock_response = mocker.Mock()
    mock_response.content = b"Test flag content"
    mock_response.headers = {"Content-Type": "image/png"}
    mock_response.status_code = 200
    mock_session.get.return_value = mock_response
    return mock_session


# Hook to log test results
def pytest_runtest_logreport(report):
    """
    Pytest hook that runs after each test.

    Args:
        report: A TestReport object containing information about the test run

    Purpose:
    - Logs the outcome of each test after it has been called
    - Provides a record of test results in the log file
    """
    if report.when == "call":
        logger = logging.getLogger(__name__)
        logger.info(f"Test {report.nodeid} - {report.outcome}")


# Hook to log test session finish
def pytest_sessionfinish(session, exitstatus):
    """
    Pytest hook that runs at the end of the test session.

    Args:
        session: The pytest session object
        exitstatus: The exit status of the session

    Purpose:
    - Logs the end time and exit status of the entire test session
    - Provides a clear marker in the log file for when testing concluded
    """
    logger = logging.getLogger(__name__)
    logger.info(
        f"Test session finished at {datetime.now()} with exit status: {exitstatus}"
    )
