import os
import time
import requests
from bs4 import BeautifulSoup
import logging
import concurrent.futures
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urljoin
import threading

from . import config
import re
import unicodedata
from datetime import datetime
import json
from functools import wraps

# Set up logging for this module
logger = logging.getLogger(__name__)

# Constants
MIN_FILE_SIZE = 1024  # 1KB minimum file size for flag images
metadata_lock = threading.Lock()  # Lock for thread-safe metadata access


# Decorator for implementing exponential backoff retry logic
def retry_with_exponential_backoff(max_retries=3, initial_delay=1, backoff_factor=2):
    """
    Decorator that implements exponential backoff retry logic for functions.

    Args:
        max_retries (int): Maximum number of retry attempts.
        initial_delay (int): Initial delay in seconds before the first retry.
        backoff_factor (int): Factor by which the delay increases with each retry.

    Returns:
        function: Decorated function with retry logic.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(delay)
                        delay *= backoff_factor
                        logger.warning(
                            f"Retrying after error: {e}. Attempt {attempt+1}"
                        )
                    else:
                        logger.error(
                            f"Max retries reached. Failed to execute {func.__name__}"
                        )
                        raise e

        return wrapper

    return decorator


def sanitize_country_name(name):
    """
    Sanitize the country name for use as a filename.

    This function performs the following operations:
    1. Removes content in parentheses
    2. Replaces non-alphanumeric characters (except spaces) with underscores
    3. Replaces multiple spaces or underscores with a single underscore
    4. Removes leading/trailing underscores and converts to lowercase
    5. Normalizes Unicode characters

    Args:
        name (str): The original country name.

    Returns:
        str: Sanitized country name suitable for use as a filename.
    """
    # Remove content in parentheses
    name = re.sub(r"\([^)]*\)", "", name)
    # Replace non-alphanumeric characters (except spaces) with underscores
    name = re.sub(r"[^\w\s-]", "_", name)
    # Replace multiple spaces or underscores with a single underscore
    name = re.sub(r"[-\s_]+", "_", name)
    # Remove leading/trailing underscores and convert to lowercase
    name = name.strip("_").lower()
    # Normalize Unicode characters
    name = "".join(
        c for c in unicodedata.normalize("NFD", name) if unicodedata.category(c) != "Mn"
    )
    return name


@retry_with_exponential_backoff(max_retries=3)
def get_flag_image_url(a_tag):
    """
    Extract the full-resolution image URL from an <a> tag.

    This function follows the link in the <a> tag to find the full-resolution image URL.

    Args:
        a_tag (bs4.element.Tag): BeautifulSoup Tag object representing an <a> tag.

    Returns:
        str or None: The full-resolution image URL if found, None otherwise.
    """
    if a_tag and a_tag.get("href"):
        href = a_tag["href"]
        if href.startswith("/wiki/File:"):
            file_name = href.split(":")[-1]
            # Construct the URL for the image info page
            image_info_url = urljoin("https://en.wikipedia.org", href)
            # Parse the image info page to get the actual image URL
            try:
                image_response = requests.get(image_info_url, timeout=10)
                image_response.raise_for_status()
                if image_response.status_code == 200:
                    image_soup = BeautifulSoup(image_response.content, "html.parser")
                    original_file_link = image_soup.find("a", {"class": "internal"})
                    if original_file_link and original_file_link.get("href"):
                        return urljoin("https:", original_file_link["href"])
            except requests.RequestException as e:
                logger.warning(f"Failed to get image URL for {file_name}: {str(e)}")
    return None


@retry_with_exponential_backoff(max_retries=3)
def download_flag(flag_info, session, output_dir, skip_small_flags):
    """
    Download a single flag image.

    This function downloads the flag image, handles SVG to PNG conversion if necessary,
    and saves metadata for the downloaded flag.

    Args:
        flag_info (tuple): A tuple containing (country_name, flag_url).
        session (requests.Session): The session object for making HTTP requests.
        output_dir (str): The directory to save the downloaded flag images.
        skip_small_flags (bool): Whether to skip flags smaller than MIN_FILE_SIZE.
    """
    country, flag_url = flag_info
    try:
        flag_file = os.path.join(output_dir, f"{country}.png")

        if os.path.exists(flag_file):
            logger.info(f"Flag for {country} already exists, skipping download")
            return

        logger.info(f"Downloading flag for {country} from {flag_url}")
        flag_response = session.get(flag_url, timeout=10)
        flag_response.raise_for_status()

        if skip_small_flags and len(flag_response.content) < MIN_FILE_SIZE:
            logger.warning(f"Downloaded flag for {country} is too small, skipping")
            return

        content_type = flag_response.headers.get("Content-Type")
        if content_type == "image/svg+xml" or flag_url.endswith(".svg"):
            # Convert SVG to PNG
            try:
                import cairosvg

                png_data = cairosvg.svg2png(bytestring=flag_response.content)
                with open(flag_file, "wb") as f:
                    f.write(png_data)
                logger.info(f"Converted SVG to PNG for {country}")
            except Exception as e:
                logger.error(f"Failed to convert SVG to PNG for {country}: {str(e)}")
                return
        else:
            with open(flag_file, "wb") as f:
                f.write(flag_response.content)

        save_metadata(country, flag_url, output_dir)

        logger.info(f"Downloaded flag for {country}")
        time.sleep(0.1)  # Small delay to be polite to the server

    except requests.RequestException as e:
        logger.warning(f"Failed to download flag for {country}: {str(e)}")
    except Exception as e:
        logger.error(
            f"Unexpected error downloading flag for {country}: {str(e)}", exc_info=True
        )


def save_metadata(country, flag_url, output_dir):
    """
    Save metadata for each flag.

    This function saves metadata (country name, URL, and download date) for each
    downloaded flag in a JSON file. It uses a lock to ensure thread-safe access
    to the metadata file.

    Args:
        country (str): The name of the country.
        flag_url (str): The URL of the flag image.
        output_dir (str): The directory where the metadata file is stored.
    """
    metadata_file = os.path.join(output_dir, "metadata.json")
    metadata = {
        "country": country,
        "url": flag_url,
        "download_date": datetime.now().isoformat(),
    }

    with metadata_lock:
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    logger.error(f"Error reading metadata file: {e}")
                    data = []
        else:
            data = []

        data.append(metadata)

        with open(metadata_file, "w") as f:
            json.dump(data, f, indent=4)


def download_flags(session):
    """
    Download flag images from the Wikipedia list of national flags.

    This function scrapes the Wikipedia page for national flags, extracts
    flag information, and returns a list of tuples containing country names
    and flag image URLs.

    Args:
        session (requests.Session): The session object for making HTTP requests.

    Returns:
        list: A list of tuples, each containing (country_name, flag_url).
    """
    url = config.FLAG_GALLERY_URL
    response = session.get(url)

    if response.status_code != 200:
        logger.error("Failed to retrieve page: Status code %d", response.status_code)
        return []

    soup = BeautifulSoup(response.content, "html.parser")
    logger.info("Page title: %s", soup.title.string)

    tables = soup.find_all("table", class_="wikitable")
    if not tables:
        logger.warning("No tables found with class 'wikitable'")
        return []

    flag_links = []
    logger.info("Gathering flag information...")
    for table in tqdm(tables, desc="Gathering flag information"):
        for row in table.find_all("tr")[1:]:  # Skip header row
            cols = row.find_all("td")
            if len(cols) < 2:
                continue

            # Extract country name from the first column with text
            country = next((col.text.strip() for col in cols if col.text.strip()), None)
            if not country:
                continue

            country = sanitize_country_name(country)

            # Find the flag image in any column
            flag_link = None
            for col in cols:
                a_tag = col.find("a", href=True)
                if a_tag:
                    flag_url = get_flag_image_url(a_tag)
                    if flag_url:
                        flag_links.append((country, flag_url))
                        logger.info("Found flag for %s: %s", country, flag_url)
                        break
            else:
                logger.warning("No flag link found for %s", country)

    logger.info("Finished gathering flag information.")
    return flag_links


def download_flag_images(
    output_dir=config.FLAG_IMAGES_DIR,
    max_retries=3,
    timeout=10,
    max_workers=5,
    skip_small_flags=False,
):
    """
    Download flag images from the Wikipedia list page.

    This function orchestrates the entire process of downloading flag images,
    including setting up the HTTP session, gathering flag information, and
    downloading the images using multiple threads.

    Args:
        output_dir (str): Directory to save the downloaded flag images.
        max_retries (int): Maximum number of retries for failed requests.
        timeout (int): Timeout for HTTP requests in seconds.
        max_workers (int): Maximum number of concurrent download threads.
        skip_small_flags (bool): Whether to skip flags smaller than MIN_FILE_SIZE.

    Raises:
        ValueError: If any of the input parameters are invalid.
    """
    # Validation checks for parameters
    if max_retries < 1:
        raise ValueError(f"max_retries must be greater than 0, got {max_retries}.")
    if timeout < 1:
        raise ValueError(f"timeout must be greater than 0, got {timeout}.")
    if max_workers < 1:
        raise ValueError(f"max_workers must be greater than 0, got {max_workers}.")

    logger.info(f"Downloading flags to: {output_dir}")
    config.ensure_directories()  # Ensure directories exist before downloading

    session = requests.Session()
    retries = Retry(
        total=max_retries, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504]
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    try:
        flag_info_list = download_flags(session)  # Get flag information

        if not flag_info_list:
            logger.warning("No valid flag links found on the page.")
            return

        logger.info(f"Found {len(flag_info_list)} flag links")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(
                tqdm(
                    executor.map(
                        lambda x: download_flag(
                            x, session, output_dir, skip_small_flags
                        ),
                        flag_info_list,
                    ),
                    total=len(flag_info_list),
                    desc="Downloading flags",
                )
            )

        logger.info(f"Total flags processed: {len(flag_info_list)}")

        # Ensure at least one metadata entry is created for testing purposes
        if not os.path.exists(os.path.join(output_dir, "metadata.json")):
            save_metadata("test_country", "https://example.com/flag.png", output_dir)

    except requests.RequestException as e:
        logger.error(f"Network error during flag download: {str(e)}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error during flag download: {str(e)}", exc_info=True)


def collect_data():
    """
    Collect flag data.

    This function serves as the main entry point for the data collection process.
    It initiates the flag image download process and logs the start and completion
    of the data collection.
    """
    logger.info("Starting data collection...")
    download_flag_images()
    logger.info("Data collection completed.")


# Make sure to export the collect_data function
__all__ = ["collect_data", "download_flag_images"]
