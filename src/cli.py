import click
from .data_collection import collect_data
from .image_processing import process_images
from .train import train_models
from . import config
from .predict import predict_country
import sys
import logging
import os

# Initialize logger for this module
logger = logging.getLogger(__name__)


def setup_logging():
    """
    Set up logging configuration for the application.

    This function configures logging to write to both a file and the console.
    The log file is stored in the directory specified by config.LOG_DIR.
    """
    log_file = os.path.join(config.LOG_DIR, "flag_prediction.log")
    logging.basicConfig(
        filename=log_file,
        level=config.LOG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Add console handler for CLI output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(config.LOG_LEVEL)
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)


@click.group()
def cli():
    """
    Main CLI group for the flag prediction project.

    This function serves as the entry point for the CLI application.
    It sets up logging before executing any commands.
    """
    setup_logging()
    pass


@cli.command()
def download():
    """
    Command to download flag images.

    This function initiates the process of downloading flag images from a predefined source.
    It ensures necessary directories exist before starting the download.
    """
    click.echo("Starting flag image download...")
    logger.info("Starting flag image download...")
    config.ensure_directories()
    collect_data()


@cli.command()
@click.option(
    "--duplicate-times",
    "-d",
    default=config.AUGMENTATION_FACTOR,
    type=click.IntRange(min=1),
    help="Number of times to duplicate each image for augmentation",
)
def process(duplicate_times):
    """
    Command to process collected images.

    This function initiates the image processing pipeline, including any augmentation.

    Args:
        duplicate_times (int): Number of times to duplicate each image for augmentation.
    """
    click.echo(f"Starting image processing with {duplicate_times} duplications...")
    logger.info(f"Starting image processing with {duplicate_times} duplications...")
    config.ensure_directories()
    try:
        process_images(duplicate_times)
        click.echo("Image processing completed successfully.")
        logger.info("Image processing completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred during image processing: {str(e)}")
        click.echo(f"An error occurred during image processing: {str(e)}", err=True)
    finally:
        click.echo("Exiting image processing.")
        logger.info("Exiting image processing.")


@cli.command()
@click.option("--multi-label", is_flag=True, help="Use multi-label classification")
@click.option("--cross-validate", is_flag=True, help="Perform cross-validation")
@click.option(
    "--models",
    "-m",
    multiple=True,
    type=click.Choice(config.MODELS_TO_TRAIN),
    help="Select models to train (can be used multiple times)",
)
def train(multi_label, cross_validate, models):
    """
    Command to train the model(s).

    This function initiates the training process for the specified models.

    Args:
        multi_label (bool): Whether to use multi-label classification.
        cross_validate (bool): Whether to perform cross-validation.
        models (tuple): Tuple of model names to train. If empty, all models in config.MODELS_TO_TRAIN will be trained.
    """
    click.echo("Starting model training...")
    logger.info("Starting model training...")
    click.echo(f"Multi-label: {multi_label}")
    click.echo(f"Cross-validate: {cross_validate}")
    if models:
        click.echo(f"Training models: {', '.join(models)}")
        config.MODELS_TO_TRAIN = list(models)
    else:
        click.echo(f"Training all models: {', '.join(config.MODELS_TO_TRAIN)}")
    logger.info(
        f"Training configuration: Multi-label: {multi_label}, Cross-validate: {cross_validate}, Models: {config.MODELS_TO_TRAIN}"
    )
    try:
        train_models(multi_label=multi_label, cross_validate=cross_validate)
        click.echo("Model training completed successfully.")
        logger.info("Model training completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred during model training: {str(e)}")
        click.echo(f"An error occurred during model training: {str(e)}", err=True)


@cli.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--multi-label", is_flag=True, help="Use multi-label classification")
@click.option(
    "--model",
    type=click.Choice(config.MODELS_TO_TRAIN),
    default="efficientnet_b0",
    help="Select model for prediction",
)
def predict(image_path, multi_label, model):
    """
    Command to predict country from a flag image.

    This function uses a trained model to predict the country (or countries) for a given flag image.

    Args:
        image_path (str): Path to the image file to predict.
        multi_label (bool): Whether to use multi-label classification.
        model (str): Name of the model to use for prediction.
    """
    try:
        countries = predict_country(
            image_path, multi_label=multi_label, model_type=model
        )
        if countries:
            result = (
                f"Predicted countries for image {image_path} using {model}: {', '.join(countries)}"
                if multi_label
                else f"Predicted country for image {image_path} using {model}: {countries[0]}"
            )
            click.echo(result)
            logger.info(result)
            # Log prediction to a separate file
            with open(os.path.join(config.LOG_DIR, "prediction_results.log"), "a") as f:
                f.write(f"{result}\n")
        else:
            click.echo("Failed to predict the country.")
            logger.warning("Failed to predict the country.")
    except Exception as e:
        logger.error(f"An error occurred during prediction: {str(e)}")
        click.echo(f"An error occurred during prediction: {str(e)}", err=True)


@cli.command()
@click.option(
    "--duplicate-times",
    default=config.AUGMENTATION_FACTOR,
    help="Number of times to duplicate each image for augmentation",
)
@click.option("--multi-label", is_flag=True, help="Use multi-label classification")
@click.option("--cross-validate", is_flag=True, help="Perform cross-validation")
@click.option(
    "--models",
    "-m",
    multiple=True,
    type=click.Choice(config.MODELS_TO_TRAIN),
    help="Select models to train (can be used multiple times)",
)
def full_pipeline(duplicate_times, multi_label, cross_validate, models):
    """
    Command to run the full pipeline: download, process, and train.

    This function executes the entire workflow from data collection to model training.

    Args:
        duplicate_times (int): Number of times to duplicate each image for augmentation.
        multi_label (bool): Whether to use multi-label classification.
        cross_validate (bool): Whether to perform cross-validation.
        models (tuple): Tuple of model names to train. If empty, all models in config.MODELS_TO_TRAIN will be trained.
    """
    click.echo("Starting full pipeline...")
    logger.info("Starting full pipeline...")
    config.ensure_directories()
    try:
        collect_data()
        process_images(duplicate_times)
        if models:
            click.echo(f"Training models: {', '.join(models)}")
            config.MODELS_TO_TRAIN = list(models)
        else:
            click.echo(f"Training all models: {', '.join(config.MODELS_TO_TRAIN)}")
        logger.info(f"Training models: {config.MODELS_TO_TRAIN}")
        train_models(multi_label=multi_label, cross_validate=cross_validate)
        click.echo("Full pipeline completed successfully.")
        logger.info("Full pipeline completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred during the full pipeline: {str(e)}")
        click.echo(f"An error occurred during the full pipeline: {str(e)}", err=True)
    finally:
        click.echo("Exiting full pipeline.")
        logger.info("Exiting full pipeline.")
        sys.exit(0)


if __name__ == "__main__":
    cli()
