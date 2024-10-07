# Flag Prediction Application

## Overview

This application predicts the name of a country (or countries) based on an input flag image. It uses advanced image processing techniques and deep learning models built with PyTorch to classify flags accurately.

## Table of Contents

- [Flag Prediction Application](#flag-prediction-application)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Project Structure](#project-structure)
  - [Setup](#setup)
    - [Prerequisites](#prerequisites)
    - [Steps](#steps)
  - [Usage](#usage)
    - [Command Line Interface](#command-line-interface)
  - [Running Tests](#running-tests)
  - [Docker Deployment](#docker-deployment)
  - [Conda Environment Setup](#conda-environment-setup)
  - [Configuration](#configuration)
  - [Logs and File Access](#logs-and-file-access)
  - [Troubleshooting](#troubleshooting)
  - [Contributing](#contributing)
  - [License](#license)

## Features

- **Automated Flag Image Collection**: Downloads flag images from Wikipedia automatically.
- **Advanced Image Processing**: Preprocesses images for optimal model input.
- **Multiple Deep Learning Models**: Supports ResNet50, MobileNetV2, and EfficientNet-B0 models built with PyTorch for flag classification.
- **Training Pipeline**: Trains the models using processed and augmented flag images.
- **Inference Pipeline**: Predicts the country from a given flag image using the trained models.
- **Command Line Interface (CLI)**: Provides an easy-to-use interface for all functionalities.
- **Extensive Data Augmentation**: Implements flag-specific data augmentation techniques to generate multiple augmented images per flag.
- **Detailed Logging**: Provides comprehensive logs of the training and prediction processes.
- **Transfer Learning**: Utilizes transfer learning for improved performance.
- **Cross-validation**: Implements k-fold cross-validation for more robust model evaluation.
- **Mixed Precision Training**: Utilizes mixed precision training for faster computation on compatible GPUs.
- **Batch Processing and Multiprocessing**: Improves efficiency in image processing and data augmentation.

## Project Structure

```sh
flag_prediction_project/
├── data/
│   ├── flag_images/       # Downloaded flag images
│   ├── processed/         # Processed images
├── logs/                  # Log files
├── models/                # Saved models and label encoders
├── src/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── data_collection.py
│   ├── data_utils.py
│   ├── image_processing.py
│   ├── main.py
│   ├── model.py
│   ├── predict.py
│   ├── train.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_cli.py
│   ├── test_data_collection.py
│   ├── test_data_utils.py
│   ├── test_image_processing.py
│   ├── test_model.py
│   ├── test_predict.py
│   ├── test_train.py
├── .dockerignore
├── Conda_Environment_Setup.md
├── Dockerfile
├── nginx.conf
├── pyproject.toml
├── README.md
├── requirements-dev.txt
├── requirements.txt
├── run.py
├── setup.cfg
```

## Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- PyTorch 1.9.0 or higher with CUDA support (recommended for faster training)
- NVIDIA GPU (optional but recommended for training and inference)
- Docker (optional but recommended for reproducibility and ease of use)
  
### Steps

1. **Clone the Repository**

   ```sh
   git clone https://github.com/Xza85hrf/flag_prediction_project.git
   cd flag_prediction_project
   ```

2. **Create a Virtual Environment** (Optional but Recommended)

   ```sh
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

3. **Install Dependencies**

   ```sh
   pip install -r requirements-dev.txt
   ```

   **Note:** Ensure that PyTorch is installed with CUDA support if you have a compatible GPU. You can install PyTorch with CUDA by following instructions on the [PyTorch website](https://pytorch.org/get-started/locally/).

4. **Optional: Install the Package**

   ```sh
   pip install -e .
   ```

## Usage

### Command Line Interface

The application provides several commands through its Command Line Interface (CLI). Below are detailed explanations of each command, including what they do, what they affect, and any potential side effects:

1. **Download Flag Images**

   Python:

   ```sh
   python run.py download
   ```

   Docker:

   ```sh
   docker run --gpus all -it --rm -v ${PWD}/data:/app/data --memory=24g --cpus=16 --shm-size=50g flag_prediction_app download
   ```

   Example:

   ```sh
   docker run --gpus all -it --rm -v /home/user/flag_project/data:/app/data --memory=24g --cpus=16 --shm-size=50g flag_prediction_app download
   ```

   This command downloads flag images from Wikipedia. It populates the `data/flag_images/` directory with the downloaded images. This step is necessary before processing or training.

   **Effects**: Creates new files in the `data/flag_images/` directory.
   **Side effects**: May overwrite existing files if run multiple times.

2. **Process and Augment Images**

   Python:

   ```sh
   python run.py process --duplicate-times <number>
   ```

   Docker:

   ```sh
   docker run --gpus all -it --rm -v ${PWD}/data:/app/data --memory=24g --cpus=16 --shm-size=50g flag_prediction_app process --duplicate-times <number>
   ```

   Example:

   ```sh
   python run.py process --duplicate-times 50
   docker run --gpus all -it --rm -v /home/user/flag_project/data:/app/data --memory=24g --cpus=16 --shm-size=50g flag_prediction_app process --duplicate-times 50
   ```

   This command processes the downloaded flag images and creates augmented versions. The `--duplicate-times` parameter specifies how many augmented versions to create for each original image.

   **Effects**: Creates processed and augmented images in the `data/processed/` directory.
   **Side effects**: May overwrite existing processed images if run multiple times.

3. **Train the Model**

   Python:

   ```sh
   python run.py train [--cross-validate] [--models <model1> --models <model2> ...]
   ```

   Docker:

   ```sh
   docker run --gpus all -it --rm -v ${PWD}/data:/app/data -v ${PWD}/models:/app/models -v ${PWD}/logs:/app/logs --memory=24g --cpus=16 --shm-size=50g flag_prediction_app train [--cross-validate] [--models <model1> --models <model2> ...]
   ```

   Examples:

   ```sh
   python run.py train --cross-validate # trains all of the models
   python run.py train --models resnet50 --models efficientnet_b0

   docker run --gpus all -it --rm -v ${PWD}/data:/app/data -v ${PWD}/models:/app/models -v ${PWD}/logs:/app/logs --memory=24g --cpus=16 --shm-size=50g flag_prediction_app train --cross-validate # trains all of the models
   docker run --gpus all -it --rm -v ${PWD}/data:/app/data -v ${PWD}/models:/app/models -v ${PWD}/logs:/app/logs --memory=24g --cpus=16 --shm-size=50g flag_prediction_app train --models resnet50 --models efficientnet_b0
   ```

   This command trains the specified model(s) on the processed flag images.

   - Use `--cross-validate` to perform k-fold cross-validation for more robust model evaluation.
   - Use `--models` to specify which models to train. You can specify multiple models.

   **Effects**: Creates trained model files in the `models/` directory and generates training logs.
   **Side effects**: Overwrites existing model files with the same names.
   **Side effects**: May take a long time to run depending on the model(s) and dataset

   **Important Note**: Do NOT use the `--multi-label` argument for this dataset. This argument is intended for datasets with multiple labels per image and does not apply to this flag prediction task.

4. **Predict a Country from a Flag Image**

   Python:

   ```sh
   python run.py predict <path_to_flag_image> [--model <model_name>]
   ```

   Docker:

   ```sh
   docker run --gpus all -it --rm -v ${PWD}/data:/app/data -v ${PWD}/models:/app/models -v ${PWD}/images:/app/images --memory=24g --cpus=16 --shm-size=50g flag_prediction_app predict /app/images/<image_name> [--model <model_name>]
   ```

   Examples:

   ```sh
   python run.py predict path/to/flag.jpg
   python run.py predict path/to/flag.jpg --model mobilenet_v2

   docker run --gpus all -it --rm -v ${PWD}/data:/app/data -v ${PWD}/models:/app/models -v /home/user/flag_images:/app/images --memory=24g --cpus=16 --shm-size=50g flag_prediction_app predict /app/images/flag.jpg
   docker run --gpus all -it --rm -v ${PWD}/data:/app/data -v ${PWD}/models:/app/models -v /home/user/flag_images:/app/images --memory=24g --cpus=16 --shm-size=50g flag_prediction_app predict /app/images/flag.jpg --model mobilenet_v2
   ```

   This command uses a trained model to predict the country for a given flag image.

   - Specify the path to the flag image you want to predict.
   - Use `--model` to choose a specific model for prediction (optional).

   **Effects**: Outputs the prediction result to the console and logs.
   **Side effects**: None.

5. **Full Pipeline**

   Python:

   ```sh
   python run.py full-pipeline --duplicate-times <number> [--cross-validate] [--models <model1> --models <model2> ...]
   ```

   Docker:

   ```sh
   docker run --gpus all -it --rm -v ${PWD}/data:/app/data -v ${PWD}/models:/app/models -v ${PWD}/logs:/app/logs --memory=24g --cpus=16 --shm-size=50g flag_prediction_app full-pipeline --duplicate-times <number> [--cross-validate] [--models <model1> --models <model2> ...]
   ```

   Examples:

   ```sh
   python run.py full-pipeline --duplicate-times 100 --cross-validate # trains all of the models
   python run.py full-pipeline --duplicate-times 100 --models resnet50 --models mobilenet_v2 --models efficientnet_b0

   docker run --gpus all -it --rm -v ${PWD}/data:/app/data -v ${PWD}/models:/app/models -v ${PWD}/logs:/app/logs --memory=24g --cpus=16 --shm-size=50g flag_prediction_app full-pipeline --duplicate-times 100 --cross-validate   # trains all of the models

   docker run --gpus all -it --rm -v ${PWD}/data:/app/data -v ${PWD}/models:/app/models -v ${PWD}/logs:/app/logs --memory=24g --cpus=16 --shm-size=50g flag_prediction_app full-pipeline --duplicate-times 100 --models resnet50 --models mobilenet_v2 --models efficientnet_b0
   ```

   This command runs the entire pipeline: downloading images, processing them, and training the model(s).

   - `--duplicate-times` specifies the number of augmented images to create.
   - Use `--cross-validate` for cross-validation during training.
   - Specify models with `--models` as in the train command.

   **Effects**: Downloads images, processes them, trains models, and generates all associated files and logs.
   **Side effects**: May overwrite existing files in data, models, and logs directories.

**Important Note** Do not use the `--multi-label` argument as it does not apply to this dataset.

## Running Tests

To run all unit tests, execute:

```sh
pytest
```

To run a specific test file:

```sh
pytest tests/test_file_name.py
```

To run a specific test function:

```sh
pytest tests/test_file_name.py::test_function_name
```

For example, to run the `test_download_flag_images` test in the `test_data_collection.py` file:

```sh
pytest tests/test_data_collection.py::test_download_flag_images
```

## Docker Deployment

1. **Build the Docker Image**

   ```sh
   docker build -t flag_prediction_app .
   ```

2. **Run the Docker Container**

   Basic command structure:

   ```bash
   docker run [options] flag_prediction_app [command] [arguments]
   ```

   Example for training with cross-validation:

   ```bash
   docker run --gpus all -it --rm -v ${PWD}/data:/app/data -v ${PWD}/models:/app/models -v ${PWD}/logs:/app/logs --memory=24g --cpus=16 --shm-size=50g flag_prediction_app train --cross-validate
   ```

   This command:
   - Uses all available GPUs (`--gpus all`)
   - Runs the container interactively (`-it`)
   - Removes the container after it exits (`--rm`)
   - Mounts the local `data`, `models`, and `logs` directories
   - Limits memory usage to 24GB (`--memory=24g`)
   - Limits CPU usage to 16 cores (`--cpus=16`)
   - Sets shared memory size to 50GB (`--shm-size=50g`)
   - Runs the training with cross-validation

   After training, check the contents of the models directory:

   ```bash
   ls -l ./models
   ```

3. **Other Docker Examples**

   Download flag images:

   ```bash
   docker run --gpus all -it --rm -v ${PWD}/data:/app/data --memory=24g --cpus=16 --shm-size=50g flag_prediction_app download
   ```

   Process images:

   ```bash
   docker run --gpus all -it --rm -v ${PWD}/data:/app/data --memory=24g --cpus=16 --shm-size=50g flag_prediction_app process --duplicate-times 50
   ```

   Predict using a specific model:

   ```bash
   docker run --gpus all -it --rm -v ${PWD}/data:/app/data -v ${PWD}/models:/app/models -v /path/to/image/directory:/app/images --memory=24g --cpus=16 --shm-size=50g flag_prediction_app predict /app/images/flag.jpg --model resnet50
   ```

   Run full pipeline:

   ```bash
   docker run --gpus all -it --rm -v ${PWD}/data:/app/data -v ${PWD}/models:/app/models -v ${PWD}/logs:/app/logs --memory=24g --cpus=16 --shm-size=50g flag_prediction_app full-pipeline --duplicate-times 100 --cross-validate --models resnet50 --models efficientnet_b0 --models mobilenet_v2
   ```

## Conda Environment Setup

For detailed instructions on setting up a Conda environment for this project, please refer to the [Conda_Environment_Setup.md](Conda_Environment_Setup.md) file.

## Configuration

You can modify settings such as image size, model parameters, and data augmentation options in `src/config.py`. Key configurations include:

- `MODELS_TO_TRAIN`: List of models to train (e.g., `["resnet50", "mobilenet_v2", "efficientnet_b0"]`)
- `AUGMENTATION_FACTOR`: Number of augmented images to create per original image
- `IMAGE_SIZE`: Input image size for the models
- `INITIAL_LEARNING_RATE`: Learning rate for training
- `BATCH_SIZE`: Batch size for training
- `EPOCHS`: Number of training epochs
- Various data augmentation parameters

## Logs and File Access

Logs are saved in the `logs/` directory. To access and view files via Docker:

1. List files in a directory:

   ```sh
   docker run -it --rm -v ${PWD}/logs:/app/logs alpine ls -l /app/logs
   ```

2. View content of a specific file:

   ```sh
   docker run -it --rm -v ${PWD}/logs:/app/logs alpine cat /app/logs/flag_prediction.log
   ```

3. Start an interactive shell:

   ```sh
   docker run -it --rm -v ${PWD}/models:/app/models -v ${PWD}/logs:/app/logs -v ${PWD}/data:/app/data alpine sh
   ```

4. View images using Nginx:

   ```sh
   docker run -d --rm -p 8080:80 -v ${PWD}/data:/usr/share/nginx/html -v ${PWD}/nginx.conf:/etc/nginx/conf.d/default.conf --name data_server nginx
   ```

   Then open a web browser and go to `http://localhost:8080`.

To stop the Nginx server:

```sh
docker stop data_server
```

You can also view files in the `models/` and `logs/` directories using similar commands, replacing `data` with `models` or `logs` as needed.

## Troubleshooting

- **PyTorch Installation Issues**: Refer to the [official installation guide](https://pytorch.org/get-started/locally/) for your system.
- **GPU Memory Issues**: Try reducing the batch size in `config.py` or use a GPU with more memory.
- **Long Training Times**: Consider using a GPU or adjusting the `AUGMENTATION_FACTOR` if needed.
- **Cross-validation Issues**: If cross-validation takes too long, consider reducing the `N_SPLITS` in `config.py`.
- **CUDA Compatibility**: Ensure that your PyTorch installation matches your CUDA version for GPU acceleration.

## Contributing

## License
