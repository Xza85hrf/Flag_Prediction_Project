# Conda Environment Setup for Flag Prediction Project

## 1. Create a new Conda environment

```bash
conda create -n flag_prediction python=3.10
```

This creates a new environment named `flag_prediction` with Python 3.10.

## 2. Activate the environment

```bash
conda activate flag_prediction
```

## 3. Install core dependencies

```bash
conda install -c conda-forge numpy pandas scikit-learn matplotlib seaborn tqdm click pillow requests beautifulsoup4
```

## 4. Install PyTorch (CPU version)

If you don't have a CUDA-capable GPU, use this command:

```bash
conda install pytorch torchvision cpuonly -c pytorch
```

For CUDA 11.7 (adjust based on your CUDA version):

```bash
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

## 5. Install additional dependencies

```bash
pip install opencv-python-headless scikit-image imbalanced-learn
```

## 6. Install development tools

```bash
conda install -c conda-forge pytest black isort flake8
```

## 7. Set up the project

Navigate to your project directory and install it in editable mode:

```bash
cd path/to/flag_prediction_project
pip install -e .
```

## 8. Create a requirements file

Create a `requirements.txt` file in your project directory:

```bash
pip freeze > requirements.txt
```

## 9. Test the installation

Run a simple Python command to check if everything is installed correctly:

```bash
python -c "import torch; import numpy; import pandas; import sklearn; import matplotlib; import seaborn; import cv2; print('All packages imported successfully!')"
```

If this runs without errors, your environment is set up correctly.

## 10. Running the project

Now you can run your project using:

```bash
python run.py
```

Or use the CLI command:

```bash
flag_predict
```

Remember to always activate the Conda environment (`conda activate flag_prediction`) before working on your project.
