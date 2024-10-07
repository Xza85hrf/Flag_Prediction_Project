# Build stage
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04 AS builder

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies and Python 3.8
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-dev \
    build-essential \
    libffi-dev \
    libssl-dev \
    python3.8 \
    python3.8-dev \
    python3-pip \
    python3-setuptools \
    libcairo2-dev \
    libgirepository1.0-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools, install Python dependencies
COPY requirements.txt requirements-dev.txt /app/
RUN python3.8 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3.8 -m pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

# Install PyTorch with CUDA support in the build stage
RUN python3.8 -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy the application code and install the project
COPY . /app
RUN python3.8 -m pip install --no-cache-dir .

# Final stage
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

# Install only the necessary runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-dev \
    python3.8 \
    python3-pip \
    libcairo2 \
    libgirepository1.0-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages and application files from the builder
COPY --from=builder /usr/local/lib/python3.8/ /usr/local/lib/python3.8/
COPY --from=builder /app /app

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/logs

EXPOSE 8080

ENTRYPOINT ["python3.8", "run.py"]