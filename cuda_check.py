"""
CUDA and PyTorch Information Script

This script provides detailed information about the PyTorch installation,
CUDA availability, and GPU devices (if present). It also demonstrates
a simple tensor operation on the GPU if CUDA is available.
"""

import torch

def main():
    """
    Main function to display PyTorch and CUDA information,
    and perform a sample GPU operation if available.
    """
    # Display PyTorch and CUDA information
    pytorch_version = torch.__version__
    cuda_available = torch.cuda.is_available()
    cuda_version = torch.version.cuda

    print(f"PyTorch Version: {pytorch_version}")
    print(f"CUDA Available: {cuda_available}")
    print(f"CUDA Version: {cuda_version}")

    if cuda_available:
        display_cuda_info()
        perform_sample_gpu_operation()
    else:
        print("No CUDA devices found. Running on CPU.")

def display_cuda_info():
    """
    Display information about available CUDA devices,
    including device name and memory usage.
    """
    cuda_device_count = torch.cuda.device_count()
    print(f"Number of CUDA Devices: {cuda_device_count}")

    for i in range(cuda_device_count):
        device_name = torch.cuda.get_device_name(i)
        # Convert bytes to megabytes for readability
        memory_allocated = torch.cuda.memory_allocated(i) / (1024 ** 2)
        memory_reserved = torch.cuda.memory_reserved(i) / (1024 ** 2)
        print(f"Device {i}: {device_name}")
        print(f"  - Memory Allocated: {memory_allocated:.2f} MB")
        print(f"  - Memory Reserved: {memory_reserved:.2f} MB")

def perform_sample_gpu_operation():
    """
    Perform a sample tensor operation on the GPU to demonstrate CUDA functionality.
    """
    # Use the first available CUDA device
    device = torch.device("cuda")
    # Create a random 5x3 tensor and move it to the GPU
    x = torch.rand(5, 3).to(device)
    print(f"Sample Tensor on GPU: \n{x}")

if __name__ == "__main__":
    main()
