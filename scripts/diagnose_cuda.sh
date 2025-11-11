#!/bin/bash
# Script to diagnose CUDA and GPU configuration issues
# Run this to check your system setup before training

echo "=== CUDA and GPU Diagnostic Script ==="
echo ""

echo "1. Checking NVIDIA Driver Version:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=driver_version,name,compute_cap --format=csv,noheader
    echo ""
    echo "Full nvidia-smi output:"
    nvidia-smi
else
    echo "ERROR: nvidia-smi not found. NVIDIA driver may not be installed."
fi
echo ""

echo "2. Checking CUDA Toolkit Version:"
if command -v nvcc &> /dev/null; then
    nvcc --version
else
    echo "CUDA toolkit not found in PATH"
fi
echo ""

echo "3. Checking PyTorch CUDA Configuration:"
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version (compiled): {torch.version.cuda}')
    print(f'Number of GPUs: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('WARNING: CUDA is not available to PyTorch')
    print(f'Built with CUDA: {torch.version.cuda}')
" 2>&1
echo ""

echo "4. Checking System CUDA Libraries:"
ldconfig -p | grep -i cuda | head -n 10
echo ""

echo "5. Environment Variables:"
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA_PATH: $CUDA_PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""

echo "6. Checking for CUDA Runtime Libraries:"
python3 -c "
import ctypes
import os

cuda_paths = [
    '/usr/local/cuda/lib64/libcudart.so',
    '/usr/lib/x86_64-linux-gnu/libcudart.so',
]

for path in cuda_paths:
    if os.path.exists(path):
        print(f'Found: {path}')
        try:
            lib = ctypes.CDLL(path)
            print(f'  Successfully loaded')
        except:
            print(f'  Failed to load')
"
echo ""

echo "=== Diagnostic Complete ==="
echo ""
echo "Common Issues and Solutions:"
echo "1. If NVIDIA driver is too old (< 450.80 for CUDA 11.x):"
echo "   - Update NVIDIA driver: sudo apt update && sudo apt install nvidia-driver-XXX"
echo ""
echo "2. If PyTorch CUDA version doesn't match system CUDA:"
echo "   - Reinstall PyTorch with correct CUDA version"
echo "   - Visit: https://pytorch.org/get-started/locally/"
echo ""
echo "3. If driver is newer than PyTorch CUDA (forward compatibility issue):"
echo "   - Either downgrade driver OR upgrade PyTorch"
echo "   - Check compatibility: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/"

