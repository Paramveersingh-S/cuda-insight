# Getting Started with CUDA Insight

## Prerequisites
- Python 3.9+
- NVIDIA GPU with `nvidia-smi` in PATH
- CUDA Toolkit (`nvcc`)
- Nsight Compute (`ncu`)

## Installation
```bash
pip install -e ".[dev]"
```

## First Profile
1. Check GPU Info: `cuda-insight gpu-info`
2. Compile and see PTX: `cuda-insight ptx kernels/matmul/naive.cu --annotate`
3. Profile: `nvcc kernels/matmul/naive.cu -o naive_matmul` and then `cuda-insight profile naive_matmul --html --roofline`
