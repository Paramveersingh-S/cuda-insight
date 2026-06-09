# Matrix Multiplication Kernels

This directory contains reference implementations of Single-Precision Matrix Multiplication (SGEMM).
- `naive.cu`: Unoptimized global memory bound version.
- `tiled.cu`: Optimized version using shared memory tiling to reduce global memory bandwidth usage.
- `cublas_ref.cu`: Marker file representing the cuBLAS highly-tuned implementation.
