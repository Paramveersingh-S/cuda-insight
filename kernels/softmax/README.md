# Softmax Kernels

This directory contains implementations of the Softmax operation.
- `naive.cu`: Two-pass implementation (find max, then sum exponentials).
- `online_softmax.cu`: Single-pass numerically stable implementation using the online normalizer algorithm.
