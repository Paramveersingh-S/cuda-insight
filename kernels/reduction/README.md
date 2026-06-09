# Parallel Reduction Kernels

This directory contains reference implementations of Parallel Sum Reduction.
- `atomic.cu`: Naive version using `atomicAdd` on global memory. (Very slow)
- `block_reduce.cu`: Standard shared memory tree-reduction.
- `warp_reduce.cu`: Highly optimized version using Kepler+ `__shfl_down_sync` warp primitives.
