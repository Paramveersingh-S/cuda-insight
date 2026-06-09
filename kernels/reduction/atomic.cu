/*
 * KERNEL: Parallel Reduction (Sum)
 * OPTIMIZATION: Atomic additions (Naive approach)
 * KEY INSIGHT: Extremely simple to write, but serializes execution on the atomic lock, leading to terrible contention.
 * EXPECTED SPEEDUP: Slower than CPU for very large arrays.
 * KNOWN BOTTLENECK: High atomic contention on a single global memory address.
 * FURTHER READING: https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/
 */

extern "C" __global__ void atomic_reduce(const float* input, float* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        atomicAdd(output, input[tid]);
    }
}
