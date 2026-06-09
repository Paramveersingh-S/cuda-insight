/*
 * KERNEL: Parallel Reduction (Sum)
 * OPTIMIZATION: Warp-level reduction primitives
 * KEY INSIGHT: Uses `__shfl_down_sync` to perform fast reductions within a 32-thread warp without shared memory.
 * EXPECTED SPEEDUP: 2-3x over shared memory block reduction, 100x+ over atomic reduce.
 * KNOWN BOTTLENECK: Still memory bandwidth bound for large inputs.
 * FURTHER READING: https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
 */

__device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

extern "C" __global__ void warp_reduce(const float* input, float* output, int N) {
    float sum = 0.0f;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Grid-stride loop to handle N > grid size
    for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }
    
    sum = warpReduceSum(sum);
    
    if (threadIdx.x % warpSize == 0) {
        atomicAdd(output, sum);
    }
}
