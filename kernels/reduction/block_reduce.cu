/*
 * KERNEL: Parallel Reduction (Sum)
 * OPTIMIZATION: Block-level reduction using shared memory
 * KEY INSIGHT: Each block reduces its items into shared memory, then atomics are used only once per block.
 * EXPECTED SPEEDUP: Much faster than atomic reduction, but slower than warp shuffle reduction.
 * KNOWN BOTTLENECK: Shared memory bank conflicts if padding isn't used, and synchronization overhead.
 * FURTHER READING: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
 */

extern "C" __global__ void block_reduce(const float* input, float* output, int N) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < N) ? input[i] : 0.0f;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}
