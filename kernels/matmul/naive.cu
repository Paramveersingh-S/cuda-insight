/*
 * KERNEL: Naive Matrix Multiplication
 * OPTIMIZATION: None (Global memory only)
 * KEY INSIGHT: Each thread computes one element of C, reading an entire row of A and column of B from slow global memory.
 * EXPECTED SPEEDUP: Baseline (1x)
 * KNOWN BOTTLENECK: Heavily memory-bandwidth bound. O(N^3) global memory reads.
 * FURTHER READING: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
 */

extern "C" __global__ void naive_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}
