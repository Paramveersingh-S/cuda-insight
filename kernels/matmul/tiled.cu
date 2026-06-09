/*
 * KERNEL: Tiled Matrix Multiplication
 * OPTIMIZATION: Shared memory tiling (TILE_SIZE x TILE_SIZE)
 * KEY INSIGHT: Reduces global memory reads from O(N^3) to O(N^3 / TILE_SIZE)
 * EXPECTED SPEEDUP: 10-20x over naive on modern GPUs
 * KNOWN BOTTLENECK: Still memory-bandwidth bound for very large tiles, potential bank conflicts if not padded.
 * FURTHER READING: https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
 */

#define TILE_SIZE 32

extern "C" __global__ void tiled_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            sA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        if (t * TILE_SIZE + threadIdx.y < K && col < N)
            sB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int M = 1024;
    int N = 1024;
    int K = 1024;
    size_t bytes = M * N * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    for (int i = 0; i < M * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

    tiled_matmul<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
