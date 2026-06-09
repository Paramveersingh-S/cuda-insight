/*
 * KERNEL: INT8 Matrix Multiplication (Quantization)
 * OPTIMIZATION: Use of `dp4a` instruction
 * KEY INSIGHT: Packs 4 INT8 values into a 32-bit register and computes 4 multiply-accumulates in a single cycle.
 * EXPECTED SPEEDUP: Up to 4x peak throughput over FP32 on supported architectures (sm_61+).
 * KNOWN BOTTLENECK: Quantization/dequantization overhead if not done beforehand.
 * FURTHER READING: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#dp4a
 */

#include <stdint.h>

extern "C" __global__ void int8_gemm(const int8_t* A, const int8_t* B, int* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        int sum = 0;
        
        // Process 4 elements at a time
        for (int i = 0; i < K / 4; ++i) {
            // Read 4 bytes as a 32-bit integer
            int a_val = *((const int*)(&A[row * K + i * 4]));
            int b_val = *((const int*)(&B[i * 4 * N + col])); 
            // Note: B needs to be packed properly for coalesced dp4a in a real scenario
            
            // Perform 4-way dot product accumulate
            sum = __dp4a(a_val, b_val, sum);
        }
        
        C[row * N + col] = sum;
    }
}
