/*
 * KERNEL: Softmax
 * OPTIMIZATION: None (Two-pass approach)
 * KEY INSIGHT: Computes max first, then exponentiates and sums, requiring multiple global memory passes.
 * EXPECTED SPEEDUP: Baseline
 * KNOWN BOTTLENECK: High memory bandwidth usage due to reading/writing the input twice.
 * FURTHER READING: https://arxiv.org/abs/1805.02867 (Online Normalizer Calculation)
 */

#include <math.h>

extern "C" __global__ void naive_softmax(const float* input, float* output, int num_rows, int num_cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float max_val = -INFINITY;
        for (int i = 0; i < num_cols; ++i) {
            if (input[row * num_cols + i] > max_val) {
                max_val = input[row * num_cols + i];
            }
        }
        
        float sum_exp = 0.0f;
        for (int i = 0; i < num_cols; ++i) {
            sum_exp += expf(input[row * num_cols + i] - max_val);
        }
        
        for (int i = 0; i < num_cols; ++i) {
            output[row * num_cols + i] = expf(input[row * num_cols + i] - max_val) / sum_exp;
        }
    }
}
