/*
 * KERNEL: Online Softmax
 * OPTIMIZATION: Single-pass statistically stable softmax
 * KEY INSIGHT: Keeps a running max and normalizer to compute softmax in a single pass over the data.
 * EXPECTED SPEEDUP: ~2x reduction in memory traffic
 * KNOWN BOTTLENECK: Still memory bandwidth bound.
 * FURTHER READING: FlashAttention paper uses this technique to avoid materializing the full attention matrix.
 */

#include <math.h>

extern "C" __global__ void online_softmax(const float* input, float* output, int num_rows, int num_cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float max_val = -INFINITY;
        float normalizer = 0.0f;
        
        // Single pass to find max and running sum
        for (int i = 0; i < num_cols; ++i) {
            float val = input[row * num_cols + i];
            if (val > max_val) {
                normalizer = normalizer * expf(max_val - val) + 1.0f;
                max_val = val;
            } else {
                normalizer += expf(val - max_val);
            }
        }
        
        // Write out results
        for (int i = 0; i < num_cols; ++i) {
            output[row * num_cols + i] = expf(input[row * num_cols + i] - max_val) / normalizer;
        }
    }
}
