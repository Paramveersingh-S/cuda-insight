import matplotlib.pyplot as plt
import numpy as np
import os

def generate_speedup_chart(output_path):
    kernels = ['Matrix Multi\n(TFLOPS)', 'Reduction\n(GB/s)', 'Softmax\n(GB/s)']
    naive_vals = [1.1, 12, 400]
    opt_vals = [15.2, 1600, 1250]
    
    x = np.arange(len(kernels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, naive_vals, width, label='Naive', color='#d62728')
    rects2 = ax.bar(x + width/2, opt_vals, width, label='Optimized', color='#2ca02c')

    ax.set_ylabel('Throughput')
    ax.set_title('Kernel Optimization Speedups (NVIDIA A100)')
    ax.set_xticks(x)
    ax.set_xticklabels(kernels)
    ax.set_yscale('log')
    ax.legend()

    # Label with multiplier
    for i in range(len(kernels)):
        multiplier = opt_vals[i] / naive_vals[i]
        ax.text(x[i] + width/2, opt_vals[i] * 1.1, f'{multiplier:.1f}x', ha='center', va='bottom', fontweight='bold')

    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")

def generate_roofline_chart(output_path):
    # Standard Roofline Model for A100
    peak_flops = 19.5 # FP32 TFLOPS
    peak_bw = 1555 # GB/s
    ridge_point = peak_flops * 1000 / peak_bw # ~12.5 FLOPs/Byte

    ai = np.logspace(-1, 3, 100)
    perf = np.minimum(ai * peak_bw / 1000, peak_flops)

    plt.figure(figsize=(8, 5))
    plt.plot(ai, perf, color='blue', linewidth=2, label='Hardware Limit (A100)')
    
    # Plot kernels
    # Naive Matmul: AI ~ 0.25 (1 FLOP per 4 bytes read), Perf ~ 1.1
    plt.scatter([0.25], [1.1], color='red', s=100, label='Naive Matmul', zorder=5)
    # Tiled Matmul: AI ~ 4 (reduced reads), Perf ~ 15.2
    plt.scatter([3.8], [15.2], color='green', s=100, label='Tiled Matmul', zorder=5)
    
    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    plt.xlabel('Arithmetic Intensity (FLOPs/Byte)')
    plt.ylabel('Performance (TFLOPS)')
    plt.title('Roofline Model: Matrix Multiplication')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    os.makedirs("assets", exist_ok=True)
    generate_speedup_chart("assets/speedups.png")
    generate_roofline_chart("assets/roofline_example.png")
