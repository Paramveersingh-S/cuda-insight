import os
import matplotlib.pyplot as plt
import numpy as np

def generate_roofline_chart(
    peak_flops: float, 
    peak_bandwidth: float, 
    arithmetic_intensity: float, 
    achieved_flops: float,
    output_path: str
):
    """
    Generate a simple roofline chart using matplotlib.
    peak_flops in TFLOPS
    peak_bandwidth in GB/s
    achieved_flops in TFLOPS
    arithmetic_intensity in FLOP/Byte
    """
    ridge_point = (peak_flops * 1e12) / (peak_bandwidth * 1e9)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # x axis is arithmetic intensity
    ai = np.logspace(-2, 3, 100)
    
    # y axis is Performance (TFLOPS)
    # Memory bound roof: ai * peak_bandwidth / 1000
    # Compute bound roof: peak_flops
    perf_mem = ai * (peak_bandwidth / 1000.0) 
    perf_compute = np.full_like(ai, peak_flops)
    perf_roof = np.minimum(perf_mem, perf_compute)
    
    ax.plot(ai, perf_roof, label='Roofline', color='black', linewidth=2)
    ax.plot(ai, perf_mem, color='blue', linestyle='--', alpha=0.5)
    ax.plot(ai, perf_compute, color='red', linestyle='--', alpha=0.5)
    
    # Plot the kernel point
    ax.scatter([arithmetic_intensity], [achieved_flops], color='orange', s=100, zorder=5, label='Kernel')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Arithmetic Intensity (FLOP/Byte)')
    ax.set_ylabel('Performance (TFLOPS)')
    ax.set_title('Roofline Model')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
