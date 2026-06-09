import subprocess
import logging
from dataclasses import dataclass
from typing import Optional, Dict

logger = logging.getLogger(__name__)

@dataclass
class GPUInfo:
    name: str
    compute_capability: str
    sm_count: int
    cuda_cores_per_sm: int
    total_cuda_cores: int
    vram_gb: float
    memory_bandwidth_gbps: float
    l2_cache_mb: float
    peak_tflops_fp32: float
    peak_tflops_fp16: float
    warp_size: int
    max_threads_per_block: int
    max_shared_mem_per_block_kb: int
    driver_version: str
    cuda_version: str

def get_gpu_info() -> Optional[GPUInfo]:
    """
    Query the system for GPU info using nvidia-smi and nvcc.
    Returns None if no NVIDIA GPU is found or driver is missing.
    """
    try:
        # Check nvidia-smi
        smi_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False
        )
        if smi_result.returncode != 0:
            logger.warning(f"nvidia-smi failed: {smi_result.stderr}")
            return None

        parts = smi_result.stdout.strip().split(', ')
        if len(parts) < 3:
            return None
            
        name = parts[0]
        vram_str = parts[1]
        driver_version = parts[2]
        
        vram_gb = 0.0
        if "MiB" in vram_str:
            vram_gb = float(vram_str.replace("MiB", "").strip()) / 1024.0

        # Check nvcc
        nvcc_result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            check=False
        )
        cuda_version = "Unknown"
        if nvcc_result.returncode == 0:
            for line in nvcc_result.stdout.split('\n'):
                if "release" in line:
                    cuda_version = line.split("release")[-1].strip().split(",")[0].strip()

        # For a full implementation, we would query cuDeviceGetAttribute or have a lookup table.
        # Here we provide some defaults for the purpose of the tool skeleton,
        # which can be overridden or filled by a hardware DB in the future.
        return GPUInfo(
            name=name,
            compute_capability="8.0", # Placeholder
            sm_count=108,             # Placeholder for A100
            cuda_cores_per_sm=64,     # Placeholder
            total_cuda_cores=6912,    # Placeholder
            vram_gb=round(vram_gb, 2),
            memory_bandwidth_gbps=1555.0, # Placeholder
            l2_cache_mb=40.0,         # Placeholder
            peak_tflops_fp32=19.5,    # Placeholder
            peak_tflops_fp16=78.0,    # Placeholder
            warp_size=32,
            max_threads_per_block=1024,
            max_shared_mem_per_block_kb=164,
            driver_version=driver_version,
            cuda_version=cuda_version
        )
    except FileNotFoundError:
        logger.warning("nvidia-smi or nvcc not found in PATH")
        return None
    except Exception as e:
        logger.error(f"Error querying GPU info: {e}")
        return None
