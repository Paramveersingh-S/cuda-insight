import subprocess
import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
from cuda_insight.utils.gpu_info import GPUInfo
from cuda_insight.ncu_parser import parse_ncu_csv

logger = logging.getLogger(__name__)

NCU_METRICS = [
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__occupancy_theoretical",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_static",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "sm__inst_executed.sum",
    "sm__warps_eligible.sum",
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "gpu__time_duration.sum",
]

@dataclass
class ProfileResult:
    kernel_name: str
    metrics: Dict[str, float]
    raw_csv: str

def profile_kernel(binary_path: str, kernel_name: str, args: List[str], gpu_info: Optional[GPUInfo] = None) -> ProfileResult:
    """
    Profile a kernel using ncu.
    """
    if not os.path.exists(binary_path):
        raise FileNotFoundError(f"Binary not found: {binary_path}")
        
    metrics_str = ",".join(NCU_METRICS)
    
    cmd = [
        "ncu", "--csv", "--page", "raw", "--metrics", metrics_str
    ]
    if kernel_name:
        cmd.extend(["--kernel-name-base", "function", "--kernel-name", kernel_name])
        
    cmd.append(binary_path)
    cmd.extend(args)
    
    logger.info(f"Running profiler: {' '.join(cmd)}")
    
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    
    if res.returncode != 0:
        logger.error(f"ncu profiling failed. Stderr: {res.stderr}")
        if "Permissions" in res.stderr or "sudo" in res.stderr:
             logger.warning("ncu may require root privileges or kernel.perf_event_paranoid=0")
    
    parsed_results = parse_ncu_csv(res.stdout)
    metrics = parsed_results[0] if parsed_results else {}
    
    return ProfileResult(
        kernel_name=kernel_name,
        metrics=metrics,
        raw_csv=res.stdout
    )
