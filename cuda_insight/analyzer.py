from enum import Enum
from dataclasses import dataclass
from typing import List, Optional
from cuda_insight.utils.gpu_info import GPUInfo
from cuda_insight.profiler import ProfileResult

class Bottleneck(Enum):
    MEMORY_BANDWIDTH = "memory_bandwidth"
    SHARED_MEMORY_BANK_CONFLICTS = "shared_memory_bank_conflicts"
    LOW_OCCUPANCY_REGISTER_PRESSURE = "low_occupancy_register_pressure"
    LOW_OCCUPANCY_SHARED_MEM = "low_occupancy_shared_mem"
    WARP_DIVERGENCE = "warp_divergence"
    INSTRUCTION_THROUGHPUT = "instruction_throughput"
    LAUNCH_CONFIG = "launch_config"
    MEMORY_COALESCING = "memory_coalescing"

@dataclass
class AnalysisReport:
    kernel_name: str
    duration_ns: float
    achieved_occupancy: float
    theoretical_occupancy: float
    arithmetic_intensity: float
    bandwidth_util_pct: float
    compute_util_pct: float
    achieved_flops: float
    achieved_bandwidth: float
    bank_conflicts: float
    bottlenecks: List[Bottleneck]

def analyze(profile_result: ProfileResult, gpu_info: GPUInfo) -> AnalysisReport:
    metrics = profile_result.metrics
    
    duration_ns = metrics.get("gpu__time_duration.sum", 0.0)
    duration_s = duration_ns / 1e9
    
    achieved_occupancy = metrics.get("sm__warps_active.avg.pct_of_peak_sustained_active", 0.0)
    theoretical_occupancy = metrics.get("launch__occupancy_theoretical", 0.0)
    
    # Bytes
    bytes_read = metrics.get("dram__bytes_read.sum", 0.0)
    bytes_write = metrics.get("dram__bytes_write.sum", 0.0)
    total_bytes = bytes_read + bytes_write
    
    achieved_bandwidth = 0.0
    if duration_s > 0:
        achieved_bandwidth = (total_bytes / duration_s) / 1e9 # GB/s
        
    bandwidth_util_pct = 0.0
    if gpu_info.memory_bandwidth_gbps > 0:
        bandwidth_util_pct = (achieved_bandwidth / gpu_info.memory_bandwidth_gbps) * 100.0

    # Flops
    fadds = metrics.get("smsp__sass_thread_inst_executed_op_fadd_pred_on.sum", 0.0)
    fmuls = metrics.get("smsp__sass_thread_inst_executed_op_fmul_pred_on.sum", 0.0)
    ffmas = metrics.get("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum", 0.0)
    # FMA is 2 ops
    total_flops = fadds + fmuls + (2 * ffmas)
    
    achieved_flops = 0.0
    if duration_s > 0:
        achieved_flops = (total_flops / duration_s) / 1e12 # TFLOPS
        
    compute_util_pct = 0.0
    if gpu_info.peak_tflops_fp32 > 0:
        compute_util_pct = (achieved_flops / gpu_info.peak_tflops_fp32) * 100.0
        
    arithmetic_intensity = 0.0
    if total_bytes > 0:
        arithmetic_intensity = total_flops / total_bytes
        
    bank_conflicts = metrics.get("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld", 0.0)
    registers_per_thread = metrics.get("launch__registers_per_thread", 0.0)
    shared_mem_per_block = metrics.get("launch__shared_mem_per_block_static", 0.0)
    
    # Bottleneck analysis
    bottlenecks = []
    if bank_conflicts > 0:
        bottlenecks.append(Bottleneck.SHARED_MEMORY_BANK_CONFLICTS)
    
    if registers_per_thread > 64 and achieved_occupancy < 80.0:
        bottlenecks.append(Bottleneck.LOW_OCCUPANCY_REGISTER_PRESSURE)
        
    if shared_mem_per_block > 49152 and achieved_occupancy < 80.0:
        bottlenecks.append(Bottleneck.LOW_OCCUPANCY_SHARED_MEM)
        
    if bandwidth_util_pct > compute_util_pct and bandwidth_util_pct > 60.0:
        bottlenecks.append(Bottleneck.MEMORY_BANDWIDTH)
        
    if compute_util_pct > bandwidth_util_pct and compute_util_pct > 60.0:
        bottlenecks.append(Bottleneck.INSTRUCTION_THROUGHPUT)
        
    if theoretical_occupancy < 50.0:
        bottlenecks.append(Bottleneck.LAUNCH_CONFIG)
        
    return AnalysisReport(
        kernel_name=profile_result.kernel_name,
        duration_ns=duration_ns,
        achieved_occupancy=achieved_occupancy,
        theoretical_occupancy=theoretical_occupancy,
        arithmetic_intensity=arithmetic_intensity,
        bandwidth_util_pct=bandwidth_util_pct,
        compute_util_pct=compute_util_pct,
        achieved_flops=achieved_flops,
        achieved_bandwidth=achieved_bandwidth,
        bank_conflicts=bank_conflicts,
        bottlenecks=bottlenecks
    )
