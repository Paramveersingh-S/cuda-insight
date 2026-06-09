from typing import List, Dict
from dataclasses import dataclass
from cuda_insight.analyzer import Bottleneck, AnalysisReport

@dataclass
class Suggestion:
    bottleneck: str
    explanation: str
    fix_snippet: str
    impact: str
    doc_link: str

def get_suggestions(report: AnalysisReport) -> List[Suggestion]:
    suggestions = []
    
    for bottleneck in report.bottlenecks:
        if bottleneck == Bottleneck.SHARED_MEMORY_BANK_CONFLICTS:
            suggestions.append(Suggestion(
                bottleneck="Shared Memory Bank Conflicts",
                explanation="Multiple threads in a warp are accessing different addresses that map to the same shared memory bank, causing serialization.",
                fix_snippet="__shared__ float smem[TILE][TILE+1]; // Add padding",
                impact="Typically 1.5x - 2x speedup on shared memory bound kernels.",
                doc_link="https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory"
            ))
        elif bottleneck == Bottleneck.LOW_OCCUPANCY_REGISTER_PRESSURE:
            suggestions.append(Suggestion(
                bottleneck="Low Occupancy (Register Pressure)",
                explanation=f"Kernel uses too many registers per thread, limiting the number of active warps. (Currently {report.achieved_occupancy:.1f}% occupancy)",
                fix_snippet="__launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)",
                impact="+20-35% occupancy -> +10-15% throughput.",
                doc_link="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#launch-bounds"
            ))
        elif bottleneck == Bottleneck.LOW_OCCUPANCY_SHARED_MEM:
            suggestions.append(Suggestion(
                bottleneck="Low Occupancy (Shared Memory)",
                explanation="Kernel allocates too much shared memory per block, preventing multiple blocks from residing on the same SM.",
                fix_snippet="Reduce TILE_SIZE or split kernel into multiple passes.",
                impact="Depends on memory latency hiding requirements.",
                doc_link="https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy"
            ))
        elif bottleneck == Bottleneck.MEMORY_BANDWIDTH:
            suggestions.append(Suggestion(
                bottleneck="Memory Bandwidth Bound",
                explanation=f"Kernel is bottlenecked by global memory bandwidth. Utilizing {report.bandwidth_util_pct:.1f}% of peak.",
                fix_snippet="float4 val = reinterpret_cast<const float4*>(ptr)[i]; // Vectorized loads",
                impact="Better coalescing and reduced instruction overhead.",
                doc_link="https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/"
            ))
        elif bottleneck == Bottleneck.INSTRUCTION_THROUGHPUT:
            suggestions.append(Suggestion(
                bottleneck="Instruction Throughput Bound",
                explanation=f"Kernel is compute bound. Utilizing {report.compute_util_pct:.1f}% of peak FLOPS.",
                fix_snippet="Use __fma_rn() or Tensor Cores via wmma API.",
                impact="Can reach 10x-30x peak FLOPS with Tensor Cores.",
                doc_link="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma"
            ))
        elif bottleneck == Bottleneck.LAUNCH_CONFIG:
            suggestions.append(Suggestion(
                bottleneck="Suboptimal Launch Configuration",
                explanation=f"Theoretical occupancy is very low ({report.theoretical_occupancy:.1f}%). Check block size and grid size.",
                fix_snippet="Ensure block size is a multiple of 32 (e.g. 128, 256) and grid provides enough blocks.",
                impact="More blocks/threads can better hide memory latency.",
                doc_link="https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#execution-configuration"
            ))
            
    return suggestions
