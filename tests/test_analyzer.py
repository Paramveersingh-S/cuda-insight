from cuda_insight.analyzer import analyze, Bottleneck
from cuda_insight.profiler import ProfileResult
from cuda_insight.utils.gpu_info import GPUInfo

def test_analyzer():
    gpu = GPUInfo(
        name="A100", compute_capability="8.0", sm_count=108, cuda_cores_per_sm=64,
        total_cuda_cores=6912, vram_gb=80.0, memory_bandwidth_gbps=1555.0, l2_cache_mb=40.0,
        peak_tflops_fp32=19.5, peak_tflops_fp16=78.0, warp_size=32, max_threads_per_block=1024,
        max_shared_mem_per_block_kb=164, driver_version="535", cuda_version="12.2"
    )
    
    metrics = {
        "gpu__time_duration.sum": 1e6, # 1 ms
        "sm__warps_active.avg.pct_of_peak_sustained_active": 40.0,
        "launch__occupancy_theoretical": 100.0,
        "dram__bytes_read.sum": 1e9, # 1 GB
        "dram__bytes_write.sum": 0.5e9, # 0.5 GB
        "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum": 1e9, # 1 G FMA (2 GFLOPS)
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld": 100,
        "launch__registers_per_thread": 96
    }
    
    res = ProfileResult(kernel_name="test", metrics=metrics, raw_csv="")
    
    report = analyze(res, gpu)
    
    assert report.bank_conflicts == 100
    assert Bottleneck.SHARED_MEMORY_BANK_CONFLICTS in report.bottlenecks
    assert Bottleneck.LOW_OCCUPANCY_REGISTER_PRESSURE in report.bottlenecks
    # total bytes = 1.5GB in 1ms = 1500 GB/s. Max is 1555.
    assert report.bandwidth_util_pct > 90.0
    assert Bottleneck.MEMORY_BANDWIDTH in report.bottlenecks
