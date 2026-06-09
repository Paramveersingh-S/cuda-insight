from cuda_insight.analyzer import Bottleneck, AnalysisReport
from cuda_insight.suggestions import get_suggestions

def test_get_suggestions():
    report = AnalysisReport(
        kernel_name="test",
        duration_ns=1000,
        achieved_occupancy=40.0,
        theoretical_occupancy=100.0,
        arithmetic_intensity=1.0,
        bandwidth_util_pct=10.0,
        compute_util_pct=80.0,
        achieved_flops=10.0,
        achieved_bandwidth=100.0,
        bank_conflicts=100,
        bottlenecks=[Bottleneck.SHARED_MEMORY_BANK_CONFLICTS, Bottleneck.INSTRUCTION_THROUGHPUT]
    )
    
    suggestions = get_suggestions(report)
    assert len(suggestions) == 2
    assert suggestions[0].bottleneck == "Shared Memory Bank Conflicts"
    assert suggestions[1].bottleneck == "Instruction Throughput Bound"
