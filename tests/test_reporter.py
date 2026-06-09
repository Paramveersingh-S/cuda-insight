import os
from rich.console import Console
from cuda_insight.analyzer import Bottleneck, AnalysisReport
from cuda_insight.suggestions import Suggestion
from cuda_insight.reporter import generate_html_report, print_terminal_report

def test_generate_html_report(tmp_path):
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
        bottlenecks=[Bottleneck.SHARED_MEMORY_BANK_CONFLICTS]
    )
    suggs = [
        Suggestion(
            bottleneck="Bank Conflicts",
            explanation="Testing",
            fix_snippet="Pad",
            impact="Fast",
            doc_link="http"
        )
    ]
    
    out_path = str(tmp_path / "report.html")
    generate_html_report(report, suggs, out_path)
    
    assert os.path.exists(out_path)
    with open(out_path, "r", encoding="utf-8") as f:
        content = f.read()
        assert "CUDA Insight Profile: test" in content
        assert "Bank Conflicts" in content

def test_print_terminal_report():
    report = AnalysisReport(
        kernel_name="test", duration_ns=1000, achieved_occupancy=40.0,
        theoretical_occupancy=100.0, arithmetic_intensity=1.0,
        bandwidth_util_pct=10.0, compute_util_pct=80.0,
        achieved_flops=10.0, achieved_bandwidth=100.0,
        bank_conflicts=100, bottlenecks=[]
    )
    console = Console()
    print_terminal_report(report, [], console)
