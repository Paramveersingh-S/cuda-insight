import os
import pytest
from unittest.mock import patch, MagicMock
from cuda_insight.ncu_parser import parse_ncu_csv
from cuda_insight.profiler import profile_kernel, ProfileResult

def test_parse_ncu_csv():
    csv_content = """==PROF== Connected to process
ID,Process ID,Process Name,Host Name,Kernel Name,Kernel Time,Context,Stream,Section Name,Metric Name,Metric Unit,Metric Value
0,1234,app,host,tiled_matmul,2023-01-01,1,7,Command line profiler metrics,sm__warps_active.avg.pct_of_peak_sustained_active,%,62.5
0,1234,app,host,tiled_matmul,2023-01-01,1,7,Command line profiler metrics,launch__registers_per_thread,register,96
"""
    results = parse_ncu_csv(csv_content)
    assert len(results) == 1
    assert results[0]["sm__warps_active.avg.pct_of_peak_sustained_active"] == 62.5
    assert results[0]["launch__registers_per_thread"] == 96.0

def test_profile_kernel(tmp_path):
    binary_path = str(tmp_path / "dummy_bin")
    with open(binary_path, "w") as f:
        f.write("dummy")

    with patch("subprocess.run") as mock_run:
        mock_res = MagicMock()
        mock_res.returncode = 0
        mock_res.stdout = "ID,Metric Name,Metric Value\n0,gpu__time_duration.sum,1000.5\n"
        mock_run.return_value = mock_res
        
        result = profile_kernel(binary_path, "tiled_matmul", [])
        
        assert isinstance(result, ProfileResult)
        assert result.kernel_name == "tiled_matmul"
        assert result.metrics["gpu__time_duration.sum"] == 1000.5
        assert "1000.5" in result.raw_csv
