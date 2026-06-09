import os
from cuda_insight.utils.perf_model import generate_roofline_chart

def test_generate_roofline_chart(tmp_path):
    output = str(tmp_path / "chart.png")
    generate_roofline_chart(19.5, 1555.0, 10.0, 5.0, output)
    assert os.path.exists(output)
