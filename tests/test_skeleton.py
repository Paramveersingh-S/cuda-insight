import cuda_insight

def test_imports():
    assert cuda_insight.profile is not None
    assert cuda_insight.analyze is not None
    assert cuda_insight.report is not None
