from unittest.mock import patch, MagicMock
from cuda_insight.torch_ext import profile_torch_kernel
import sys

def test_profile_torch_kernel_no_torch():
    # Simulate no torch
    with patch.dict('sys.modules', {'torch': None}):
        def dummy(): pass
        res = profile_torch_kernel(dummy)
        assert res is None

def test_profile_torch_kernel_with_mock_torch():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    
    mock_prof_instance = MagicMock()
    mock_prof_instance.key_averages.return_value.table.return_value = "Mock Table"
    
    mock_prof = MagicMock()
    mock_prof.__enter__.return_value = mock_prof_instance
    mock_torch.profiler.profile.return_value = mock_prof
    
    with patch.dict('sys.modules', {'torch': mock_torch}):
        def dummy(): return 42
        res = profile_torch_kernel(dummy)
        assert res == 42
        mock_torch.profiler.profile.assert_called_once()
