import pytest
from unittest.mock import patch, MagicMock
from cuda_insight.utils.gpu_info import get_gpu_info, GPUInfo

def test_get_gpu_info_success():
    with patch("subprocess.run") as mock_run:
        # Setup mock for nvidia-smi
        mock_smi = MagicMock()
        mock_smi.returncode = 0
        mock_smi.stdout = "NVIDIA A100-SXM4-80GB, 81920 MiB, 535.104.05\n"
        
        # Setup mock for nvcc
        mock_nvcc = MagicMock()
        mock_nvcc.returncode = 0
        mock_nvcc.stdout = "nvcc: NVIDIA (R) Cuda compiler driver\nCopyright (c) 2005-2023 NVIDIA Corporation\nBuilt on Tue_Aug_15_22:02:13_PDT_2023\nCuda compilation tools, release 12.2, V12.2.140\nBuild cuda_12.2.r12.2/compiler.33191640_0\n"
        
        mock_run.side_effect = [mock_smi, mock_nvcc]
        
        info = get_gpu_info()
        assert info is not None
        assert info.name == "NVIDIA A100-SXM4-80GB"
        assert info.vram_gb == 80.0
        assert info.driver_version == "535.104.05"
        assert info.cuda_version == "12.2"

def test_get_gpu_info_no_smi():
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError()
        
        info = get_gpu_info()
        assert info is None

def test_get_gpu_info_smi_fails():
    with patch("subprocess.run") as mock_run:
        mock_smi = MagicMock()
        mock_smi.returncode = 1
        mock_smi.stderr = "NVIDIA-SMI has failed"
        mock_run.return_value = mock_smi
        
        info = get_gpu_info()
        assert info is None
