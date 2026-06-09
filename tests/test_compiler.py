import os
import pytest
from unittest.mock import patch, MagicMock
from cuda_insight.compiler import compile_cu, CompileResult, CompileError
from cuda_insight.ptx_annotator import parse_ptx_and_annotate, PTXLine

def test_compile_cu_success(tmp_path):
    with patch("subprocess.run") as mock_run:
        # Mock NVCC PTX
        mock_ptx_res = MagicMock()
        mock_ptx_res.returncode = 0
        
        # Mock NVCC SO
        mock_so_res = MagicMock()
        mock_so_res.returncode = 0
        
        # Mock cuobjdump SASS
        mock_sass_res = MagicMock()
        mock_sass_res.returncode = 0
        mock_sass_res.stdout = "SASS code"
        
        mock_run.side_effect = [mock_ptx_res, mock_so_res, mock_sass_res]
        
        # Create a fake PTX file to be read
        ptx_content = ".visible .entry _Z10my_kernel1v()\n{\n}"
        source_path = str(tmp_path / "test.cu")
        with open(source_path, "w") as f:
            f.write("test")
            
        output_dir = str(tmp_path / "out")
        os.makedirs(output_dir, exist_ok=True)
        ptx_path = os.path.join(output_dir, "test.ptx")
        with open(ptx_path, "w") as f:
            f.write(ptx_content)
            
        res = compile_cu(source_path, "sm_80", output_dir)
        
        assert res.source_path == source_path
        assert "test.ptx" in res.ptx_path
        assert len(res.kernel_names) == 1
        assert res.kernel_names[0] == "_Z10my_kernel1v"

def test_compile_cu_failure(tmp_path):
    with patch("subprocess.run") as mock_run:
        mock_ptx_res = MagicMock()
        mock_ptx_res.returncode = 1
        mock_ptx_res.stderr = "Syntax error"
        mock_run.return_value = mock_ptx_res
        
        with pytest.raises(CompileError):
            compile_cu(str(tmp_path / "test.cu"), "sm_80", str(tmp_path / "out"))

def test_ptx_annotator(tmp_path):
    ptx_content = """
.file 1 "test.cu"
.loc 1 2 0
add.s32 %r1, %r2, %r3;
"""
    source_content = "int main() {\n  int a = b + c;\n}"
    
    ptx_path = str(tmp_path / "test.ptx")
    with open(ptx_path, "w") as f:
        f.write(ptx_content)
        
    source_path = str(tmp_path / "test.cu")
    with open(source_path, "w") as f:
        f.write(source_content)
        
    lines = parse_ptx_and_annotate(ptx_path, [source_path])
    
    assert len(lines) == 1
    assert lines[0].ptx_instruction == "add.s32 %r1, %r2, %r3;"
    assert lines[0].source_file == "test.cu"
    assert lines[0].source_line == 2
    assert lines[0].source_snippet == "int a = b + c;"
