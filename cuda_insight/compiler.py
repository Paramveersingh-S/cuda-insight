import subprocess
import os
import re
import logging
import platform
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

@dataclass
class CompileResult:
    source_path: str
    output_dir: str
    ptx_path: Optional[str]
    sass_path: Optional[str]
    so_path: Optional[str]
    kernel_names: List[str]

@dataclass
class CompileError(Exception):
    message: str
    stderr: str

def compile_cu(source_path: str, sm_arch: str, output_dir: str, extra_flags: List[str] = None) -> CompileResult:
    """
    Compile a .cu file to PTX, SASS, and shared library (.so)
    """
    if extra_flags is None:
        extra_flags = []
    
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.basename(source_path).replace(".cu", "")
    
    ptx_path = os.path.join(output_dir, f"{basename}.ptx")
    so_path = os.path.join(output_dir, f"{basename}.so")
    
    # 1. Compile to PTX
    nvcc_cmd_ptx = [
        "nvcc", source_path, "-O3", "-lineinfo", "--ptx",
        f"-arch={sm_arch}", "-o", ptx_path
    ] + extra_flags
    
    res = subprocess.run(nvcc_cmd_ptx, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise CompileError(f"Failed to compile {source_path} to PTX", res.stderr)
        
    # 2. Compile to Shared Library (for cuobjdump and benchmarking)
    is_windows = platform.system() == "Windows"
    pic_flag = [] if is_windows else ["--compiler-options", "'-fPIC'"]
    
    nvcc_cmd_so = [
        "nvcc", source_path, "-O3", "-lineinfo", "-shared"
    ] + pic_flag + [
        f"-arch={sm_arch}", "-o", so_path
    ] + extra_flags
    
    res_so = subprocess.run(nvcc_cmd_so, capture_output=True, text=True, check=False)
    actual_so_path = so_path if res_so.returncode == 0 else None

    # 3. Extract SASS if .so was generated
    sass_path = None
    if actual_so_path:
        sass_path = os.path.join(output_dir, f"{basename}.sass")
        sass_cmd = ["cuobjdump", "--dump-sass", actual_so_path]
        res_sass = subprocess.run(sass_cmd, capture_output=True, text=True, check=False)
        if res_sass.returncode == 0:
            with open(sass_path, "w", encoding="utf-8") as f:
                f.write(res_sass.stdout)
    
    # 4. Extract Kernel Names from PTX
    kernel_names = []
    if os.path.exists(ptx_path):
        with open(ptx_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Find lines like `.visible .entry _Z12tiled_matmul...`
            matches = re.findall(r"\.visible\s+\.entry\s+([_a-zA-Z0-9]+)", content)
            kernel_names = matches
            
    return CompileResult(
        source_path=source_path,
        output_dir=output_dir,
        ptx_path=ptx_path,
        sass_path=sass_path,
        so_path=actual_so_path,
        kernel_names=kernel_names
    )
