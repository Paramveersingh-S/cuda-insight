import re
import os
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class PTXLine:
    ptx_instruction: str
    source_file: str
    source_line: int
    source_snippet: str

def parse_ptx_and_annotate(ptx_path: str, source_paths: List[str]) -> List[PTXLine]:
    """
    Parse a PTX file and map its instructions to source code lines using .loc directives.
    """
    if not os.path.exists(ptx_path):
        return []
        
    # Read source files into memory
    source_cache: Dict[str, List[str]] = {}
    for sp in source_paths:
        if os.path.exists(sp):
            with open(sp, "r", encoding="utf-8") as f:
                source_cache[os.path.basename(sp)] = f.read().splitlines()
                
    annotated_lines = []
    
    with open(ptx_path, "r", encoding="utf-8") as f:
        ptx_lines = f.read().splitlines()
        
    current_file_idx = -1
    current_line_num = -1
    
    file_map: Dict[int, str] = {}
    
    for line in ptx_lines:
        line = line.strip()
        if not line:
            continue
            
        # Parse file map: .file 1 "mykernel.cu"
        file_match = re.match(r"\.file\s+(\d+)\s+\"([^\"]+)\"", line)
        if file_match:
            file_map[int(file_match.group(1))] = os.path.basename(file_match.group(2))
            continue
            
        # Parse loc: .loc 1 42 8
        loc_match = re.match(r"\.loc\s+(\d+)\s+(\d+)\s+(\d+)", line)
        if loc_match:
            current_file_idx = int(loc_match.group(1))
            current_line_num = int(loc_match.group(2))
            continue
            
        # It's an instruction
        if not line.startswith(".") and not line.startswith("//"):
            filename = file_map.get(current_file_idx, "")
            snippet = ""
            if filename in source_cache and current_line_num > 0:
                if current_line_num - 1 < len(source_cache[filename]):
                    snippet = source_cache[filename][current_line_num - 1].strip()
                    
            annotated_lines.append(PTXLine(
                ptx_instruction=line,
                source_file=filename,
                source_line=current_line_num,
                source_snippet=snippet
            ))
            
    return annotated_lines
