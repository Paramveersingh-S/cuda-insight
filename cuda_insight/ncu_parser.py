import csv
from typing import Dict, List

def parse_ncu_csv(csv_content: str) -> List[Dict[str, float]]:
    """
    Parse ncu CSV output.
    Returns a list of dicts (one per kernel invocation) mapping metric name to value.
    """
    lines = [line for line in csv_content.splitlines() if not line.startswith("==")]
    if not lines:
        return []
        
    reader = csv.DictReader(lines)
    results = []
    
    current_kernel_metrics = {}
    current_id = None
    
    for row in reader:
        m_name = row.get("Metric Name", "")
        m_val_str = row.get("Metric Value", "0").replace(",", "")
        k_id = row.get("ID", "0")
        
        if k_id != current_id:
            if current_id is not None and current_kernel_metrics:
                results.append(current_kernel_metrics)
            current_id = k_id
            current_kernel_metrics = {}
            
        try:
            val = float(m_val_str)
        except ValueError:
            val = 0.0
            
        if m_name:
            current_kernel_metrics[m_name] = val
            
    if current_kernel_metrics:
        results.append(current_kernel_metrics)
        
    return results
