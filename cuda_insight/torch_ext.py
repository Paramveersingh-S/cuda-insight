import sys
from rich.console import Console

def profile_torch_kernel(func, *args, **kwargs):
    """
    Profile a PyTorch CUDA kernel using PyTorch's native profiler.
    This provides high-level ops information. 
    For low-level hardware metrics, use the CLI `cuda-insight profile` on the compiled binary.
    """
    console = Console()
    try:
        import torch
    except ImportError:
        console.print("[bold red]Error:[/bold red] PyTorch is not installed. Please install PyTorch to use this feature.")
        return None

    if not torch.cuda.is_available():
        console.print("[bold yellow]Warning:[/bold yellow] PyTorch cannot detect a CUDA device. Profiling will run on CPU only.")

    console.print(f"[bold green]Profiling PyTorch execution of {func.__name__}...[/bold green]")
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        res = func(*args, **kwargs)
        
    console.print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    return res
