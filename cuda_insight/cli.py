import click
from rich.console import Console
from rich.table import Table
from cuda_insight.utils.gpu_info import get_gpu_info

@click.group()
def cli():
    """CUDA Kernel Profiling, Debugging & Optimization Toolkit."""
    pass

@cli.command()
def profile():
    """Profile a CUDA kernel."""
    click.echo("Profile command coming soon.")

@cli.command(name="gpu-info")
def gpu_info_cmd():
    """Display information about the detected GPU."""
    console = Console()
    info = get_gpu_info()
    
    if not info:
        console.print("[bold red]Error:[/bold red] No NVIDIA GPU found, or nvidia-smi/nvcc not in PATH.")
        return

    table = Table(title="GPU Environment Information")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("GPU Name", info.name)
    table.add_row("Driver Version", info.driver_version)
    table.add_row("CUDA Version", info.cuda_version)
    table.add_row("Compute Capability", info.compute_capability)
    table.add_row("VRAM", f"{info.vram_gb} GB")
    table.add_row("SM Count", str(info.sm_count))
    table.add_row("Total CUDA Cores", str(info.total_cuda_cores))
    table.add_row("Memory Bandwidth", f"{info.memory_bandwidth_gbps} GB/s")
    table.add_row("Peak FP32 TFLOPS", str(info.peak_tflops_fp32))

    console.print(table)

if __name__ == "__main__":
    cli()
