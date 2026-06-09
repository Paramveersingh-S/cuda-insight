import click
import os
from rich.console import Console
from rich.table import Table
from cuda_insight.utils.gpu_info import get_gpu_info
from cuda_insight.compiler import compile_cu
from cuda_insight.ptx_annotator import parse_ptx_and_annotate

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

@cli.command(name="ptx")
@click.argument("source_file", type=click.Path(exists=True))
@click.option("--annotate", is_flag=True, help="Annotate PTX with source lines")
@click.option("--arch", default="sm_80", help="Target architecture")
def ptx_cmd(source_file, annotate, arch):
    """Compile and show PTX."""
    console = Console()
    output_dir = "build"
    try:
        res = compile_cu(source_file, arch, output_dir)
    except Exception as e:
        console.print(f"[bold red]Compile Error:[/bold red] {e}")
        return
    
    if not res.ptx_path or not os.path.exists(res.ptx_path):
        console.print("[bold red]Error:[/bold red] PTX file not generated.")
        return

    if annotate:
        lines = parse_ptx_and_annotate(res.ptx_path, [source_file])
        for line in lines:
            if line.source_snippet:
                console.print(f"[cyan]{line.source_line:4d} | {line.source_snippet}[/cyan]")
            console.print(f"       {line.ptx_instruction}")
    else:
        with open(res.ptx_path, "r", encoding="utf-8") as f:
            console.print(f.read())

if __name__ == "__main__":
    cli()
