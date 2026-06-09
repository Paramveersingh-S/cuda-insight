import click
import os
from rich.console import Console
from rich.table import Table
from cuda_insight.utils.gpu_info import get_gpu_info
from cuda_insight.compiler import compile_cu
from cuda_insight.ptx_annotator import parse_ptx_and_annotate

from cuda_insight.profiler import profile_kernel
from cuda_insight.analyzer import analyze
from cuda_insight.suggestions import get_suggestions
from cuda_insight.reporter import print_terminal_report, generate_html_report
from cuda_insight.utils.perf_model import generate_roofline_chart

@click.group()
def cli():
    """CUDA Kernel Profiling, Debugging & Optimization Toolkit."""
    pass

@cli.command()
@click.argument("binary_path", type=click.Path(exists=True))
@click.option("--kernel", help="Name of the kernel to profile (optional, runs all if empty)")
@click.option("--html", is_flag=True, help="Generate an HTML report")
@click.option("--roofline", is_flag=True, help="Generate a Roofline chart (roofline.png)")
@click.argument("args", nargs=-1)
def profile(binary_path, kernel, html, roofline, args):
    """Profile a CUDA kernel binary."""
    console = Console()
    gpu = get_gpu_info()
    if not gpu:
        console.print("[bold red]Error:[/bold red] No NVIDIA GPU found, or nvidia-smi/nvcc not in PATH.")
        return

    with console.status(f"Profiling {binary_path}..."):
        try:
            res = profile_kernel(binary_path, kernel, list(args), gpu)
        except Exception as e:
            console.print(f"[bold red]Profiling Error:[/bold red] {e}")
            return
            
        if not res.metrics:
            console.print("[bold red]Error:[/bold red] No profiling data captured. Ensure ncu is installed and you have sufficient permissions.")
            return
            
        report = analyze(res, gpu)
        suggs = get_suggestions(report)
        
    print_terminal_report(report, suggs, console)
    
    if html:
        out_path = f"{kernel or 'report'}_cuda_insight.html"
        generate_html_report(report, suggs, out_path)
        console.print(f"[bold green]Saved HTML report to:[/bold green] {out_path}")
        
    if roofline:
        out_path = f"{kernel or 'roofline'}.png"
        generate_roofline_chart(
            gpu.peak_tflops_fp32, 
            gpu.memory_bandwidth_gbps, 
            report.arithmetic_intensity, 
            report.achieved_flops, 
            out_path
        )
        console.print(f"[bold green]Saved Roofline chart to:[/bold green] {out_path}")

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

@cli.command(name="profile-torch")
@click.argument("script_path", type=click.Path(exists=True))
@click.option("--op", help="Name of the custom op to profile")
def profile_torch(script_path, op):
    """Profile a PyTorch custom op script."""
    console = Console()
    console.print(f"[bold green]Running and profiling PyTorch script:[/bold green] {script_path}")
    os.system(f"python {script_path}")

@cli.command(name="compare")
@click.argument("binary1", type=click.Path(exists=True))
@click.argument("binary2", type=click.Path(exists=True))
@click.option("--kernel-names", help="Comma separated kernel names e.g. naive,tiled")
def compare(binary1, binary2, kernel_names):
    """Compare two compiled kernels."""
    console = Console()
    console.print("[bold yellow]Compare feature is coming soon![/bold yellow] For now, please run profile separately and use the HTML report.")

@cli.command(name="benchmark")
@click.option("--suite", help="Suite to run (e.g. matmul)")
@click.option("--sizes", help="Comma separated sizes e.g. 256,512,1024")
def benchmark(suite, sizes):
    """Run built-in benchmark suite."""
    console = Console()
    console.print(f"[bold yellow]Running benchmark suite '{suite}' with sizes {sizes}...[/bold yellow]")
    console.print("[dim]Benchmark runner is coming in a future update.[/dim]")

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
