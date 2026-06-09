import os
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from cuda_insight.analyzer import AnalysisReport
from cuda_insight.suggestions import Suggestion

def generate_html_report(report: AnalysisReport, suggestions: list[Suggestion], output_path: str):
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template('report_template.html')

    html_content = template.render(
        kernel_name=report.kernel_name,
        date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        duration_ns=report.duration_ns,
        achieved_occupancy=report.achieved_occupancy,
        theoretical_occupancy=report.theoretical_occupancy,
        arithmetic_intensity=report.arithmetic_intensity,
        bandwidth_util_pct=report.bandwidth_util_pct,
        compute_util_pct=report.compute_util_pct,
        achieved_flops=report.achieved_flops,
        achieved_bandwidth=report.achieved_bandwidth,
        bank_conflicts=report.bank_conflicts,
        bottlenecks=[b.value for b in report.bottlenecks],
        suggestions=suggestions
    )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

def print_terminal_report(report: AnalysisReport, suggestions: list[Suggestion], console: Console):
    console.print(Panel(f"[bold green]Analysis Report for {report.kernel_name}[/bold green]", expand=False))
    
    table = Table(title="Metrics Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Duration", f"{report.duration_ns / 1000.0:.2f} us")
    table.add_row("Achieved Occupancy", f"{report.achieved_occupancy:.1f}%")
    table.add_row("Compute Utilization", f"{report.compute_util_pct:.1f}%")
    table.add_row("Bandwidth Utilization", f"{report.bandwidth_util_pct:.1f}%")
    table.add_row("Arithmetic Intensity", f"{report.arithmetic_intensity:.2f} FLOP/B")
    console.print(table)
    
    if suggestions:
        console.print("\n[bold yellow]Suggestions:[/bold yellow]")
        for s in suggestions:
            console.print(f"- [bold]{s.bottleneck}[/bold]: {s.explanation}")
            console.print(f"  [cyan]Fix:[/cyan] {s.fix_snippet}")
            console.print(f"  [cyan]Impact:[/cyan] {s.impact}")
            console.print(f"  [dim]Read more: {s.doc_link}[/dim]\n")
    else:
        console.print("\n[bold green]No major bottlenecks found![/bold green]")
