"""
DomainBench CLI - Command line interface for running benchmarks
"""

import typer
from typing import Optional, List
from pathlib import Path
from rich.console import Console

app = typer.Typer(
    name="domainbench",
    help="DomainBench - LLM Benchmarking Framework",
    add_completion=False,
)

console = Console()


@app.command()
def run(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c",
        help="Path to benchmark configuration YAML file"
    ),
    dataset: Path = typer.Option(
        ..., "--dataset", "-d",
        help="Path to dataset JSONL file"
    ),
    domain: str = typer.Option(
        "restaurant_waiter", "--domain",
        help="Domain name or path to domain config"
    ),
    models: Optional[List[str]] = typer.Option(
        None, "--models", "-m",
        help="Models to compare (format: provider/model)"
    ),
    output: Path = typer.Option(
        Path("./results"), "--output", "-o",
        help="Output directory for results"
    ),
    max_items: Optional[int] = typer.Option(
        None, "--max-items",
        help="Maximum number of test cases to run"
    ),
    judge_model: str = typer.Option(
        "gpt-4o", "--judge",
        help="Model to use as judge"
    ),
):
    """
    Run a benchmark comparing LLM models.
    
    Example:
        domainbench run -d waiterbench.jsonl -m openai/gpt-4o -m gemini/gemini-2.0-flash
    """
    from dotenv import load_dotenv
    load_dotenv()
    
    from domainbench.core.config import (
        BenchmarkConfig, ModelConfig, JudgeConfig, 
        BenchmarkSettings, OutputConfig, ProviderType
    )
    from domainbench.core.engine import BenchmarkEngine
    from domainbench.domains import load_domain
    
    # Build config from options or load from file
    if config and config.exists():
        bench_config = BenchmarkConfig.from_yaml(str(config))
    else:
        # Parse model specs
        if not models or len(models) < 2:
            console.print("[red]Error: At least 2 models required for comparison[/red]")
            console.print("Use: -m provider/model -m provider/model")
            raise typer.Exit(1)
        
        model_configs = []
        for model_spec in models:
            parts = model_spec.split("/", 1)
            if len(parts) != 2:
                console.print(f"[red]Invalid model spec: {model_spec}[/red]")
                console.print("Expected format: provider/model (e.g., openai/gpt-4o)")
                raise typer.Exit(1)
            
            provider_str, model_name = parts
            try:
                provider = ProviderType(provider_str.lower())
            except ValueError:
                console.print(f"[red]Unknown provider: {provider_str}[/red]")
                console.print(f"Available: {[p.value for p in ProviderType]}")
                raise typer.Exit(1)
            
            model_configs.append(ModelConfig(
                provider=provider,
                model=model_name,
                alias=f"{provider_str}/{model_name}",
            ))
        
        # Load domain
        try:
            domain_config = load_domain(domain)
        except ValueError as e:
            console.print(f"[red]Error loading domain: {e}[/red]")
            raise typer.Exit(1)
        
        # Build benchmark config
        bench_config = BenchmarkConfig(
            name=f"Benchmark: {' vs '.join([m.display_name for m in model_configs])}",
            models=model_configs,
            domain=domain,
            domain_config=domain_config,
            judge=JudgeConfig(model=judge_model),
            settings=BenchmarkSettings(max_items=max_items),
            output=OutputConfig(directory=str(output)),
        )
    
    # Create and run engine
    console.print(f"\n[bold]Starting benchmark...[/bold]")
    console.print(f"Domain: {bench_config.domain}")
    console.print(f"Dataset: {dataset}")
    console.print(f"Models: {', '.join([m.display_name for m in bench_config.models])}")
    console.print(f"Judge: {bench_config.judge.model}\n")
    
    engine = BenchmarkEngine(bench_config)
    
    try:
        results = engine.run(str(dataset), verbose=True)
        
        # Save results
        output_path = engine.save_results()
        console.print(f"\n[green]Results saved to: {output_path}[/green]")
        
    except Exception as e:
        console.print(f"\n[red]Benchmark failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def generate(
    domain: str = typer.Option(
        "restaurant_waiter", "--domain", "-d",
        help="Domain to generate test cases for"
    ),
    count: int = typer.Option(
        100, "--count", "-n",
        help="Number of test cases to generate"
    ),
    output: Path = typer.Option(
        Path("dataset.jsonl"), "--output", "-o",
        help="Output JSONL file path"
    ),
    seed: int = typer.Option(
        42, "--seed", "-s",
        help="Random seed for reproducibility"
    ),
):
    """
    Generate test cases for a domain.
    
    Example:
        domainbench generate -d restaurant_waiter -n 100 -o waiterbench.jsonl
    """
    import json
    
    console.print(f"\n[bold]Generating test cases...[/bold]")
    console.print(f"Domain: {domain}")
    console.print(f"Count: {count}")
    console.print(f"Seed: {seed}")
    
    # Currently only restaurant_waiter has a generator
    if domain == "restaurant_waiter":
        from domainbench.domains.builtin.restaurant_waiter import generate_test_cases
        items = generate_test_cases(count, seed)
    else:
        console.print(f"[red]No generator available for domain: {domain}[/red]")
        console.print("Currently supported: restaurant_waiter")
        raise typer.Exit(1)
    
    # Write to file
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    console.print(f"\n[green]Generated {len(items)} test cases to: {output}[/green]")


@app.command()
def domains():
    """
    List available domains.
    """
    from rich.table import Table
    from domainbench.domains import list_builtin_domains
    
    table = Table(title="Available Domains")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Type", style="green")
    
    for domain in list_builtin_domains():
        table.add_row(domain["name"], domain["description"], "built-in")
    
    console.print(table)


@app.command()
def capabilities():
    """
    List available benchmark capabilities.
    """
    from rich.table import Table
    from domainbench.capabilities import list_capabilities
    
    table = Table(title="Available Capabilities")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    
    for cap in list_capabilities():
        table.add_row(cap["name"], cap["description"])
    
    console.print(table)


@app.command()
def compare(
    results: List[Path] = typer.Argument(
        ...,
        help="Result JSON files to compare"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output file for comparison report"
    ),
    format: str = typer.Option(
        "table", "--format", "-f",
        help="Output format: table, json, markdown"
    ),
):
    """
    Compare benchmark results.
    
    Example:
        domainbench compare results1.json results2.json
    """
    import json
    from rich.table import Table
    
    all_results = []
    for result_path in results:
        if not result_path.exists():
            console.print(f"[red]File not found: {result_path}[/red]")
            raise typer.Exit(1)
        
        with open(result_path, 'r', encoding='utf-8') as f:
            all_results.append(json.load(f))
    
    if format == "table":
        table = Table(title="Benchmark Comparison")
        table.add_column("Benchmark", style="cyan")
        table.add_column("Winner", style="green")
        table.add_column("Model A Wins", justify="right")
        table.add_column("Model B Wins", justify="right")
        table.add_column("Ties", justify="right")
        
        for result in all_results:
            summary = result.get("summary", {})
            models = list(summary.get("models", {}).keys())
            
            if len(models) >= 2:
                m1_stats = summary["models"][models[0]]
                m2_stats = summary["models"][models[1]]
                
                table.add_row(
                    result.get("benchmark_name", "Unknown"),
                    summary.get("overall_winner", "tie"),
                    str(m1_stats.get("total_wins", 0)),
                    str(m2_stats.get("total_wins", 0)),
                    str(m1_stats.get("total_ties", 0)),
                )
        
        console.print(table)
    
    elif format == "json":
        comparison = {"benchmarks": all_results}
        if output:
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, indent=2)
            console.print(f"[green]Comparison saved to: {output}[/green]")
        else:
            console.print(json.dumps(comparison, indent=2))


@app.command()
def version():
    """
    Show version information.
    """
    from domainbench import __version__
    console.print(f"DomainBench v{__version__}")


def main():
    app()


if __name__ == "__main__":
    main()
