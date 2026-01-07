"""
DomainBench CLI - Command line interface for running benchmarks
"""

import typer
from typing import Optional, List
from pathlib import Path
from rich.console import Console

app = typer.Typer(
    name="domainbench",
    help="""DomainBench - LLM Benchmarking Framework

Supported Providers & Models:

  OpenAI:     gpt-4o, gpt-4.1, gpt-5, gpt-5.2, o1, o3, o4-mini
  
  Gemini:     gemini-2.0-flash, gemini-2.5-pro, gemini-3-flash, gemini-3-pro
  
  Anthropic:  claude-3-5-sonnet, claude-sonnet-4, claude-4.5-opus/sonnet/haiku

Model format: provider/model (e.g., openai/gpt-5.2, gemini/gemini-3-flash)
""",
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
    
    Supported model formats: provider/model
    
    Examples:
        domainbench run -d dataset.jsonl -m openai/gpt-5.2 -m gemini/gemini-3-flash
        domainbench run -d dataset.jsonl -m openai/gpt-4o -m anthropic/claude-4.5-sonnet
        domainbench run -d dataset.jsonl -m gemini/gemini-3-pro -m anthropic/claude-4.5-opus
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


@app.command("create-domain")
def create_domain(
    description: str = typer.Argument(
        ...,
        help="Description of the domain to create (e.g., 'doctor assistant', 'banking customer service')"
    ),
    provider: str = typer.Option(
        "openai", "--provider", "-p",
        help="LLM provider to use for generation (openai, anthropic, gemini)"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="Model to use for generation (default: gpt-5.2-2025-12-11)"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o",
        help="Custom output directory (default: builtin domains)"
    ),
):
    """
    Create a new domain using AI.
    
    This command uses an LLM to generate a complete domain definition including
    domain.yaml and generator.py based on your description.
    
    Supported providers: openai, anthropic, gemini
    
    Examples:
        domainbench create-domain "doctor assistant"
        domainbench create-domain "banking customer service" --provider anthropic --model claude-4.5-sonnet
        domainbench create-domain "tech support agent" --provider gemini --model gemini-3-pro
        domainbench create-domain "legal advisor" --provider openai --model gpt-5.2
    """
    from dotenv import load_dotenv
    load_dotenv()
    
    from domainbench.domains.creator import (
        create_domain_with_ai, 
        validate_generated_domain,
        list_domain_categories,
        DEFAULT_CREATOR_MODEL,
    )
    
    # Use default model if not specified
    if model is None:
        model = DEFAULT_CREATOR_MODEL
    
    console.print(f"\n[bold]Creating domain: {description}[/bold]")
    console.print(f"Using: {provider}/{model}\n")
    
    try:
        with console.status("[bold green]Generating domain files..."):
            domain_path, domain_slug = create_domain_with_ai(
                domain_description=description,
                provider=provider,
                model=model,
                output_dir=output_dir,
            )
        
        console.print(f"[green]✓[/green] Domain files created at: {domain_path}")
        
        # Validate the generated domain
        console.print("\n[dim]Validating generated files...[/dim]")
        is_valid, error = validate_generated_domain(domain_path)
        
        if is_valid:
            console.print(f"[green]✓[/green] Validation passed!")
            
            # Show categories
            categories = list_domain_categories(domain_slug)
            if categories:
                console.print(f"\n[bold]Generated categories ({len(categories)}):[/bold]")
                for cat in categories:
                    console.print(f"  • {cat}")
            
            console.print(f"\n[bold green]Domain '{domain_slug}' is ready to use![/bold green]")
            console.print(f"\nNext steps:")
            console.print(f"  1. Generate test cases: [cyan]domainbench generate -d {domain_slug} -n 100 -o dataset.jsonl[/cyan]")
            console.print(f"  2. Run benchmark: [cyan]domainbench run -d dataset.jsonl -m openai/gpt-5.2 -m gemini/gemini-3-flash --domain {domain_slug}[/cyan]")
        else:
            console.print(f"[yellow]⚠[/yellow] Validation warning: {error}")
            console.print(f"The domain was created but may need manual fixes at: {domain_path}")
            
    except Exception as e:
        console.print(f"\n[red]Error creating domain: {e}[/red]")
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
        domainbench generate -d doctor_assistant -n 50 -o doctor_test.jsonl
    """
    import json
    
    console.print(f"\n[bold]Generating test cases...[/bold]")
    console.print(f"Domain: {domain}")
    console.print(f"Count: {count}")
    console.print(f"Seed: {seed}")
    
    # Dynamically load generator from domain
    items = None
    
    # Check for generator in the domain folder
    from pathlib import Path as PathLib
    from domainbench.domains.loader import BUILTIN_DOMAINS_DIR
    
    # Try builtin domain first
    domain_path = BUILTIN_DOMAINS_DIR / domain
    generator_path = domain_path / "generator.py"
    
    # Also check if domain is a path
    if not generator_path.exists():
        domain_as_path = PathLib(domain)
        if domain_as_path.exists():
            if domain_as_path.is_dir():
                generator_path = domain_as_path / "generator.py"
            else:
                generator_path = domain_as_path.parent / "generator.py"
    
    if generator_path.exists():
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("generator", generator_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if hasattr(module, 'generate_test_cases'):
                items = module.generate_test_cases(count, seed)
                console.print(f"[dim]Loaded generator from: {generator_path}[/dim]")
            else:
                console.print(f"[red]Generator found but missing generate_test_cases function[/red]")
                raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error loading generator: {e}[/red]")
            raise typer.Exit(1)
    else:
        console.print(f"[red]No generator available for domain: {domain}[/red]")
        console.print(f"[dim]Looked for: {generator_path}[/dim]")
        console.print("\nTo create a new domain with AI, use:")
        console.print(f"  [cyan]domainbench create-domain \"{domain}\"[/cyan]")
        raise typer.Exit(1)
    
    # Write to file
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    console.print(f"\n[green]Generated {len(items)} test cases to: {output}[/green]")


@app.command()
def convert(
    input_file: Path = typer.Argument(
        ...,
        help="Input file path (YAML or CSV format)"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output JSONL file path (auto-generated if not provided)"
    ),
):
    """
    Convert test cases from YAML or CSV to JSONL format.
    
    This allows you to create test cases in a more user-friendly format
    and convert them for use with the benchmark engine.
    
    Examples:
        domainbench convert test_cases.yaml
        domainbench convert test_cases.csv -o dataset.jsonl
    
    See examples/templates/ for template files with documentation.
    """
    from domainbench.domains.converter import convert_to_jsonl, detect_format
    
    if not input_file.exists():
        console.print(f"[red]Error: File not found: {input_file}[/red]")
        raise typer.Exit(1)
    
    try:
        format_type = detect_format(str(input_file))
        console.print(f"\n[bold]Converting test cases...[/bold]")
        console.print(f"Input: {input_file} ({format_type.upper()} format)")
        
        output_path, count = convert_to_jsonl(
            str(input_file),
            str(output) if output else None
        )
        
        console.print(f"\n[green]✓ Converted {count} test cases to: {output_path}[/green]")
        console.print(f"\nNext steps:")
        console.print(f"  Run benchmark: [cyan]domainbench run -d {output_path} -m openai/gpt-5.2 -m anthropic/claude-4.5-sonnet[/cyan]")
        
    except Exception as e:
        console.print(f"\n[red]Error converting file: {e}[/red]")
        raise typer.Exit(1)


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
    Compare benchmark results from multiple benchmark runs.
    
    Examples:
        domainbench compare results1.json results2.json
        domainbench compare gpt5_vs_gemini3.json claude45_vs_gpt5.json -f markdown
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
