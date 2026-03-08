"""
DomainBench CLI - Command line interface for running benchmarks
"""

import typer
from typing import Optional, List
from pathlib import Path
from rich.console import Console

# Main app
app = typer.Typer(
    name="domainbench",
    help="""DomainBench - LLM Benchmarking Framework

Organized by capability type:
  chat      - Chat completion benchmarks (LLM-as-Judge)
  ocr       - OCR/Vision extraction benchmarks (fuzzy matching)
  func-call - Function calling benchmarks (AST validation)
  voice     - Voice agent benchmarks (multi-turn, two-phase scoring)
  viewer    - Launch web-based result visualization

Supported Providers & Models:
  OpenAI:     gpt-4o, gpt-4.1, gpt-5.2, gpt-5.2-chat-latest, gpt-5.2-codex, gpt-5.2-pro, o1, o3, o4-mini
  Gemini:     gemini-2.0-flash, gemini-2.5-pro/flash, gemini-3-pro-preview, gemini-3-flash-preview
  Anthropic:  claude-3-5-sonnet, claude-sonnet-4, claude-4.5-opus/sonnet/haiku
  Cerebras:   llama3.1-8b, gpt-oss-120b, qwen-3-235b-a22b-instruct-2507, zai-glm-4.7

Model format: provider/model (e.g., openai/gpt-5.2-chat-latest, cerebras/llama3.1-8b)

Note: GPT-5.x and O-series models are reasoning models - temperature is automatically
omitted. They support reasoning_effort and verbosity parameters.
""",
    add_completion=False,
    invoke_without_command=True,
)

# ASCII art banner
BANNER = r"""
    ,---,                        ____                                      ,---,.                                  ,---,
  .'  .' `\                    ,'  , `.             ,--,                 ,'  .'  \                               ,--.' |
,---.'     \    ,---.       ,-+-,.' _ |           ,--.'|         ,---, ,---.' .' |               ,---,           |  |  :
|   |  .`\  |  '   ,'\   ,-+-. ;   , ||           |  |,      ,-+-. /  ||   |  |: |           ,-+-. /  |          :  :  :
:   : |  '  | /   /   | ,--.'|'   |  || ,--.--.   `--'_     ,--.'|'   |:   :  :  /   ,---.  ,--.'|'   |   ,---.  :  |  |,--.
|   ' '  ;  :.   ; ,. :|   |  ,', |  |,/       \  ,' ,'|   |   |  ,"' |:   |    ;   /     \|   |  ,"' |  /     \ |  :  '   |
'   | ;  .  |'   | |: :|   | /  | |--'.--.  .-. | '  | |   |   | /  | ||   :     \ /    /  |   | /  | | /    / ' |  |   /' :
|   | :  |  ''   | .; :|   : |  | ,    \__\/: . . |  | :   |   | |  | ||   |   . |.    ' / |   | |  | |.    ' /  '  :  | | |
'   : | /  ; |   :    ||   : |  |/     ," .--.; | '  : |__ |   | |  |/ '   :  '; |'   ;   /|   | |  |/ '   ; :__ |  |  ' | :
|   | '` ,/   \   \  / |   | |`-'     /  /  ,.  | |  | '.'||   | |--'  |   |  | ; '   |  / |   | |--'  '   | '.'||  :  :_:,'
;   :  .'      `----'  |   ;/        ;  :   .'   \;  :    ;|   |/      |   :   /  |   :    |   |/      |   :    :|  | ,'
|   ,.'                '---'         |  ,     .-./|  ,   / '---'       |   | ,'    \   \  /'---'        \   \  / `--''
'---'                                 `--`---'     ---`-'              `----'       `----'               `----'
"""

console = Console()


@app.callback()
def main_callback(ctx: typer.Context):
    """
    DomainBench - LLM Benchmarking Framework
    """
    if ctx.invoked_subcommand is None:
        console.print(BANNER, style="bold cyan")
        console.print()
        console.print("  Welcome to [bold green]DomainBench[/bold green]!", justify="center")
        console.print("  Your comprehensive LLM benchmarking framework.", justify="center")
        console.print()
        console.print("  Capabilities: [cyan]chat[/cyan] | [yellow]ocr[/yellow] | [magenta]func-call[/magenta] | [blue]voice[/blue] | [green]viewer[/green]", justify="center")
        console.print("  Get started with: [bold yellow]domainbench --help[/bold yellow]", justify="center")
        console.print()

# Chat completion sub-app
chat_app = typer.Typer(
    name="chat",
    help="""Chat completion benchmarks - LLM conversational capabilities

What it does:
  Evaluates multi-turn chat conversations using LLM-as-Judge methodology.
  Compares model responses across different scenarios and domains.

Input needed:
  - Dataset: JSONL file with test cases (conversations with expected behaviors)
  - Models: 2+ models to compare (provider/model format)
  - Domain: Predefined or custom domain configuration

Output:
  - Win/loss/tie statistics per model
  - Detailed evaluation results with judge reasoning
  - Response quality scores and comparison metrics

Examples:
  domainbench chat run -d dataset.jsonl -m openai/gpt-5.2-chat-latest -m gemini/gemini-3-flash-preview
  domainbench chat run -d dataset.jsonl -m openai/gpt-5.2-codex -m anthropic/claude-4.5-sonnet
  domainbench chat generate -d restaurant_waiter -n 100 -o test_cases.jsonl
  domainbench chat create-domain "medical assistant" --provider openai
""",
)

# OCR sub-app
ocr_app = typer.Typer(
    name="ocr",
    help="""OCR/Vision extraction benchmarks - Document and image data extraction

What it does:
  Evaluates structured data extraction from images and PDFs (menus, receipts, documents).
  Uses fuzzy matching against ground truth (not LLM-as-Judge).
  Supports single model evaluation or pairwise comparison.

Input needed:
  - Dataset: JSONL file with image/PDF paths + ground truth, OR single image/PDF file
  - Models: 1 model (evaluation) or 2 models (comparison)
  - Ground truth: JSON file with expected extracted data (when using single file input)
  - Schema: Extraction format (menu, receipt, document, or custom JSON schema)

Output:
  - Extraction accuracy scores (fuzzy match percentage)
  - Field-level precision/recall metrics
  - Head-to-head comparison results (when using 2 models)
  - Parsed extraction data vs ground truth

Note:
  For complex extractions with many items, increase --max-tokens if responses are truncated.

Examples:
  domainbench ocr run -d dataset.jsonl -m openai/gpt-4o
  domainbench ocr run -d menu.pdf -gt truth.json -so schema.json -m openai/gpt-4o
  domainbench ocr run -d receipt.png -gt expected.json -m openai/gpt-4o -m gemini/gemini-2.5-flash
  domainbench ocr run -d invoices.jsonl -m anthropic/claude-4.5-sonnet --max-tokens 32000
""",
)

# Function calling sub-app
func_call_app = typer.Typer(
    name="func-call",
    help="""Function calling benchmarks - LLM tool use accuracy

What it does:
  Evaluates function/tool calling capabilities using AST-based validation.
  Adapted from Berkeley Function Call Leaderboard (BFCL) algorithm.
  Supports single model evaluation or pairwise comparison.

Categories:
  - simple: Single function call validation
  - parallel: Multiple independent function calls (order doesn't matter)
  - multiple: Same function called multiple times (order matters)
  - multi_turn: Sequential conversation with state tracking
  - agentic: Complex multi-step tasks with text response validation

Input needed:
  - Dataset: JSONL file with test cases (queries + expected function calls)
  - Models: 1 model (evaluation) or 2 models (comparison)
  - Category: Which evaluation type to run (auto-detected from dataset if not specified)

Output:
  - Accuracy scores per category
  - Detailed error analysis
  - Head-to-head comparison results (when using 2 models)

Examples:
  domainbench func-call run -d dataset.jsonl -m openai/gpt-4o
  domainbench func-call run -d dataset.jsonl -m openai/gpt-4o -m anthropic/claude-sonnet-4 -c parallel
  domainbench func-call generate -d weather_api -n 100 -o test_cases.jsonl
  domainbench func-call domains
  domainbench func-call domain generate -n "Restaurant Waiter" -d "Restaurant ordering API"
""",
)

# Function calling domain sub-app
func_call_domain_app = typer.Typer(
    name="domain",
    help="""Manage function calling domains - Create new domains with AI

Commands:
  generate  Create a new function calling domain using AI
  list      List all available function calling domains

Examples:
  domainbench func-call domain generate -n "Restaurant Waiter" -d "Restaurant ordering API"
  domainbench func-call domain list
""",
)


# =============================================================================
# CHAT COMPLETION COMMANDS
# =============================================================================

@chat_app.command("run")
def chat_run(
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
    Run a chat completion benchmark comparing LLM models.

    Examples:
        domainbench chat run -d dataset.jsonl -m openai/gpt-5.2-chat-latest -m gemini/gemini-3-flash-preview
        domainbench chat run -d dataset.jsonl -m openai/gpt-5.2-codex -m anthropic/claude-4.5-sonnet
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
    console.print(f"\n[bold]Starting chat completion benchmark...[/bold]")
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


@chat_app.command("create-domain")
def chat_create_domain(
    description: str = typer.Argument(
        ...,
        help="Description of the domain to create (e.g., 'A doctor assistant that helps patients with medical questions')"
    ),
    name: Optional[str] = typer.Option(
        None, "--name", "-n",
        help="Short name for the domain folder (e.g., 'doctor_assistant'). If not provided, generated from description."
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
    Create a new chat completion domain using AI.

    Examples:
        domainbench chat create-domain "A doctor assistant" -n doctor_assistant
        domainbench chat create-domain "banking customer service" -n banking_support --provider anthropic
        domainbench chat create-domain "tech support agent" -n tech_support --model gpt-5.2
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
                domain_name=name,
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
            console.print(f"  1. Generate test cases: [cyan]domainbench chat generate -d {domain_slug} -n 100 -o dataset.jsonl[/cyan]")
            console.print(f"  2. Run benchmark: [cyan]domainbench chat run -d dataset.jsonl -m openai/gpt-5.2-chat-latest -m gemini/gemini-3-flash-preview --domain {domain_slug}[/cyan]")
        else:
            console.print(f"[yellow]⚠[/yellow] Validation warning: {error}")
            console.print(f"The domain was created but may need manual fixes at: {domain_path}")

    except Exception as e:
        console.print(f"\n[red]Error creating domain: {e}[/red]")
        raise typer.Exit(1)


@chat_app.command("generate")
def chat_generate(
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
    Generate chat completion test cases for a domain.

    Examples:
        domainbench chat generate -d restaurant_waiter -n 100 -o waiterbench.jsonl
        domainbench chat generate -d doctor_assistant -n 50 -o doctor_test.jsonl
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
        console.print(f"  [cyan]domainbench chat create-domain \"{domain}\"[/cyan]")
        raise typer.Exit(1)

    # Write to file
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    console.print(f"\n[green]Generated {len(items)} test cases to: {output}[/green]")


@chat_app.command("convert")
def chat_convert(
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
    Convert chat test cases from YAML or CSV to JSONL format.

    Examples:
        domainbench chat convert test_cases.yaml
        domainbench chat convert test_cases.csv -o dataset.jsonl
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
        console.print(f"  Run benchmark: [cyan]domainbench chat run -d {output_path} -m openai/gpt-5.2-chat-latest -m gemini/gemini-3-flash-preview[/cyan]")

    except Exception as e:
        console.print(f"\n[red]Error converting file: {e}[/red]")
        raise typer.Exit(1)


@chat_app.command("domains")
def chat_domains():
    """
    List available chat completion domains.
    """
    from rich.table import Table
    from domainbench.domains import list_builtin_domains

    table = Table(title="Available Chat Completion Domains")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Type", style="green")

    for domain in list_builtin_domains():
        table.add_row(domain["name"], domain["description"], "built-in")

    console.print(table)


# =============================================================================
# OCR COMMANDS
# =============================================================================

@ocr_app.command("run")
def ocr_run(
    dataset: Path = typer.Option(
        ..., "--dataset", "-d",
        help="Path to dataset JSONL file, or a single image/PDF file directly"
    ),
    models: List[str] = typer.Option(
        ..., "--models", "-m",
        help="Model(s) to benchmark (format: provider/model). Use 1 for single eval, 2 for comparison."
    ),
    ground_truth: Optional[Path] = typer.Option(
        None, "--ground-truth", "-gt",
        help="Path to ground truth JSON file (required when -d is a single image/PDF)"
    ),
    output_schema: Optional[Path] = typer.Option(
        None, "--schema-output", "-so",
        help="Path to JSON schema file defining the expected structured output format"
    ),
    schema_type: str = typer.Option(
        "menu", "--schema", "-s",
        help="Extraction schema type: menu, receipt, document, or custom"
    ),
    output: Path = typer.Option(
        Path("./results"), "--output", "-o",
        help="Output directory for results"
    ),
    max_items: Optional[int] = typer.Option(
        None, "--max-items",
        help="Maximum number of test cases to run"
    ),
    threshold: float = typer.Option(
        0.7, "--threshold", "-t",
        help="Fuzzy match threshold for accuracy (0.0-1.0)"
    ),
    pdf_dpi: int = typer.Option(
        150, "--pdf-dpi",
        help="DPI resolution for PDF to image conversion (default: 150)"
    ),
    pdf_max_pages: Optional[int] = typer.Option(
        None, "--pdf-max-pages",
        help="Maximum pages to process per PDF (default: all)"
    ),
    verbose: bool = typer.Option(
        True, "--verbose/--quiet", "-v/-q",
        help="Show detailed progress"
    ),
    max_tokens: int = typer.Option(
        16384, "--max-tokens",
        help="Maximum tokens for model response (default: 16384, increase for complex extractions)"
    ),
):
    """
    Run an OCR/Vision extraction benchmark.

    Supports single model evaluation or two model comparison.
    Uses fuzzy matching against ground truth (not LLM-as-Judge).

    Supported input formats:
        1. JSONL dataset file (multiple test cases)
        2. Single image file (PNG, JPG, etc.) with -gt for ground truth
        3. Single PDF file with -gt for ground truth

    Examples:
        domainbench ocr run -d menu_dataset.jsonl -m openai/gpt-4o
        domainbench ocr run -d menu.pdf -gt truth.json -so schema.json -m openai/gpt-4o
        domainbench ocr run -d receipt.png -gt expected.json -m openai/gpt-4o -m gemini/gemini-2.5-flash
    """
    from dotenv import load_dotenv
    load_dotenv()

    import json
    import time
    from datetime import datetime
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.table import Table

    from domainbench.core.config import ModelConfig, ProviderType
    from domainbench.providers import get_provider
    from domainbench.capabilities.ocr import (
        OCRCapability,
        get_schema_config,
    )

    # Validate model count
    if len(models) < 1 or len(models) > 2:
        console.print("[red]Error: Provide 1 model (single eval) or 2 models (comparison)[/red]")
        raise typer.Exit(1)

    is_comparison = len(models) == 2

    # Parse model specs
    model_configs = []
    providers = {}

    for model_spec in models:
        parts = model_spec.split("/", 1)
        if len(parts) != 2:
            console.print(f"[red]Invalid model spec: {model_spec}[/red]")
            console.print("Expected format: provider/model (e.g., openai/gpt-4o)")
            raise typer.Exit(1)

        provider_str, model_name = parts
        try:
            provider_type = ProviderType(provider_str.lower())
        except ValueError:
            console.print(f"[red]Unknown provider: {provider_str}[/red]")
            console.print(f"Available: {[p.value for p in ProviderType]}")
            raise typer.Exit(1)

        model_config = ModelConfig(
            provider=provider_type,
            model=model_name,
            alias=f"{provider_str}/{model_name}",
        )
        model_configs.append(model_config)

        # Initialize provider
        provider = get_provider(model_config)
        providers[model_config.display_name] = provider

    # Load dataset - supports JSONL or direct image/PDF file
    if not dataset.exists():
        console.print(f"[red]Dataset not found: {dataset}[/red]")
        raise typer.Exit(1)

    # Check if input is a direct image/PDF file or JSONL dataset
    file_ext = dataset.suffix.lower()
    is_direct_file = file_ext in ['.pdf', '.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp']

    # Load output schema if provided
    user_output_schema = None
    if output_schema:
        if not output_schema.exists():
            console.print(f"[red]Output schema file not found: {output_schema}[/red]")
            raise typer.Exit(1)

        try:
            with open(output_schema, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                # Try to parse as JSON first
                try:
                    user_output_schema = json.loads(content)
                except json.JSONDecodeError:
                    # If not valid JSON, use as raw text schema description
                    user_output_schema = content
        except Exception as e:
            console.print(f"[red]Error reading output schema file: {e}[/red]")
            raise typer.Exit(1)

    if is_direct_file:
        # Direct image/PDF file mode
        gt_data = {}

        if ground_truth:
            if not ground_truth.exists():
                console.print(f"[red]Ground truth file not found: {ground_truth}[/red]")
                raise typer.Exit(1)

            # Load ground truth (supports JSON)
            try:
                with open(ground_truth, 'r', encoding='utf-8') as f:
                    gt_data = json.load(f)
            except json.JSONDecodeError as e:
                console.print(f"[red]Invalid JSON in ground truth file: {e}[/red]")
                raise typer.Exit(1)
        else:
            console.print("[yellow]Warning: No ground truth provided (-gt). Running inference only.[/yellow]")

        # Create single test case from the file
        test_case = {
            "id": dataset.stem,  # Use filename without extension as ID
            "ground_truth": gt_data,
        }

        # Add output schema to test case if provided
        if user_output_schema:
            test_case["output_schema"] = user_output_schema

        # Set the appropriate path field based on file type
        if file_ext == '.pdf':
            test_case["pdf_path"] = str(dataset)
        else:
            test_case["image_path"] = str(dataset)

        test_cases = [test_case]
        input_mode = f"Direct file: {dataset.name}"

    else:
        # JSONL dataset mode
        if ground_truth:
            console.print("[yellow]Warning: -gt is ignored when using JSONL dataset (ground truth is in the file)[/yellow]")

        with open(dataset, 'r', encoding='utf-8') as f:
            test_cases = [json.loads(line) for line in f if line.strip()]

        # Apply output schema to all test cases if provided (overrides per-case schemas)
        if user_output_schema:
            for tc in test_cases:
                tc["output_schema"] = user_output_schema

        input_mode = f"Dataset: {dataset} ({len(test_cases)} items)"

    if max_items:
        test_cases = test_cases[:max_items]

    # Get schema config
    schema_config = get_schema_config(schema_type)
    schema_config["threshold"] = threshold

    # Initialize capability with PDF settings
    capability = OCRCapability(
        schema_config=schema_config,
        pdf_dpi=pdf_dpi,
        pdf_max_pages=pdf_max_pages,
    )

    # Count PDF files in dataset for display
    pdf_count = sum(
        1 for tc in test_cases
        if tc.get("pdf_path") or tc.get("pdf_paths")
    )

    # Print header
    console.print(f"\n[bold blue]DomainBench OCR Benchmark[/bold blue]")
    console.print(f"Mode: {'Comparison' if is_comparison else 'Single Model Evaluation'}")
    console.print(f"{input_mode}")
    if pdf_count > 0:
        console.print(f"PDF files: {pdf_count} (DPI: {pdf_dpi}, max pages: {pdf_max_pages or 'all'})")
    console.print(f"Models: {', '.join([m.display_name for m in model_configs])}")
    console.print(f"Evaluation schema: {schema_type} (threshold: {threshold})")
    if user_output_schema:
        schema_preview = str(user_output_schema)[:80] + "..." if len(str(user_output_schema)) > 80 else str(user_output_schema)
        console.print(f"Output schema: [cyan]{output_schema.name}[/cyan]")
    console.print()

    # Results storage
    results = []
    model_metrics = {m.display_name: {"scores": [], "total_time": 0} for m in model_configs}

    if is_comparison:
        comparison_stats = {"A_wins": 0, "B_wins": 0, "ties": 0}

    # Run benchmark
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
        disable=not verbose,
    ) as progress:
        task = progress.add_task("Running OCR benchmark...", total=len(test_cases))

        for idx, test_case in enumerate(test_cases):
            case_id = test_case.get("id", f"case_{idx}")
            ground_truth = test_case.get("ground_truth", {})

            # Build messages with images
            messages = capability.build_messages(test_case, system_prompt="")

            responses = {}

            # Get response from each model
            for model_config in model_configs:
                provider = providers[model_config.display_name]

                start_time = time.time()
                try:
                    response = provider.chat_completion(
                        model=model_config.model,
                        messages=messages,
                        temperature=0.1,
                        max_tokens=max_tokens,
                    )
                    response_text = response.get("content", "")

                    # Check for potential truncation (incomplete JSON)
                    if response_text.strip().startswith("{") or response_text.strip().startswith("```"):
                        # Try to detect truncated JSON
                        open_braces = response_text.count("{") - response_text.count("}")
                        open_brackets = response_text.count("[") - response_text.count("]")
                        if open_braces > 0 or open_brackets > 0:
                            console.print(f"[yellow]Warning: {model_config.display_name} response may be truncated (unbalanced brackets). Consider increasing --max-tokens.[/yellow]")
                except Exception as e:
                    console.print(f"[yellow]Warning: {model_config.display_name} failed on {case_id}: {e}[/yellow]")
                    response_text = "{}"

                elapsed = (time.time() - start_time) * 1000  # ms
                model_metrics[model_config.display_name]["total_time"] += elapsed
                responses[model_config.display_name] = response_text

            # Evaluate
            if is_comparison:
                model_a = model_configs[0].display_name
                model_b = model_configs[1].display_name

                eval_result = capability.evaluate_pair(
                    response_a=responses[model_a],
                    response_b=responses[model_b],
                    ground_truth=ground_truth,
                    schema_config=schema_config,
                )

                # Track wins
                winner = eval_result["winner"]
                if winner == "A":
                    comparison_stats["A_wins"] += 1
                elif winner == "B":
                    comparison_stats["B_wins"] += 1
                else:
                    comparison_stats["ties"] += 1

                model_metrics[model_a]["scores"].append(eval_result["score_A"])
                model_metrics[model_b]["scores"].append(eval_result["score_B"])

                result = {
                    "test_id": case_id,
                    "winner": winner,
                    "scores": {model_a: eval_result["score_A"], model_b: eval_result["score_B"]},
                    "reasons": eval_result.get("reasons", []),
                    "raw_responses": {
                        model_a: responses[model_a],
                        model_b: responses[model_b],
                    },
                    "parsed_extractions": {
                        model_a: eval_result.get("details", {}).get("parsed_a", capability.parse_response(responses[model_a])),
                        model_b: eval_result.get("details", {}).get("parsed_b", capability.parse_response(responses[model_b])),
                    },
                    "ground_truth": ground_truth,
                }
            else:
                # Single model evaluation
                model_name = model_configs[0].display_name
                eval_result = capability.evaluate_single(
                    response=responses[model_name],
                    ground_truth=ground_truth,
                    schema_config=schema_config,
                )

                model_metrics[model_name]["scores"].append(eval_result["overall_score"])

                result = {
                    "test_id": case_id,
                    "score": eval_result["overall_score"],
                    "metrics": eval_result["metrics"],
                    "raw_response": responses[model_name],
                    "parsed_extraction": eval_result.get("parsed_result", {}),
                    "ground_truth": ground_truth,
                }

            results.append(result)
            progress.update(task, advance=1)

    # Calculate summary statistics
    model_summaries = {}
    for model_config in model_configs:
        name = model_config.display_name
        scores = model_metrics[name]["scores"]
        model_summaries[name] = {
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "avg_time_ms": model_metrics[name]["total_time"] / len(test_cases) if test_cases else 0,
        }

    # Determine overall winner for comparison mode
    overall_winner = None
    if is_comparison:
        if comparison_stats["A_wins"] > comparison_stats["B_wins"]:
            overall_winner = model_configs[0].display_name
        elif comparison_stats["B_wins"] > comparison_stats["A_wins"]:
            overall_winner = model_configs[1].display_name
        else:
            overall_winner = "TIE"

    # Display Results Summary
    console.print("\n[bold]Results Summary[/bold]")
    console.print("-" * 50)

    summary_table = Table(title="Model Performance")
    summary_table.add_column("Model", style="cyan")
    summary_table.add_column("Avg Score", justify="right")
    summary_table.add_column("Min", justify="right")
    summary_table.add_column("Max", justify="right")
    summary_table.add_column("Avg Time (ms)", justify="right")

    for model_config in model_configs:
        name = model_config.display_name
        summary = model_summaries[name]
        summary_table.add_row(
            name,
            f"{summary['avg_score']:.1f}%",
            f"{summary['min_score']:.1f}%",
            f"{summary['max_score']:.1f}%",
            f"{summary['avg_time_ms']:.0f}",
        )

    console.print(summary_table)

    # Display comparison results if in comparison mode
    if is_comparison:
        console.print()
        comparison_table = Table(title="Head-to-Head Comparison")
        comparison_table.add_column("Model", style="cyan")
        comparison_table.add_column("Wins", justify="right", style="green")
        comparison_table.add_column("Losses", justify="right", style="red")
        comparison_table.add_column("Ties", justify="right")

        model_a_name = model_configs[0].display_name
        model_b_name = model_configs[1].display_name

        comparison_table.add_row(
            model_a_name,
            str(comparison_stats["A_wins"]),
            str(comparison_stats["B_wins"]),
            str(comparison_stats["ties"]),
        )
        comparison_table.add_row(
            model_b_name,
            str(comparison_stats["B_wins"]),
            str(comparison_stats["A_wins"]),
            str(comparison_stats["ties"]),
        )

        console.print(comparison_table)
        console.print(f"\n[bold]Overall Winner: [green]{overall_winner}[/green][/bold]")

    # Save results
    output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output / f"ocr_results_{timestamp}.json"

    full_results = {
        "benchmark_type": "ocr",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "models": [m.display_name for m in model_configs],
            "schema_type": schema_type,
            "threshold": threshold,
            "dataset": str(dataset),
            "test_count": len(test_cases),
            "output_schema": str(output_schema) if output_schema else None,
        },
        "summary": {
            "model_metrics": model_summaries,
        },
        "results": results,
    }

    if is_comparison:
        full_results["summary"]["comparison"] = comparison_stats
        full_results["summary"]["overall_winner"] = overall_winner

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)

    console.print(f"\n[green]✓ Results saved to: {output_file}[/green]")

    # Also save a separate extraction-only file for easy inspection
    extraction_file = output / f"ocr_extractions_{timestamp}.json"
    extractions = []

    for r in results:
        extraction_entry = {"test_id": r["test_id"]}

        if is_comparison:
            extraction_entry["extractions"] = r.get("parsed_extractions", {})
        else:
            extraction_entry["extraction"] = r.get("parsed_extraction", {})

        extraction_entry["ground_truth"] = r.get("ground_truth", {})
        extractions.append(extraction_entry)

    with open(extraction_file, 'w', encoding='utf-8') as f:
        json.dump(extractions, f, indent=2, ensure_ascii=False)

    console.print(f"[green]✓ Extractions saved to: {extraction_file}[/green]")


# =============================================================================
# FUNCTION CALLING COMMANDS
# =============================================================================

@func_call_app.command("run")
def func_call_run(
    dataset: Path = typer.Option(
        ..., "--dataset", "-d",
        help="Path to dataset JSONL file with function calling test cases"
    ),
    models: List[str] = typer.Option(
        ..., "--models", "-m",
        help="Model(s) to benchmark (format: provider/model). Use 1 for single eval, 2 for comparison."
    ),
    category: Optional[str] = typer.Option(
        None, "--category", "-c",
        help="Evaluation category: simple, parallel, multiple, multi_turn, agentic (auto-detect if not specified)"
    ),
    output: Path = typer.Option(
        Path("./results"), "--output", "-o",
        help="Output directory for results"
    ),
    max_items: Optional[int] = typer.Option(
        None, "--max-items",
        help="Maximum number of test cases to run"
    ),
    strict: bool = typer.Option(
        True, "--strict/--lenient",
        help="Strict parameter matching (default: strict)"
    ),
    verbose: bool = typer.Option(
        True, "--verbose/--quiet", "-v/-q",
        help="Show detailed progress"
    ),
):
    """
    Run a function calling benchmark.

    Evaluates LLM function/tool calling accuracy using AST-based validation.
    Supports single model evaluation or two model comparison.

    Examples:
        domainbench func-call run -d dataset.jsonl -m openai/gpt-4o
        domainbench func-call run -d dataset.jsonl -m openai/gpt-4o -m anthropic/claude-sonnet-4
        domainbench func-call run -d dataset.jsonl -m openai/gpt-4o -c parallel
    """
    from dotenv import load_dotenv
    load_dotenv()

    import json
    import time
    from datetime import datetime
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.table import Table

    from domainbench.core.config import ModelConfig, ProviderType
    from domainbench.providers import get_provider
    from domainbench.capabilities.function_calling import (
        FunctionCallingCapability,
    )
    from domainbench.capabilities.function_calling.function_calling import (
        calculate_category_scores,
    )
    from domainbench.capabilities.function_calling.checkers.utils import (
        parse_tool_calls_from_response,
    )

    # Validate model count
    if len(models) < 1 or len(models) > 2:
        console.print("[red]Error: Provide 1 model (single eval) or 2 models (comparison)[/red]")
        raise typer.Exit(1)

    is_comparison = len(models) == 2

    # Parse model specs
    model_configs = []
    providers = {}

    for model_spec in models:
        parts = model_spec.split("/", 1)
        if len(parts) != 2:
            console.print(f"[red]Invalid model spec: {model_spec}[/red]")
            console.print("Expected format: provider/model (e.g., openai/gpt-4o)")
            raise typer.Exit(1)

        provider_str, model_name = parts
        try:
            provider_type = ProviderType(provider_str.lower())
        except ValueError:
            console.print(f"[red]Unknown provider: {provider_str}[/red]")
            console.print(f"Available: {[p.value for p in ProviderType]}")
            raise typer.Exit(1)

        model_config = ModelConfig(
            provider=provider_type,
            model=model_name,
            alias=f"{provider_str}/{model_name}",
        )
        model_configs.append(model_config)

        # Initialize provider
        provider = get_provider(model_config)
        providers[model_config.display_name] = provider

    # Load dataset
    if not dataset.exists():
        console.print(f"[red]Dataset not found: {dataset}[/red]")
        raise typer.Exit(1)

    with open(dataset, 'r', encoding='utf-8') as f:
        test_cases = [json.loads(line) for line in f if line.strip()]

    if max_items:
        test_cases = test_cases[:max_items]

    # Detect category from dataset if not specified
    if category is None:
        categories_found = set(tc.get("category", "simple") for tc in test_cases)
        if len(categories_found) == 1:
            category = categories_found.pop()
        else:
            category = "mixed"

    # Initialize capability
    capability = FunctionCallingCapability(
        category=category if category != "mixed" else "simple",
        strict_mode=strict,
    )

    # Print header
    console.print(f"\n[bold blue]DomainBench Function Calling Benchmark[/bold blue]")
    console.print(f"Mode: {'Comparison' if is_comparison else 'Single Model Evaluation'}")
    console.print(f"Dataset: {dataset} ({len(test_cases)} items)")
    console.print(f"Category: {category}")
    console.print(f"Models: {', '.join([m.display_name for m in model_configs])}")
    console.print(f"Strict mode: {strict}")
    console.print()

    # Results storage
    results = []
    model_metrics = {m.display_name: {"scores": [], "total_time": 0, "correct": 0} for m in model_configs}

    if is_comparison:
        comparison_stats = {"A_wins": 0, "B_wins": 0, "ties": 0}

    # Run benchmark
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
        disable=not verbose,
    ) as progress:
        task = progress.add_task("Running function calling benchmark...", total=len(test_cases))

        for idx, test_case in enumerate(test_cases):
            case_id = test_case.get("id", f"case_{idx}")
            ground_truth = test_case.get("ground_truth", "")
            functions = test_case.get("functions", [])
            tc_category = test_case.get("category", category if category != "mixed" else "simple")

            # Build messages
            messages = capability.build_messages(test_case, system_prompt="")

            responses = {}

            # Get response from each model
            for model_config in model_configs:
                provider = providers[model_config.display_name]

                start_time = time.time()
                try:
                    # Handle multi_turn category specially - make sequential API calls
                    if tc_category == "multi_turn":
                        turns = test_case.get("turns", [])
                        turn_responses = []
                        conversation = list(messages)  # Copy initial messages

                        for turn_idx, turn in enumerate(turns):
                            # For turn 0, messages already has the first query
                            # For subsequent turns, add the query
                            if turn_idx > 0:
                                conversation.append({
                                    "role": "user",
                                    "content": turn.get("query", ""),
                                })

                            # Make API call for this turn
                            if hasattr(provider, 'function_call') and functions:
                                turn_response = provider.function_call(
                                    model=model_config.model,
                                    messages=conversation,
                                    functions=functions,
                                    temperature=0.1,
                                )
                            else:
                                turn_response = provider.chat_completion(
                                    model=model_config.model,
                                    messages=conversation,
                                    temperature=0.1,
                                    tools=[{"type": "function", "function": f} for f in functions] if functions else None,
                                )

                            turn_responses.append(turn_response)

                            # Add assistant response to conversation for next turn
                            # Extract tool calls to build assistant message
                            tool_calls = turn_response.get("tool_calls", [])
                            assistant_content = turn_response.get("content", "")

                            if tool_calls:
                                # Build assistant message with tool calls
                                # Ensure each tool_call has the required 'type' field for OpenAI API
                                formatted_tool_calls = []
                                for tc in tool_calls:
                                    formatted_tc = {
                                        "id": tc.get("id", f"call_{turn_idx}_{len(formatted_tool_calls)}"),
                                        "type": "function",
                                        "function": tc.get("function", {}),
                                    }
                                    formatted_tool_calls.append(formatted_tc)

                                assistant_msg = {
                                    "role": "assistant",
                                    "content": assistant_content or None,
                                    "tool_calls": formatted_tool_calls,
                                }
                                conversation.append(assistant_msg)

                                # Add simulated tool responses (needed for conversation continuity)
                                for tc in formatted_tool_calls:
                                    tool_call_id = tc.get("id")
                                    conversation.append({
                                        "role": "tool",
                                        "tool_call_id": tool_call_id,
                                        "content": json.dumps({"status": "success"}),
                                    })
                            else:
                                conversation.append({
                                    "role": "assistant",
                                    "content": assistant_content,
                                })

                        response = turn_responses  # List of responses, one per turn
                    else:
                        # Standard single-call handling for other categories
                        if hasattr(provider, 'function_call') and functions:
                            response = provider.function_call(
                                model=model_config.model,
                                messages=messages,
                                functions=functions,
                                temperature=0.1,
                            )
                        else:
                            # Fallback to chat completion with tools
                            response = provider.chat_completion(
                                model=model_config.model,
                                messages=messages,
                                temperature=0.1,
                                tools=[{"type": "function", "function": f} for f in functions] if functions else None,
                            )
                except Exception as e:
                    console.print(f"[yellow]Warning: {model_config.display_name} failed on {case_id}: {e}[/yellow]")
                    response = {"content": "", "tool_calls": []}

                elapsed = (time.time() - start_time) * 1000  # ms
                model_metrics[model_config.display_name]["total_time"] += elapsed
                responses[model_config.display_name] = response

            # Evaluate
            if is_comparison:
                model_a = model_configs[0].display_name
                model_b = model_configs[1].display_name

                eval_result = capability.evaluate_pair(
                    response_a=responses[model_a],
                    response_b=responses[model_b],
                    ground_truth=ground_truth,
                    test_case=test_case,
                )

                # Track wins
                winner = eval_result["winner"]
                if winner == "A":
                    comparison_stats["A_wins"] += 1
                elif winner == "B":
                    comparison_stats["B_wins"] += 1
                else:
                    comparison_stats["ties"] += 1

                model_metrics[model_a]["scores"].append(eval_result["score_A"])
                model_metrics[model_b]["scores"].append(eval_result["score_B"])

                if eval_result["eval_A"].get("is_correct"):
                    model_metrics[model_a]["correct"] += 1
                if eval_result["eval_B"].get("is_correct"):
                    model_metrics[model_b]["correct"] += 1

                # Extract tool calls for detailed view
                tool_calls_detail = {}
                for mc in model_configs:
                    mn = mc.display_name
                    resp = responses[mn]
                    if tc_category == "multi_turn" and isinstance(resp, list):
                        tool_calls_detail[mn] = [parse_tool_calls_from_response(tr) for tr in resp]
                    else:
                        tool_calls_detail[mn] = parse_tool_calls_from_response(resp)

                result = {
                    "test_id": case_id,
                    "category": tc_category,
                    "winner": winner,
                    "scores": {model_a: eval_result["score_A"], model_b: eval_result["score_B"]},
                    "is_correct": {
                        model_a: eval_result["eval_A"].get("is_correct", False),
                        model_b: eval_result["eval_B"].get("is_correct", False),
                    },
                    "errors": {
                        model_a: eval_result["eval_A"].get("errors", []),
                        model_b: eval_result["eval_B"].get("errors", []),
                    },
                    "tool_calls": tool_calls_detail,
                    "reasons": eval_result.get("reasons", []),
                }

                # Save expected calls / ground truth
                if tc_category == "multi_turn":
                    turns_data = test_case.get("turns", [])
                    result["expected_calls"] = [t.get("expected_calls", []) for t in turns_data]
                else:
                    result["ground_truth"] = ground_truth
            else:
                # Single model evaluation
                model_name = model_configs[0].display_name
                eval_result = capability.evaluate_single(
                    response=responses[model_name],
                    ground_truth=ground_truth,
                    test_case=test_case,
                )

                model_metrics[model_name]["scores"].append(eval_result["score"])
                if eval_result.get("is_correct"):
                    model_metrics[model_name]["correct"] += 1

                # Extract tool calls for detailed view
                resp = responses[model_name]
                if tc_category == "multi_turn" and isinstance(resp, list):
                    single_tool_calls = [parse_tool_calls_from_response(tr) for tr in resp]
                else:
                    single_tool_calls = parse_tool_calls_from_response(resp)

                result = {
                    "test_id": case_id,
                    "category": tc_category,
                    "is_correct": eval_result.get("is_correct", False),
                    "score": eval_result["score"],
                    "errors": eval_result.get("errors", []),
                    "tool_calls": single_tool_calls,
                }

                # Save expected calls / ground truth
                if tc_category == "multi_turn":
                    turns_data = test_case.get("turns", [])
                    result["expected_calls"] = [t.get("expected_calls", []) for t in turns_data]
                else:
                    result["ground_truth"] = ground_truth

            results.append(result)
            progress.update(task, advance=1)

    # Calculate summary statistics
    model_summaries = {}
    for model_config in model_configs:
        name = model_config.display_name
        scores = model_metrics[name]["scores"]
        model_summaries[name] = {
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "accuracy": (model_metrics[name]["correct"] / len(test_cases) * 100) if test_cases else 0,
            "correct": model_metrics[name]["correct"],
            "total": len(test_cases),
            "avg_time_ms": model_metrics[name]["total_time"] / len(test_cases) if test_cases else 0,
        }

    # Determine overall winner for comparison mode
    overall_winner = None
    if is_comparison:
        if comparison_stats["A_wins"] > comparison_stats["B_wins"]:
            overall_winner = model_configs[0].display_name
        elif comparison_stats["B_wins"] > comparison_stats["A_wins"]:
            overall_winner = model_configs[1].display_name
        else:
            overall_winner = "TIE"

    # Display Results Summary
    console.print("\n[bold]Results Summary[/bold]")
    console.print("-" * 50)

    summary_table = Table(title="Model Performance")
    summary_table.add_column("Model", style="cyan")
    summary_table.add_column("Accuracy", justify="right")
    summary_table.add_column("Correct", justify="right")
    summary_table.add_column("Total", justify="right")
    summary_table.add_column("Avg Time (ms)", justify="right")

    for model_config in model_configs:
        name = model_config.display_name
        summary = model_summaries[name]
        summary_table.add_row(
            name,
            f"{summary['accuracy']:.1f}%",
            str(summary['correct']),
            str(summary['total']),
            f"{summary['avg_time_ms']:.0f}",
        )

    console.print(summary_table)

    # Display comparison results if in comparison mode
    if is_comparison:
        console.print()
        comparison_table = Table(title="Head-to-Head Comparison")
        comparison_table.add_column("Model", style="cyan")
        comparison_table.add_column("Wins", justify="right", style="green")
        comparison_table.add_column("Losses", justify="right", style="red")
        comparison_table.add_column("Ties", justify="right")

        model_a_name = model_configs[0].display_name
        model_b_name = model_configs[1].display_name

        comparison_table.add_row(
            model_a_name,
            str(comparison_stats["A_wins"]),
            str(comparison_stats["B_wins"]),
            str(comparison_stats["ties"]),
        )
        comparison_table.add_row(
            model_b_name,
            str(comparison_stats["B_wins"]),
            str(comparison_stats["A_wins"]),
            str(comparison_stats["ties"]),
        )

        console.print(comparison_table)
        console.print(f"\n[bold]Overall Winner: [green]{overall_winner}[/green][/bold]")

    # Calculate per-category scores
    category_scores = calculate_category_scores(results)
    if len(category_scores) > 2:  # More than just "overall" and one category
        console.print()
        cat_table = Table(title="Per-Category Accuracy")
        cat_table.add_column("Category", style="cyan")
        cat_table.add_column("Accuracy", justify="right")
        cat_table.add_column("Correct", justify="right")
        cat_table.add_column("Total", justify="right")

        for cat_name, cat_stats in category_scores.items():
            cat_table.add_row(
                cat_name,
                f"{cat_stats['accuracy']:.1f}%",
                str(int(cat_stats['correct'])),
                str(int(cat_stats['total'])),
            )

        console.print(cat_table)

    # Save results
    output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output / f"func_call_results_{timestamp}.json"

    full_results = {
        "benchmark_type": "function_calling",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "models": [m.display_name for m in model_configs],
            "category": category,
            "strict_mode": strict,
            "dataset": str(dataset),
            "test_count": len(test_cases),
        },
        "summary": {
            "model_metrics": model_summaries,
            "category_scores": category_scores,
        },
        "results": results,
    }

    if is_comparison:
        full_results["summary"]["comparison"] = comparison_stats
        full_results["summary"]["overall_winner"] = overall_winner

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)

    console.print(f"\n[green]Results saved to: {output_file}[/green]")


@func_call_app.command("generate")
def func_call_generate(
    domain: str = typer.Option(
        ..., "--domain", "-d",
        help="Domain to generate test cases for (e.g., weather_api, task_manager)"
    ),
    count: int = typer.Option(
        100, "--count", "-n",
        help="Number of test cases to generate"
    ),
    category: str = typer.Option(
        "simple", "--category", "-c",
        help="Category type to generate: simple, parallel, multiple, multi_turn, agentic"
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
    Generate function calling test cases for a domain.

    Examples:
        domainbench func-call generate -d weather_api -n 100 -o weather_test.jsonl
        domainbench func-call generate -d task_manager -n 50 -c multi_turn -o tasks.jsonl
    """
    import json
    from pathlib import Path as PathLib

    console.print(f"\n[bold]Generating function calling test cases...[/bold]")
    console.print(f"Domain: {domain}")
    console.print(f"Category: {category}")
    console.print(f"Count: {count}")
    console.print(f"Seed: {seed}")

    # Look for generator in builtin domains
    from domainbench.domains.loader import BUILTIN_DOMAINS_DIR

    # Check function_calling subdirectory
    domain_path = BUILTIN_DOMAINS_DIR / "function_calling" / domain
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
                items = module.generate_test_cases(count, seed, category)
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
        console.print("\nAvailable domains:")
        console.print("  [cyan]domainbench func-call domains[/cyan]")
        raise typer.Exit(1)

    # Write to file
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    console.print(f"\n[green]Generated {len(items)} test cases to: {output}[/green]")


@func_call_app.command("domains")
def func_call_domains():
    """
    List available function calling domains.
    """
    from rich.table import Table
    from pathlib import Path as PathLib

    from domainbench.domains.loader import BUILTIN_DOMAINS_DIR

    table = Table(title="Available Function Calling Domains")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Categories", style="green")

    # Check function_calling subdirectory
    fc_domains_dir = BUILTIN_DOMAINS_DIR / "function_calling"

    if fc_domains_dir.exists():
        for domain_dir in fc_domains_dir.iterdir():
            if domain_dir.is_dir() and (domain_dir / "domain.yaml").exists():
                import yaml
                with open(domain_dir / "domain.yaml", 'r') as f:
                    config = yaml.safe_load(f)

                domain_info = config.get("domain", {})
                name = domain_info.get("name", domain_dir.name)
                description = domain_info.get("description", "")
                categories = domain_info.get("categories", ["simple"])

                table.add_row(
                    domain_dir.name,
                    description,
                    ", ".join(categories),
                )

    if table.row_count == 0:
        console.print("[yellow]No function calling domains found.[/yellow]")
        console.print("\nTo create a domain, add a folder under:")
        console.print(f"  {fc_domains_dir}")
        console.print("\nWith files: domain.yaml, generator.py, __init__.py")
    else:
        console.print(table)


# =============================================================================
# FUNCTION CALLING DOMAIN COMMANDS
# =============================================================================

@func_call_domain_app.command("generate")
def func_call_domain_generate(
    name: str = typer.Option(
        ..., "--name", "-n",
        help="Name of the domain to create (e.g., 'Restaurant Waiter', 'Banking API')"
    ),
    description: Optional[str] = typer.Option(
        None, "--description", "-d",
        help="Description of the domain (auto-generated if not provided)"
    ),
    provider: str = typer.Option(
        "openai", "--provider", "-p",
        help="LLM provider to use for generation (openai, anthropic, gemini)"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="Model to use for generation (default: gpt-5.2)"
    ),
    categories: Optional[str] = typer.Option(
        None, "--categories", "-c",
        help="Comma-separated categories to support (default: simple,parallel,multiple)"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o",
        help="Custom output directory (default: builtin domains)"
    ),
):
    """
    Create a new function calling domain using AI.

    Generates a complete domain with:
    - domain.yaml (function definitions)
    - generator.py (test case generator)
    - __init__.py (module exports)

    Examples:
        domainbench func-call domain generate -n "Restaurant Waiter"
        domainbench func-call domain generate -n "Banking API" -d "Banking transaction functions"
        domainbench func-call domain generate -n "Task Manager" -c "simple,parallel"
    """
    from dotenv import load_dotenv
    load_dotenv()

    from domainbench.capabilities.function_calling.domain_creator import (
        create_domain_with_ai,
        validate_generated_domain,
        DEFAULT_CREATOR_MODEL,
    )

    # Use default model if not specified
    if model is None:
        model = DEFAULT_CREATOR_MODEL

    # Parse categories
    category_list = None
    if categories:
        category_list = [c.strip() for c in categories.split(",")]

    console.print(f"\n[bold]Creating function calling domain: {name}[/bold]")
    console.print(f"Using: {provider}/{model}")
    if description:
        console.print(f"Description: {description}")
    console.print()

    try:
        with console.status("[bold green]Generating domain files..."):
            domain_path, domain_slug = create_domain_with_ai(
                domain_name=name,
                domain_description=description or "",
                provider=provider,
                model=model,
                categories=category_list,
                output_dir=output_dir,
            )

        console.print(f"[green]✓[/green] Domain files created at: {domain_path}")

        # Validate the generated domain
        console.print("\n[dim]Validating generated files...[/dim]")
        is_valid, error = validate_generated_domain(domain_path)

        if is_valid:
            console.print(f"[green]✓[/green] Validation passed!")
            console.print(f"\n[bold green]Domain '{domain_slug}' is ready to use![/bold green]")
            console.print(f"\nNext steps:")
            console.print(f"  1. Generate test cases: [cyan]domainbench func-call generate -d {domain_slug} -n 100 -o dataset.jsonl[/cyan]")
            console.print(f"  2. Run benchmark: [cyan]domainbench func-call run -d dataset.jsonl -m openai/gpt-4o[/cyan]")
        else:
            console.print(f"[yellow]⚠[/yellow] Validation warning: {error}")
            console.print(f"The domain was created but may need manual fixes at: {domain_path}")

    except Exception as e:
        console.print(f"\n[red]Error creating domain: {e}[/red]")
        raise typer.Exit(1)


@func_call_domain_app.command("list")
def func_call_domain_list():
    """
    List all available function calling domains.

    Shows domain name, description, categories, and function count.
    """
    from rich.table import Table

    from domainbench.capabilities.function_calling.domain_creator import (
        list_function_calling_domains,
    )

    domains = list_function_calling_domains()

    if not domains:
        console.print("[yellow]No function calling domains found.[/yellow]")
        console.print("\nTo create a new domain with AI, use:")
        console.print("  [cyan]domainbench func-call domain generate -n \"Domain Name\"[/cyan]")
        return

    table = Table(title="Function Calling Domains")
    table.add_column("Slug", style="cyan")
    table.add_column("Name")
    table.add_column("Description")
    table.add_column("Categories", style="green")
    table.add_column("Functions", justify="right")

    for domain in domains:
        table.add_row(
            domain["slug"],
            domain["name"],
            domain["description"][:50] + "..." if len(domain["description"]) > 50 else domain["description"],
            ", ".join(domain["categories"]),
            str(domain["function_count"]),
        )

    console.print(table)


# =============================================================================
# GLOBAL COMMANDS
# =============================================================================

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
    table.add_column("Command", style="green")

    for cap in list_capabilities():
        cmd = f"domainbench {cap['name'].replace('_', '-')}"
        table.add_row(cap["name"], cap["description"], cmd)

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
        domainbench compare gpt5_vs_gemini25.json claude45_vs_gpt5.json -f markdown
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


@app.command()
def viewer(
    results_dir: Optional[Path] = typer.Option(
        None, "--results", "-r",
        help="Directory containing benchmark results (default: ./results)"
    ),
    host: str = typer.Option(
        "127.0.0.1", "--host", "-h",
        help="Host to bind the server to"
    ),
    port: int = typer.Option(
        5000, "--port", "-p",
        help="Port to run the server on"
    ),
    debug: bool = typer.Option(
        False, "--debug",
        help="Run in debug mode"
    ),
):
    """
    Launch the web-based result viewer.

    Opens a local web interface to visualize benchmark results with charts
    and detailed breakdowns for each capability (chat, ocr, function calling).

    Examples:
      domainbench viewer
      domainbench viewer -r ./my_results -p 8080
    """
    from domainbench.viewer import run_viewer

    results_path = str(results_dir) if results_dir else None
    run_viewer(results_dir=results_path, host=host, port=port, debug=debug)


# ---------------------------------------------------------------------------
# Voice benchmark sub-app
# ---------------------------------------------------------------------------

voice_app = typer.Typer(
    name="voice",
    help="""Voice agent benchmarks - Multi-turn evaluation with tool use

What it does:
  Evaluates LLMs as voice agents using multi-turn sequential conversations.
  Based on the aiewf-eval benchmark methodology with LLM-as-Judge and
  two-phase realignment scoring.

  Scores each turn on 4 binary (pass/fail) dimensions:
    - Tool use correctness
    - Instruction following
    - Knowledge base grounding
    - Turn-taking (audio pipelines only)

Evaluation modes:
  - Single model: Absolute pass/fail scoring
  - Pairwise: Compare two models on the same scenario
  - Multi-model: Leaderboard across N models

Pipeline types:
  - Text: LLM only (default, no audio)
  - Cascaded: STT -> LLM -> TTS (via pipeline config YAML)
  - Speech-to-speech: End-to-end models (via pipeline config YAML)

Examples:
  domainbench voice run -d dataset.jsonl -m openai/gpt-4o
  domainbench voice run -d dataset.jsonl -m openai/gpt-4o -m anthropic/claude-sonnet-4
  domainbench voice run -d dataset.jsonl -m openai/gpt-4o --runs 5
  domainbench voice generate -n "Hotel Concierge" -d "A luxury hotel assistant" -o scenario.jsonl
  domainbench voice generate --builtin hotel_concierge -o scenario.jsonl
  domainbench voice domains
""",
)


def _default_api_key_env(provider: str) -> str:
    """Map a provider name to its default API key environment variable."""
    return {
        "openai": "OPENAI_API_KEY",
        "openai_realtime": "OPENAI_API_KEY",
        "whisper": "OPENAI_API_KEY",
        "deepgram": "DEEPGRAM_API_KEY",
        "google": "GOOGLE_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "elevenlabs": "ELEVENLABS_API_KEY",
        "cartesia": "CARTESIA_API_KEY",
        "azure": "AZURE_API_KEY",
    }.get(provider.lower(), f"{provider.upper()}_API_KEY")


@voice_app.command("run")
def voice_run(
    dataset: Path = typer.Option(
        ..., "-d", "--dataset",
        help="Path to JSONL dataset with voice scenarios",
    ),
    models: Optional[List[str]] = typer.Option(
        None, "-m", "--model",
        help="Model(s) to evaluate (provider/model format). Can specify multiple.",
    ),
    pipeline: Optional[Path] = typer.Option(
        None, "-p", "--pipeline",
        help="Path to pipeline config YAML (for cascaded or speech-to-speech).",
    ),
    judge_model: str = typer.Option(
        "openai/gpt-4o", "-j", "--judge",
        help="Judge model for evaluation (provider/model format).",
    ),
    num_runs: int = typer.Option(
        1, "--runs",
        help="Number of repeated runs per scenario (for consistency measurement).",
    ),
    output_dir: Path = typer.Option(
        Path("./results"), "-o", "--output",
        help="Directory to save results.",
    ),
    max_scenarios: Optional[int] = typer.Option(
        None, "--max-scenarios",
        help="Maximum number of scenarios to evaluate from the dataset.",
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Print detailed progress."),
    save_audio: bool = typer.Option(
        False, "--save-audio",
        help="Save intermediate audio files (TTS input, STT input, TTS response).",
    ),
):
    """
    Run voice agent benchmark on a dataset.

    Executes each scenario's multi-turn conversation sequentially, then
    scores with LLM-as-Judge using two-phase realignment.

    Examples:
      domainbench voice run -d scenario.jsonl -m openai/gpt-4o
      domainbench voice run -d scenario.jsonl -m openai/gpt-4o -m anthropic/claude-sonnet-4 --runs 3
    """
    import json as json_mod
    from datetime import datetime
    from dotenv import load_dotenv
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.table import Table

    load_dotenv()

    from domainbench.providers import get_provider
    from domainbench.core.config import ModelConfig, ProviderType
    from domainbench.capabilities.voice.config import VoiceScenario, PipelineConfig
    from domainbench.capabilities.voice.engine import VoiceEngine
    from domainbench.capabilities.voice.judge import VoiceJudge
    from domainbench.capabilities.voice.scorer import aggregate_runs

    if not dataset.exists():
        console.print(f"[red]Error: Dataset not found: {dataset}[/red]")
        raise typer.Exit(1)

    # Load pipeline config first (needed to validate model requirements)
    pipeline_config = PipelineConfig(type="text")
    if pipeline:
        import yaml
        with open(pipeline, "r", encoding="utf-8") as f:
            pipeline_config = PipelineConfig(**yaml.safe_load(f))

    # Parse model specs — for S2S pipeline, -m is optional (model comes from YAML)
    # For cascaded, -m overrides the LLM in YAML; if not given, uses YAML llm config
    model_configs = []
    if models:
        for spec in models:
            if "/" not in spec:
                console.print(f"[red]Error: Model must be 'provider/model', got: {spec}[/red]")
                raise typer.Exit(1)
            provider_str, model_name = spec.split("/", 1)
            try:
                ptype = ProviderType(provider_str.lower())
            except ValueError:
                console.print(f"[red]Error: Unknown provider '{provider_str}'. Use openai/anthropic/gemini/cerebras.[/red]")
                raise typer.Exit(1)
            model_configs.append(ModelConfig(provider=ptype, model=model_name))
    elif pipeline_config.type == "speech_to_speech":
        # S2S: model comes from pipeline YAML, no -m needed
        if not pipeline_config.model:
            console.print("[red]Error: S2S pipeline YAML must have 'model' section.[/red]")
            raise typer.Exit(1)
        # Create a dummy model config for iteration (engine uses s2s_instance directly)
        prov_str = pipeline_config.model.provider
        try:
            ptype = ProviderType(prov_str.lower())
        except ValueError:
            ptype = ProviderType.OPENAI  # fallback
        model_configs.append(ModelConfig(provider=ptype, model=pipeline_config.model.model))
    elif pipeline_config.type == "cascaded" and pipeline_config.llm:
        # Cascaded: use LLM from YAML config
        prov_str = pipeline_config.llm.provider
        try:
            ptype = ProviderType(prov_str.lower())
        except ValueError:
            console.print(f"[red]Error: Unknown LLM provider '{prov_str}' in pipeline YAML.[/red]")
            raise typer.Exit(1)
        model_configs.append(ModelConfig(
            provider=ptype,
            model=pipeline_config.llm.model,
            api_key_env=pipeline_config.llm.api_key_env,
        ))
    else:
        console.print("[red]Error: At least one model is required (-m provider/model)[/red]")
        raise typer.Exit(1)

    # Parse judge model
    if "/" not in judge_model:
        console.print(f"[red]Error: Judge model must be 'provider/model', got: {judge_model}[/red]")
        raise typer.Exit(1)
    judge_provider_str, judge_model_name = judge_model.split("/", 1)
    try:
        judge_ptype = ProviderType(judge_provider_str.lower())
    except ValueError:
        console.print(f"[red]Error: Unknown judge provider '{judge_provider_str}'.[/red]")
        raise typer.Exit(1)
    judge_provider = get_provider(ModelConfig(provider=judge_ptype, model=judge_model_name))

    # Load scenarios from dataset
    scenarios = []
    with open(dataset, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json_mod.loads(line)
                scenarios.append(VoiceScenario(**data))
    if max_scenarios:
        scenarios = scenarios[:max_scenarios]

    if not scenarios:
        console.print("[red]Error: No scenarios found in dataset.[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold blue]DomainBench Voice Benchmark[/bold blue]")
    console.print(f"Scenarios: {len(scenarios)} | Models: {len(model_configs)} | Runs: {num_runs}\n")

    # Initialize pipeline services based on config type
    from domainbench.capabilities.voice.stt import create_stt
    from domainbench.capabilities.voice.tts import create_tts
    from domainbench.capabilities.voice.s2s import create_s2s

    stt_instance = None
    tts_instance = None
    input_tts_instance = None
    s2s_instance = None
    pipeline_components = {}  # tracks actual provider/model per stage

    if pipeline_config.type == "cascaded":
        # Initialize STT
        if pipeline_config.stt:
            stt_kwargs = {
                "api_key_env": pipeline_config.stt.api_key_env or _default_api_key_env(pipeline_config.stt.provider),
                "model": pipeline_config.stt.model,
            }
            if pipeline_config.stt.params:
                stt_kwargs["params"] = pipeline_config.stt.params
            stt_instance = create_stt(pipeline_config.stt.provider, **stt_kwargs)
            pipeline_components["stt"] = f"{pipeline_config.stt.provider}/{pipeline_config.stt.model}"
            console.print(f"  STT: {pipeline_config.stt.provider}/{pipeline_config.stt.model}")
        else:
            console.print("[red]Error: Cascaded pipeline requires 'stt' configuration.[/red]")
            raise typer.Exit(1)

        # Initialize response TTS
        if pipeline_config.tts:
            tts_kwargs = {
                "api_key_env": pipeline_config.tts.api_key_env or _default_api_key_env(pipeline_config.tts.provider),
                "model": pipeline_config.tts.model,
            }
            if pipeline_config.tts.voice_id:
                tts_kwargs["voice_id"] = pipeline_config.tts.voice_id
            if pipeline_config.tts.params:
                tts_kwargs["params"] = pipeline_config.tts.params
            tts_instance = create_tts(pipeline_config.tts.provider, **tts_kwargs)
            pipeline_components["tts"] = f"{pipeline_config.tts.provider}/{pipeline_config.tts.model}"
            console.print(f"  TTS: {pipeline_config.tts.provider}/{pipeline_config.tts.model}")

        # Initialize input TTS (for synthesizing user audio from text)
        if pipeline_config.input_tts:
            it_cfg = pipeline_config.input_tts
            input_tts_kwargs = {
                "api_key_env": it_cfg.api_key_env or _default_api_key_env(it_cfg.provider),
                "model": it_cfg.model,
            }
            if hasattr(it_cfg, "voice") and it_cfg.voice:
                input_tts_kwargs["voice"] = it_cfg.voice
            if it_cfg.params:
                input_tts_kwargs["params"] = it_cfg.params
            input_tts_instance = create_tts(it_cfg.provider, **input_tts_kwargs)
        elif tts_instance:
            # Fall back to response TTS for input synthesis
            input_tts_instance = tts_instance

        console.print(f"  Pipeline: [bold]cascaded[/bold] (STT -> LLM -> TTS)")
        # LLM component comes from model configs (set below per model iteration)

    elif pipeline_config.type == "speech_to_speech":
        # Initialize S2S model
        if pipeline_config.model:
            s2s_kwargs = {
                "api_key_env": pipeline_config.model.api_key_env or _default_api_key_env(pipeline_config.model.provider),
                "model": pipeline_config.model.model,
            }
            if hasattr(pipeline_config.model, "voice") and pipeline_config.model.voice:
                s2s_kwargs["voice"] = pipeline_config.model.voice
            if hasattr(pipeline_config.model, "audio_format") and pipeline_config.model.audio_format:
                s2s_kwargs["audio_format"] = pipeline_config.model.audio_format
            if hasattr(pipeline_config.model, "params") and pipeline_config.model.params:
                s2s_kwargs["params"] = pipeline_config.model.params
            s2s_instance = create_s2s(pipeline_config.model.provider, **s2s_kwargs)
            pipeline_components["s2s"] = f"{pipeline_config.model.provider}/{pipeline_config.model.model}"
            console.print(f"  S2S: {pipeline_config.model.provider}/{pipeline_config.model.model}")
        else:
            console.print("[red]Error: Speech-to-speech pipeline requires 'model' configuration.[/red]")
            raise typer.Exit(1)

        # Initialize input TTS (for synthesizing user audio from text)
        if pipeline_config.input_tts:
            it_cfg = pipeline_config.input_tts
            input_tts_kwargs = {
                "api_key_env": it_cfg.api_key_env or _default_api_key_env(it_cfg.provider),
                "model": it_cfg.model,
            }
            if hasattr(it_cfg, "voice") and it_cfg.voice:
                input_tts_kwargs["voice"] = it_cfg.voice
            if it_cfg.params:
                input_tts_kwargs["params"] = it_cfg.params
            input_tts_instance = create_tts(it_cfg.provider, **input_tts_kwargs)
        elif pipeline_config.tts:
            tts_kwargs = {
                "api_key_env": pipeline_config.tts.api_key_env or _default_api_key_env(pipeline_config.tts.provider),
                "model": pipeline_config.tts.model,
            }
            if pipeline_config.tts.voice_id:
                tts_kwargs["voice_id"] = pipeline_config.tts.voice_id
            if pipeline_config.tts.params:
                tts_kwargs["params"] = pipeline_config.tts.params
            input_tts_instance = create_tts(pipeline_config.tts.provider, **tts_kwargs)
        else:
            # Default to OpenAI TTS for input synthesis
            try:
                input_tts_instance = create_tts("openai", api_key_env="OPENAI_API_KEY")
            except ValueError:
                console.print(
                    "[red]Error: S2S pipeline needs TTS for input audio. "
                    "Configure 'input_tts' in YAML or set OPENAI_API_KEY.[/red]"
                )
                raise typer.Exit(1)

        console.print(f"  Pipeline: [bold]speech_to_speech[/bold] (end-to-end)")

    else:
        console.print(f"  Pipeline: [bold]text[/bold] (LLM only)")

    # Audio directory setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = "_vs_".join(mc.model for mc in model_configs)
    audio_dir = None
    if save_audio:
        audio_dir = output_dir / "audio" / f"voice_{model_tag}_{timestamp}"
        console.print(f"  Audio save: [bold]enabled[/bold] → {audio_dir}")

    # Run benchmark for each model
    all_model_results = {}

    for mc in model_configs:
        model_display = f"{mc.provider.value}/{mc.model}"
        console.print(f"\n[cyan]Model: {model_display}[/cyan]")

        # Track LLM component for cascaded pipeline
        if pipeline_config.type in ("cascaded", "text"):
            pipeline_components["llm"] = model_display

        provider = get_provider(mc)
        judge = VoiceJudge(judge_provider, judge_model_name)
        engine = VoiceEngine(
            provider=provider,
            model=mc.model,
            judge=judge,
            pipeline_config=pipeline_config,
            stt=stt_instance,
            tts=tts_instance,
            input_tts=input_tts_instance,
            s2s=s2s_instance,
            save_audio=save_audio,
            audio_dir=audio_dir,
        )

        model_run_results = []

        with Progress(
            SpinnerColumn("line", style="cyan"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
            disable=verbose,
        ) as progress:
            total = len(scenarios) * num_runs
            task = progress.add_task(f"Running {model_display}...", total=total)

            for scenario in scenarios:
                runs = engine.run_multiple(scenario, num_runs=num_runs, verbose=verbose)
                model_run_results.extend(runs)
                progress.update(task, advance=num_runs)

        # Aggregate results per scenario
        scenario_groups = {}
        for rr in model_run_results:
            scenario_groups.setdefault(rr.scenario_id, []).append(rr)

        aggregated = []
        for sid, group in scenario_groups.items():
            agg = aggregate_runs(group)
            aggregated.append(agg)

        all_model_results[model_display] = {
            "runs": [r.model_dump() for r in model_run_results],
            "aggregated": aggregated,
        }

    # Print results table
    console.print("\n[bold green]--- Results ---[/bold green]\n")
    table = Table(title="Voice Benchmark Results")
    table.add_column("Model", style="cyan")
    table.add_column("Pass Rate", justify="right")
    table.add_column("Tool Use", justify="right")
    table.add_column("Instruction", justify="right")
    table.add_column("KB Ground", justify="right")
    table.add_column("TTFB (med)", justify="right")
    table.add_column("V2V (med)", justify="right")
    table.add_column("Sil.Pad", justify="right")
    table.add_column("Runs", justify="right")

    def _avg_across_scenarios(agg_list):
        """Average metrics across multiple scenarios."""
        if not agg_list:
            return None
        if len(agg_list) == 1:
            return agg_list[0]
        import statistics as _st
        rates = [a.get("pass_rate_median", 0) for a in agg_list]
        dim_keys = ["tool_use_correct", "instruction_following", "kb_grounding", "turn_taking"]
        dims_avg = {}
        for k in dim_keys:
            vals = [a.get("dimension_scores", {}).get(k, 0) for a in agg_list]
            dims_avg[k] = round(_st.mean(vals), 1)
        # Aggregate all latency keys
        latency_agg = {}
        lat_keys = [
            "ttfb_median_ms", "v2v_median_ms", "silence_pad_mean_ms",
            "tool_v2v_mean_ms", "non_tool_v2v_median_ms", "non_tool_v2v_max_ms",
        ]
        for lk in lat_keys:
            vals = [a.get("latency", {}).get(lk, 0) for a in agg_list if a.get("latency", {}).get(lk)]
            if vals:
                latency_agg[lk] = round(_st.mean(vals), 1)
        total_runs = sum(a.get("num_runs", 0) for a in agg_list)
        return {
            "pass_rate_median": round(_st.mean(rates), 1),
            "dimension_scores": dims_avg,
            "latency": latency_agg,
            "num_runs": total_runs,
            "num_scenarios": len(agg_list),
        }

    for model_display, data in all_model_results.items():
        agg_list = data["aggregated"]
        summary = _avg_across_scenarios(agg_list)
        if summary:
            dims = summary.get("dimension_scores", {})
            lat = summary.get("latency", {})
            table.add_row(
                model_display,
                f"{summary.get('pass_rate_median', 0):.1f}%",
                f"{dims.get('tool_use_correct', 0):.1f}%",
                f"{dims.get('instruction_following', 0):.1f}%",
                f"{dims.get('kb_grounding', 0):.1f}%",
                f"{lat.get('ttfb_median_ms', 0):.0f}ms" if lat.get('ttfb_median_ms') else "N/A",
                f"{lat.get('v2v_median_ms', 0):.0f}ms" if lat.get('v2v_median_ms') else "N/A",
                f"{lat.get('silence_pad_mean_ms', 0):.0f}ms" if lat.get('silence_pad_mean_ms') else "N/A",
                str(summary.get("num_runs", 0)),
            )

    console.print(table)

    # Determine winner (if pairwise)
    if len(model_configs) == 2:
        names = list(all_model_results.keys())
        s_a = _avg_across_scenarios(all_model_results[names[0]]["aggregated"])
        s_b = _avg_across_scenarios(all_model_results[names[1]]["aggregated"])
        rate_a = s_a.get("pass_rate_median", 0) if s_a else 0
        rate_b = s_b.get("pass_rate_median", 0) if s_b else 0
        if abs(rate_a - rate_b) < 2.0:
            console.print(f"\n[bold yellow]Result: Tie ({names[0]} vs {names[1]})[/bold yellow]")
        elif rate_a > rate_b:
            console.print(f"\n[bold yellow]Winner: {names[0]}[/bold yellow]")
        else:
            console.print(f"\n[bold yellow]Winner: {names[1]}[/bold yellow]")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"voice_{model_tag}_{timestamp}.json"

    full_results = {
        "benchmark_type": "voice",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "models": [f"{mc.provider.value}/{mc.model}" for mc in model_configs],
            "judge": judge_model,
            "num_runs": num_runs,
            "pipeline_type": pipeline_config.type,
            "dataset": str(dataset),
            "audio_dir": str(audio_dir) if audio_dir else None,
            "pipeline_components": pipeline_components if pipeline_components else None,
        },
        "results": all_model_results,
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json_mod.dump(full_results, f, indent=2, ensure_ascii=False, default=str)

    console.print(f"\nResults saved to: [bold]{result_path}[/bold]")


@voice_app.command("generate")
def voice_generate(
    domain_name: Optional[str] = typer.Option(
        None, "-n", "--name",
        help="Name for the custom domain (used with --description).",
    ),
    domain_description: Optional[str] = typer.Option(
        None, "-d", "--description",
        help="Description of the voice agent to generate a scenario for.",
    ),
    builtin: Optional[str] = typer.Option(
        None, "--builtin", "-b",
        help="Use a built-in domain template (e.g., 'hotel_concierge').",
    ),
    num_turns: int = typer.Option(
        20, "--turns", "-t",
        help="Target number of conversation turns.",
    ),
    num_scenarios: int = typer.Option(
        1, "--scenarios",
        help="Number of scenarios to generate.",
    ),
    generator_model: str = typer.Option(
        "openai/gpt-4o", "--model", "-m",
        help="Model to use for AI generation (provider/model format).",
    ),
    output: Path = typer.Option(
        Path("./voice_dataset.jsonl"), "-o", "--output",
        help="Output JSONL file path.",
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for built-in generators."),
):
    """
    Generate voice agent evaluation scenarios.

    Two modes:
      1. Built-in template: --builtin hotel_concierge
      2. AI-generated: --name "Domain Name" --description "what the agent does"

    Examples:
      domainbench voice generate --builtin hotel_concierge -o scenario.jsonl
      domainbench voice generate -n "Tech Support" -d "IT helpdesk agent" -o tech.jsonl
      domainbench voice generate -n "Clinic" -d "Medical receptionist" --turns 25 -o clinic.jsonl
    """
    import json as json_mod
    from dotenv import load_dotenv

    load_dotenv()

    from domainbench.capabilities.voice.generator import (
        generate_builtin_hotel,
        generate_with_ai,
    )

    scenarios = []

    if builtin:
        # Built-in template generation
        builtin_map = {
            "hotel_concierge": generate_builtin_hotel,
        }
        gen_func = builtin_map.get(builtin.lower())
        if gen_func is None:
            console.print(f"[red]Error: Unknown built-in domain '{builtin}'. Available: {list(builtin_map.keys())}[/red]")
            raise typer.Exit(1)

        console.print(f"Generating built-in '{builtin}' scenario ({num_turns} turns)...")
        for i in range(num_scenarios):
            s = gen_func(
                num_turns=num_turns,
                seed=seed + i,
                scenario_id=f"voice_{builtin}_{i + 1:03d}",
            )
            scenarios.append(s)

    elif domain_name and domain_description:
        # AI-powered generation
        if "/" not in generator_model:
            console.print(f"[red]Error: Model must be 'provider/model', got: {generator_model}[/red]")
            raise typer.Exit(1)
        prov_str, mod_name = generator_model.split("/", 1)

        from domainbench.providers import get_provider
        from domainbench.core.config import ModelConfig, ProviderType

        try:
            ptype = ProviderType(prov_str.lower())
        except ValueError:
            console.print(f"[red]Error: Unknown provider '{prov_str}'.[/red]")
            raise typer.Exit(1)

        provider = get_provider(ModelConfig(provider=ptype, model=mod_name))

        for i in range(num_scenarios):
            console.print(f"Generating scenario {i + 1}/{num_scenarios} via {generator_model}...")
            s = generate_with_ai(
                domain_name=domain_name,
                domain_description=domain_description,
                provider=provider,
                model=mod_name,
                num_turns=num_turns,
                scenario_id=f"voice_{domain_name.lower().replace(' ', '_')}_{i + 1:03d}",
            )
            scenarios.append(s)
            console.print(f"  Created: {s.id} ({len(s.turns)} turns, {len(s.tools)} tools)")
    else:
        console.print("[red]Error: Provide either --builtin or both --name and --description.[/red]")
        raise typer.Exit(1)

    # Save to JSONL
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for s in scenarios:
            f.write(json_mod.dumps(s.model_dump(), ensure_ascii=False) + "\n")

    console.print(f"\n[bold green]Generated {len(scenarios)} scenario(s) -> {output}[/bold green]")
    for s in scenarios:
        tool_call_turns = sum(1 for t in s.turns if t.required_function_call)
        console.print(f"  {s.id}: {len(s.turns)} turns, {tool_call_turns} tool calls, {len(s.tools)} tools")


@voice_app.command("domains")
def voice_domains():
    """
    List available built-in voice domains.

    Shows pre-built scenario templates that can be used with
    'domainbench voice generate --builtin <name>'.
    """
    from rich.table import Table

    from domainbench.capabilities.voice.generator import list_builtin_domains

    domains = list_builtin_domains()

    table = Table(title="Built-in Voice Domains")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Turns", justify="right")
    table.add_column("Tool Calls", justify="right")

    for d in domains:
        table.add_row(d["name"], d["description"], d["turns"], d["tool_calls"])

    console.print(table)


# Register sub-apps
app.add_typer(chat_app, name="chat")
app.add_typer(ocr_app, name="ocr")
app.add_typer(func_call_app, name="func-call")
app.add_typer(voice_app, name="voice")

# Register nested sub-apps
func_call_app.add_typer(func_call_domain_app, name="domain")


def main():
    app()


if __name__ == "__main__":
    main()
