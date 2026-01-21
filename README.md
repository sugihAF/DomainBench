# DomainBench

A flexible LLM benchmarking framework for comparing models across multiple capabilities and domains.

## Features

### 1. Chat Completion Benchmark

Multi-turn conversation benchmarks using **MT-Bench style evaluation** with LLM-as-Judge.

- **Multi-turn conversations**: Test models on realistic scenarios with 3-6 user turns
- **Pairwise comparison**: Compare 2+ models side-by-side on the same test cases
- **LLM-as-Judge evaluation**: Automated scoring using a strong judge model (e.g., GPT-4o)
- **Swap-order mitigation**: Reduces position bias by evaluating twice with swapped model positions
- **Domain-specific**: Create custom domains with AI or use built-in ones (restaurant waiter, doctor assistant, etc.)
- **Configurable criteria**: Define custom evaluation metrics (accuracy, helpfulness, tone, safety, etc.)

**[Read detailed documentation →](domainbench/capabilities/chat_completion/README.md)**

### 2. OCR/Vision Extraction Benchmark

Document and image extraction benchmarks with **schema-aware fuzzy matching**.

- **Single or pairwise evaluation**: Test one model against ground truth or compare two models head-to-head
- **PDF & image support**: Handles PNG, JPG, PDF (auto-converts per page), and other formats
- **Schema-aware scoring**: Two-part evaluation system:
  - **Structure score (35%)**: JSON schema validation, required fields, type correctness
  - **Content score (65%)**: Fuzzy string matching with configurable thresholds
- **Identity-based matching**: Smart list comparison using IDs/keys for unordered data
- **Configurable thresholds**: Adjust fuzzy matching sensitivity (0.5-1.0)
- **Pre-built schemas**: Menu, receipt, and document extraction templates

**[Read detailed documentation →](domainbench/capabilities/ocr/README.md)**

### 3. Function Calling Benchmark

Tool use and function calling benchmarks using **BFCL-style AST evaluation**.

- **Five categories**: simple, parallel, multiple, multi-turn, and agentic function calls
- **Single or pairwise evaluation**: Test one model or compare two models head-to-head
- **AST-based validation**: Accurate parsing of Python function call syntax
- **Order-aware matching**: Order-independent for parallel, order-dependent for multiple calls
- **State tracking**: Multi-turn conversations with state validation
- **Pre-built domains**: Weather API and more templates included
- **Custom functions**: Define your own function schemas (OpenAI format)

**[Read detailed documentation →](domainbench/capabilities/function_calling/README.md)**

## Installation

```bash
# Clone the repository
git clone https://github.com/sugihAF/DomainBench.git
cd DomainBench

# Create and activate virtual environment (recommended)
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

# Install domainbench
pip install -e .

# Verify installation
domainbench --help
```

**Alternative: Install without virtual environment**
```bash
# This may require adding the Scripts folder to PATH
pip install .
```

**Note for Windows users**: If you install without a virtual environment and get a PATH warning, either:
1. Use `python -m domainbench` instead of `domainbench`
2. Add the displayed Scripts folder to your system PATH

## Quick Start

### 1. Set up environment variables

Create a `.env` file or export variables:

```bash
export OPENAI_API_KEY=your_openai_key
export GEMINI_API_KEY=your_gemini_key
export ANTHROPIC_API_KEY=your_anthropic_key
```

### 2. Choose your capability

DomainBench is organized by capability type:

#### **Chat Completion Benchmarks**
Multi-turn conversation benchmarks with LLM-as-Judge evaluation.

```bash
# Generate test cases
domainbench chat generate -d restaurant_waiter -n 100 -o dataset.jsonl

# Run benchmark
domainbench chat run -d dataset.jsonl -m openai/gpt-4o -m gemini/gemini-2.0-flash
```

See [capabilities/chat_completion/README.md](domainbench/capabilities/chat_completion/README.md) for detailed documentation.

#### **OCR/Vision Benchmarks**
Document and image extraction benchmarks with schema-aware evaluation.

```bash
# Run single model evaluation
domainbench ocr run -d menu.pdf -gt truth.json -m openai/gpt-4o

# Compare two models
domainbench ocr run -d dataset.jsonl -m openai/gpt-4o -m gemini/gemini-2.5-flash
```

See [capabilities/ocr/README.md](domainbench/capabilities/ocr/README.md) for detailed documentation.

#### **Function Calling Benchmarks**
Tool use benchmarks with AST-based evaluation across multiple categories.

```bash
# Generate test cases from pre-built domain
domainbench func-call generate -d weather_api -n 100 -c simple -o dataset.jsonl

# Run single model evaluation
domainbench func-call run -d dataset.jsonl -m openai/gpt-4o

# Compare two models
domainbench func-call run -d dataset.jsonl -m openai/gpt-4o -m anthropic/claude-sonnet-4
```

See [capabilities/function_calling/README.md](domainbench/capabilities/function_calling/README.md) for detailed documentation.

## CLI Overview

### Main Commands

```bash
domainbench --help                    # Show main help
domainbench capabilities              # List available capabilities
domainbench compare <results...>      # Compare benchmark results
domainbench version                   # Show version info
```

### Capability Commands

Each capability has its own set of commands:

```bash
domainbench chat --help              # Chat completion commands
domainbench ocr --help               # OCR/Vision commands
domainbench func-call --help         # Function calling commands
```

## Supported Providers & Models

| Provider | Models | Status |
|----------|--------|--------|
| **OpenAI** | gpt-4o, gpt-4.1, gpt-5, gpt-5.2, o1, o3, o4-mini | ✅ Ready |
| **Google Gemini** | gemini-2.0-flash, gemini-2.5-pro/flash, gemini-3-pro/flash-preview | ✅ Ready |
| **Anthropic** | claude-3-5-sonnet, claude-sonnet-4, claude-4.5-opus/sonnet/haiku | ✅ Ready |

**Model format**: `provider/model` (e.g., `openai/gpt-5.2`, `gemini/gemini-3-flash-preview`)

## Project Structure

```
domainbench/
├── capabilities/           # Benchmark capabilities
│   ├── chat_completion/   # Chat completion benchmarks
│   │   ├── README.md      # Detailed chat documentation
│   │   └── ...
│   ├── ocr/               # OCR/Vision benchmarks
│   │   ├── README.md      # Detailed OCR documentation
│   │   └── ...
│   └── function_calling/  # Function calling benchmarks
│       ├── README.md      # Detailed function calling documentation
│       ├── checkers/      # AST and validation checkers
│       └── ...
├── core/                  # Engine, config, evaluator
├── providers/             # LLM API adapters
├── domains/               # Domain definitions
└── cli.py                 # Command line interface
```

## Capabilities Documentation

### Chat Completion
Multi-turn conversation benchmarks with domain-specific evaluation criteria and LLM-as-Judge scoring.

**[Read full documentation →](domainbench/capabilities/chat_completion/README.md)**

Features:
- Create custom domains with AI
- Generate multi-turn test cases
- LLM-as-Judge evaluation with swap mitigation
- Domain-specific personas and criteria

### OCR/Vision Extraction
Document and image extraction benchmarks with schema-aware fuzzy matching.

**[Read full documentation →](domainbench/capabilities/ocr/README.md)**

Features:
- Single model evaluation or head-to-head comparison
- PDF and image support (PNG, JPG, etc.)
- Schema-aware JSON validation
- Fuzzy text matching with configurable thresholds
- Structure and content scoring

### Function Calling
Tool use and function calling benchmarks with AST-based evaluation.

**[Read full documentation →](domainbench/capabilities/function_calling/README.md)**

Features:
- Five categories: simple, parallel, multiple, multi-turn, agentic
- Single model evaluation or head-to-head comparison
- AST-based Python function call parsing
- Order-aware matching (parallel vs sequential)
- State tracking for multi-turn conversations
- Pre-built domains (Weather API) and custom function support

## Comparing Results

Compare multiple benchmark runs:

```bash
domainbench compare results/run1.json results/run2.json
domainbench compare results/*.json --format markdown -o comparison.md
```

## Development

### Running Tests

```bash
pytest tests/
```

### Adding a New Capability

1. Create a new directory in `domainbench/capabilities/`
2. Implement `BaseCapability` interface
3. Add CLI commands to `cli.py`
4. Create detailed README.md documentation

## Roadmap

- [x] Chat completion benchmark
- [x] Vision/OCR benchmark
- [x] Function calling benchmark
- [ ] Structured output benchmark
- [ ] Code execution benchmark
- [ ] Web dashboard
- [ ] More built-in domains

## References

### Chat Completion Benchmark

The chat completion benchmark implementation is based on the MT-Bench evaluation methodology:

- **MT-Bench-101**: Multi-turn conversation benchmarking with LLM-as-Judge evaluation
  - GitHub: [https://github.com/mtbench101/mt-bench-101](https://github.com/mtbench101/mt-bench-101)

### OCR/Vision Extraction Benchmark

The OCR evaluation approach draws from established work in:

- **JSON Schema & contract validation**
  - JSON Schema Specification: [https://json-schema.org/](https://json-schema.org/)

- **Tree-based structured diffs**
  - Zhang & Shasha, "Simple Fast Algorithms for the Editing Distance Between Trees and Related Problems" (1989)

- **Entity resolution & bipartite matching**
  - Kuhn, "The Hungarian Method for the Assignment Problem" (1955)

- **OCR-tolerant string similarity**
  - Ratcliff & Metzener, "Pattern Matching: The Gestalt Approach" (1988)

- **Task-oriented information extraction evaluation**
  - Sarawagi, "Information Extraction" (Foundations & Trends in Databases, 2008)

### Function Calling Benchmark

The function calling evaluation methodology is adapted from:

- **Berkeley Function-Calling Leaderboard (BFCL)**
  - GitHub: [https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
  - Paper: Patil et al., "Gorilla: Large Language Model Connected with Massive APIs" (2023)

- **AST-based function call parsing**
  - Python Abstract Syntax Trees: [https://docs.python.org/3/library/ast.html](https://docs.python.org/3/library/ast.html)

- **Function call evaluation categories**
  - Simple: Single function call validation
  - Parallel: Order-independent multiple calls
  - Multiple: Order-dependent sequential calls
  - Multi-turn: Stateful conversation tracking
  - Agentic: Response quality assessment

## License

MIT License

## Support

- **Issues**: [GitHub Issues](https://github.com/sugihAF/DomainBench/issues)
- **Documentation**: See capability-specific READMEs in `domainbench/capabilities/`
