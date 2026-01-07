# DomainBench

A flexible LLM benchmarking framework for comparing models across multiple capabilities and domains.

## Features

- **Compare LLM Models**: Side-by-side comparison of 2+ models
- **Multiple Capabilities**: Chat completion, function calling, structured output (expanding)
- **AI-Powered Domain Creation**: Generate custom domains from natural language descriptions
- **Custom Domains**: Define your own evaluation domains or use built-in ones
- **LLM-as-Judge**: Automated evaluation with swap-order mitigation
- **Rich Reporting**: JSON, Markdown, and table output formats

## Installation

```bash
# Clone the repository
git clone https://github.com/sugihAF/DomainBench.git
cd DomainBench

# Install in development mode
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

## Quick Start

### 1. Set up environment variables

Create a `.env` file or export variables:

```bash
export OPENAI_API_KEY=your_openai_key
export GEMINI_API_KEY=your_gemini_key
export ANTHROPIC_API_KEY=your_anthropic_key  # if using Claude
```

### 2. Generate test cases

```bash
# Generate 100 restaurant waiter scenarios
domainbench generate -d restaurant_waiter -n 100 -o dataset.jsonl
```

### 3. Run a benchmark

```bash
# Compare GPT-4o vs Gemini Flash
domainbench run \
  -d dataset.jsonl \
  -m openai/gpt-4o \
  -m gemini/gemini-2.0-flash \
  --domain restaurant_waiter \
  --judge gpt-4o
```

### 4. View results

Results are saved to `./results/` by default. You can also use:

```bash
# Compare multiple result files
domainbench compare results/results_*.json
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `domainbench run` | Run a chat benchmark comparing models |
| `domainbench run-ocr` | Run an OCR/vision extraction benchmark |
| `domainbench generate` | Generate test cases for a domain |
| `domainbench create-domain` | Create a new domain using AI |
| `domainbench domains` | List available domains |
| `domainbench capabilities` | List available benchmark capabilities |
| `domainbench compare` | Compare benchmark results |
| `domainbench version` | Show version info |

## Usage Examples

### Using a config file

```bash
domainbench run --config benchmark_config.yaml --dataset dataset.jsonl
```

### Quick comparison with inline options

```bash
domainbench run \
  -d waiterbench.jsonl \
  -m openai/gpt-4o \
  -m anthropic/claude-sonnet-4-20250514 \
  --domain restaurant_waiter \
  --max-items 20
```

## Project Structure

```
domainbench/
├── core/           # Engine, config, evaluator, reporter
├── providers/      # LLM API adapters (OpenAI, Gemini, Anthropic)
├── capabilities/   # Benchmark types (chat_completion, etc.)
├── domains/        # Domain definitions and generators
└── cli.py          # Command line interface
```

## Creating Custom Domains

### Option 1: AI-Powered Creation (Recommended)

Create a complete domain with AI by simply describing what you want:

```bash
# Create a doctor assistant domain
domainbench create-domain "doctor assistant"

# Create with a different provider/model
domainbench create-domain "banking customer service" --provider anthropic --model claude-sonnet-4-20250514

# Save to a custom directory
domainbench create-domain "tech support agent" -o ./my_domains
```

This automatically generates:
- `domain.yaml` - System prompt, personas, evaluation criteria
- `generator.py` - Test case generator with 10-15 categories
- `__init__.py` - Module exports

Then use your new domain:

```bash
# Generate test cases
domainbench generate -d doctor_assistant -n 100 -o doctor_test.jsonl

# Run benchmark
domainbench run -d doctor_test.jsonl -m openai/gpt-4o -m gemini/gemini-2.0-flash --domain doctor_assistant
```

### Option 2: Manual Creation

Create a `domain.yaml` file manually:

```yaml
domain:
  name: "My Custom Domain"
  description: "Description of your domain"
  version: "1.0"
  
  system_prompt: |
    Your system prompt here...
  
  evaluation_criteria:
    - metric: "accuracy"
      weight: 0.5
    - metric: "helpfulness"
      weight: 0.5
```

Then use it:

```bash
domainbench run -d dataset.jsonl -m openai/gpt-4o -m gemini/gemini-2.0-flash --domain ./my_domain/
```

### create-domain Options

| Option | Description |
|--------|-------------|
| `DESCRIPTION` | Domain description (e.g., "doctor assistant") |
| `-p, --provider` | LLM provider for generation (default: openai) |
| `-m, --model` | Model to use (default: gpt-4.1-2025-04-14) |
| `-o, --output-dir` | Custom output directory |

## Supported Providers

| Provider | Models | Status |
|----------|--------|--------|
| OpenAI | gpt-4o, gpt-4-turbo, gpt-3.5-turbo | ✅ Ready |
| Google Gemini | gemini-2.0-flash, gemini-1.5-pro | ✅ Ready |
| Anthropic | claude-3-opus, claude-sonnet-4-20250514 | ✅ Ready |
| Ollama | Local models | 🚧 Planned |

## How It Works

### MT-Bench Style Evaluation

This framework uses an **MT-Bench style** approach:

1. **Multi-turn conversations**: Real scenarios with 3-6 user turns
2. **Pairwise comparison**: Two models respond to the same scenario
3. **LLM-as-Judge**: A strong model (e.g., GPT-4o) evaluates responses
4. **Swap mitigation**: Run comparison twice with swapped order to reduce position bias

```
┌─────────────────────────────────────────────────────────────┐
│                      Test Case                               │
│  User: "Table for 2, we're in a hurry"                      │
│  User: "What's the fastest dish?"                           │
│  User: "Any gluten-free options?"                           │
└─────────────────────────────────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼                               ▼
    ┌─────────────┐                 ┌─────────────┐
    │   Model A   │                 │   Model B   │
    │   (GPT-4o)  │                 │  (Gemini)   │
    └─────────────┘                 └─────────────┘
           │                               │
           └───────────────┬───────────────┘
                           ▼
                  ┌─────────────┐
                  │    Judge    │
                  │  (GPT-4o)   │
                  └─────────────┘
                           │
                           ▼
            Winner: A / B / Tie + Scores
```

## OCR/Vision Benchmark

Benchmark vision models on structured data extraction tasks (menus, receipts, documents).

### Single Model Evaluation

Evaluate one model against ground truth:

```bash
domainbench run-ocr -d menu_dataset.jsonl -m openai/gpt-4o
```

### Two Model Comparison

Compare two models head-to-head:

```bash
domainbench run-ocr -d menu_dataset.jsonl -m openai/gpt-4o -m gemini/gemini-2.0-flash
```

### OCR Options

| Option | Description |
|--------|-------------|
| `-d, --dataset` | JSONL file with image paths and ground truth |
| `-m, --models` | 1 model (single eval) or 2 models (comparison) |
| `-s, --schema` | Schema type: `menu`, `receipt`, `document` |
| `-t, --threshold` | Fuzzy match threshold (default: 0.7) |
| `--max-items` | Limit number of test cases |

### Dataset Format

```json
{"id": "001", "image_path": "menu.png", "ground_truth": {"items": [...], "categories": [...]}}
{"id": "002", "image_paths": ["page1.png", "page2.png"], "ground_truth": {...}}
```

### Evaluation Metrics

- **Precision**: % of extracted items that are correct
- **Recall**: % of ground truth items found
- **F1 Score**: Harmonic mean of precision and recall
- Uses fuzzy text matching (configurable threshold)

## Roadmap

- [x] Chat completion benchmark
- [x] Vision/OCR benchmark
- [ ] Function calling benchmark
- [ ] Structured output benchmark
- [ ] Code execution benchmark
- [ ] Web dashboard
- [ ] More built-in domains

## Development

See [plan.md](plan.md) for the full development roadmap and architecture details.

## License

MIT License

## Contributing

Contributions welcome! Please read the plan.md for architecture guidelines.
