# Chat Completion Benchmark

Multi-turn conversation benchmarks with domain-specific evaluation and LLM-as-Judge scoring.

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Commands](#commands)
- [Usage Examples](#usage-examples)
- [Creating Domains](#creating-domains)
- [Dataset Format](#dataset-format)
- [Evaluation](#evaluation)
- [Results](#results)

## Overview

The Chat Completion capability benchmarks LLM models on multi-turn conversations within specific domains (e.g., restaurant waiter, doctor assistant, customer service). It uses an **MT-Bench style** approach where models are compared head-to-head and evaluated by a judge model.

### Key Features

- **Multi-turn conversations**: Test realistic scenarios with 3-6 user turns
- **Domain-specific**: Built-in domains or create custom ones with AI
- **LLM-as-Judge**: Automated evaluation with configurable criteria
- **Swap mitigation**: Reduces position bias by running comparison twice
- **Pairwise comparison**: Compare 2+ models side-by-side

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                      Test Case                               │
│  User Turn 1: "Table for 2, we're in a hurry"               │
│  User Turn 2: "What's the fastest dish?"                    │
│  User Turn 3: "Any gluten-free options?"                    │
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

### Evaluation Process

1. **Load domain**: System prompt, personas, evaluation criteria
2. **Run models**: Both models respond to the same multi-turn conversation
3. **Judge evaluation**: Judge model compares responses based on criteria
4. **Swap & re-evaluate**: Repeat with swapped order to mitigate bias
5. **Aggregate**: Combine results to determine winner

## Commands

### `domainbench chat run`

Run a chat completion benchmark comparing models.

**Syntax:**
```bash
domainbench chat run [OPTIONS]
```

**Options:**

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--dataset` | `-d` | PATH | ✅ | - | Path to dataset JSONL file |
| `--models` | `-m` | TEXT | ✅ | - | Models to compare (format: provider/model) |
| `--domain` | - | TEXT | - | `restaurant_waiter` | Domain name or path to domain config |
| `--config` | `-c` | PATH | - | - | Path to benchmark configuration YAML file |
| `--output` | `-o` | PATH | - | `./results` | Output directory for results |
| `--max-items` | - | INTEGER | - | - | Maximum number of test cases to run |
| `--judge` | - | TEXT | - | `gpt-4o` | Model to use as judge |

**Examples:**

```bash
# Basic comparison
domainbench chat run \
  -d dataset.jsonl \
  -m openai/gpt-4o \
  -m gemini/gemini-2.0-flash

# With custom domain
domainbench chat run \
  -d dataset.jsonl \
  -m openai/gpt-5.2 \
  -m anthropic/claude-4.5-sonnet \
  --domain doctor_assistant \
  --judge gpt-4o

# Limited test cases for quick testing
domainbench chat run \
  -d dataset.jsonl \
  -m openai/gpt-4o \
  -m gemini/gemini-3-flash-preview \
  --max-items 10

# Using config file
domainbench chat run \
  --config benchmark.yaml \
  -d dataset.jsonl
```

---

### `domainbench chat create-domain`

Create a new chat completion domain using AI.

**Syntax:**
```bash
domainbench chat create-domain DESCRIPTION [OPTIONS]
```

**Arguments:**

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `DESCRIPTION` | TEXT | ✅ | Domain description (e.g., "doctor assistant") |

**Options:**

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--provider` | `-p` | TEXT | `openai` | LLM provider for generation |
| `--model` | `-m` | TEXT | `gpt-5.2-2025-12-11` | Model to use for generation |
| `--output-dir` | `-o` | PATH | - | Custom output directory (default: builtin domains) |

**What gets generated:**

1. **domain.yaml** - System prompt, personas, evaluation criteria
2. **generator.py** - Test case generator with 10-15 categories
3. **__init__.py** - Module exports

**Examples:**

```bash
# Simple domain creation
domainbench chat create-domain "doctor assistant"

# With specific provider
domainbench chat create-domain "banking customer service" \
  --provider anthropic \
  --model claude-4.5-sonnet

# Save to custom directory
domainbench chat create-domain "tech support agent" \
  -o ./my_domains

# Use Gemini for generation
domainbench chat create-domain "legal advisor" \
  --provider gemini \
  --model gemini-3-pro-preview
```

---

### `domainbench chat generate`

Generate chat completion test cases for a domain.

**Syntax:**
```bash
domainbench chat generate [OPTIONS]
```

**Options:**

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--domain` | `-d` | TEXT | `restaurant_waiter` | Domain to generate test cases for |
| `--count` | `-n` | INTEGER | `100` | Number of test cases to generate |
| `--output` | `-o` | PATH | `dataset.jsonl` | Output JSONL file path |
| `--seed` | `-s` | INTEGER | `42` | Random seed for reproducibility |

**Examples:**

```bash
# Generate 100 test cases
domainbench chat generate \
  -d restaurant_waiter \
  -n 100 \
  -o waiterbench.jsonl

# Different domain
domainbench chat generate \
  -d doctor_assistant \
  -n 50 \
  -o doctor_test.jsonl

# Custom seed for different variations
domainbench chat generate \
  -d restaurant_waiter \
  -n 100 \
  -s 999 \
  -o variation2.jsonl
```

---

### `domainbench chat convert`

Convert chat test cases from YAML or CSV to JSONL format.

**Syntax:**
```bash
domainbench chat convert INPUT_FILE [OPTIONS]
```

**Arguments:**

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `INPUT_FILE` | PATH | ✅ | Input file path (YAML or CSV format) |

**Options:**

| Option | Short | Type | Description |
|--------|-------|------|-------------|
| `--output` | `-o` | PATH | Output JSONL file path (auto-generated if not provided) |

**Examples:**

```bash
# Convert YAML to JSONL
domainbench chat convert test_cases.yaml

# Convert CSV with custom output
domainbench chat convert test_cases.csv -o dataset.jsonl
```

---

### `domainbench chat domains`

List available chat completion domains.

**Syntax:**
```bash
domainbench chat domains
```

**Example:**

```bash
domainbench chat domains
```

**Output:**
```
Available Chat Completion Domains
┌───────────────────┬─────────────────────────────┬──────────┐
│ Name              │ Description                 │ Type     │
├───────────────────┼─────────────────────────────┼──────────┤
│ restaurant_waiter │ Restaurant waiter assistant │ built-in │
│ doctor_assistant  │ Medical assistant           │ built-in │
└───────────────────┴─────────────────────────────┴──────────┘
```

## Usage Examples

### End-to-End Workflow

```bash
# 1. Create a new domain
domainbench chat create-domain "customer service agent"

# 2. Generate test cases
domainbench chat generate -d customer_service_agent -n 100 -o cs_test.jsonl

# 3. Run benchmark
domainbench chat run \
  -d cs_test.jsonl \
  -m openai/gpt-5.2 \
  -m gemini/gemini-3-flash-preview \
  --domain customer_service_agent

# 4. View results
cat results/benchmark_*.json
```

### Quick Testing with Built-in Domain

```bash
# Generate small dataset
domainbench chat generate -d restaurant_waiter -n 10 -o quick_test.jsonl

# Run quick benchmark
domainbench chat run \
  -d quick_test.jsonl \
  -m openai/gpt-4o \
  -m gemini/gemini-2.0-flash \
  --max-items 5
```

### Using Custom Domain Path

```bash
# Point to custom domain directory
domainbench chat run \
  -d dataset.jsonl \
  -m openai/gpt-4o \
  -m gemini/gemini-2.5-pro \
  --domain ./my_custom_domain/
```

## Creating Domains

### Option 1: AI-Generated (Recommended)

```bash
domainbench chat create-domain "your domain description"
```

This creates:
- System prompt with role definition
- Personas (character archetypes)
- Categories for test case generation
- Evaluation criteria

### Option 2: Manual Creation

Create a `domain.yaml` file:

```yaml
domain:
  name: "My Custom Domain"
  description: "A custom domain for specific use case"
  version: "1.0"

  system_prompt: |
    You are a helpful assistant specializing in...

    Your responsibilities:
    - Task 1
    - Task 2

    Guidelines:
    - Guideline 1
    - Guideline 2

  personas:
    - name: "Persona 1"
      description: "Description of persona 1"
      examples:
        - "Example interaction 1"
        - "Example interaction 2"

    - name: "Persona 2"
      description: "Description of persona 2"
      examples:
        - "Example interaction 1"

  evaluation_criteria:
    - metric: "accuracy"
      description: "How accurate are the responses?"
      weight: 0.3

    - metric: "helpfulness"
      description: "How helpful are the responses?"
      weight: 0.3

    - metric: "tone"
      description: "Is the tone appropriate?"
      weight: 0.2

    - metric: "safety"
      description: "Does it avoid harmful content?"
      weight: 0.2
```

Then create a `generator.py` (optional) for test case generation:

```python
import random
from typing import List, Dict, Any

def generate_test_cases(count: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
    random.seed(seed)
    test_cases = []

    categories = {
        "category_1": [
            # List of scenarios
        ],
        "category_2": [
            # List of scenarios
        ],
    }

    for i in range(count):
        category = random.choice(list(categories.keys()))
        scenario = random.choice(categories[category])

        test_cases.append({
            "id": f"{category}_{i:03d}",
            "category": category,
            "turns": scenario,
        })

    return test_cases
```

## Dataset Format

Chat completion datasets use JSONL format with one test case per line.

### Test Case Structure

```json
{
  "id": "unique_id",
  "category": "category_name",
  "turns": [
    "First user message",
    "Second user message",
    "Third user message"
  ]
}
```

### Example Dataset

```jsonl
{"id": "greeting_001", "category": "greeting", "turns": ["Hi, I need help", "Can you assist me?"]}
{"id": "complex_002", "category": "complex_query", "turns": ["I have a problem", "It's urgent", "What should I do?"]}
{"id": "followup_003", "category": "followup", "turns": ["Thanks for the help", "One more question", "How long will this take?"]}
```

## Evaluation

### Evaluation Criteria

Each domain defines custom evaluation criteria in `domain.yaml`:

```yaml
evaluation_criteria:
  - metric: "accuracy"
    weight: 0.3
  - metric: "helpfulness"
    weight: 0.3
  - metric: "professionalism"
    weight: 0.2
  - metric: "efficiency"
    weight: 0.2
```

### LLM-as-Judge Process

1. **Initial Comparison**: Judge evaluates Model A vs Model B
   - Assigns scores (1-10) for each criterion
   - Determines winner (A/B/Tie)
   - Provides reasoning

2. **Swap Comparison**: Judge evaluates Model B vs Model A (positions swapped)
   - Same evaluation process
   - Mitigates position bias

3. **Aggregation**: Combine both evaluations
   - If both agree: Final winner is determined
   - If disagree: Marked as "Tie"

### Judge Prompt Structure

The judge receives:
- Domain system prompt
- Evaluation criteria
- Both model responses
- Scoring guidelines (1-10 scale)

## Results

### Result Structure

```json
{
  "benchmark_name": "GPT-4o vs Gemini-2.0-Flash",
  "timestamp": "2025-01-12T10:30:00",
  "config": {
    "models": ["openai/gpt-4o", "gemini/gemini-2.0-flash"],
    "domain": "restaurant_waiter",
    "judge": "gpt-4o",
    "test_count": 100
  },
  "summary": {
    "overall_winner": "openai/gpt-4o",
    "models": {
      "openai/gpt-4o": {
        "total_wins": 45,
        "total_losses": 30,
        "total_ties": 25,
        "avg_score": 8.2,
        "avg_latency_ms": 1200
      },
      "gemini/gemini-2.0-flash": {
        "total_wins": 30,
        "total_losses": 45,
        "total_ties": 25,
        "avg_score": 7.8,
        "avg_latency_ms": 950
      }
    },
    "criteria_breakdown": {
      "accuracy": {"model_a_avg": 8.5, "model_b_avg": 8.1},
      "helpfulness": {"model_a_avg": 8.3, "model_b_avg": 7.9},
      "tone": {"model_a_avg": 8.0, "model_b_avg": 7.5}
    }
  },
  "results": [
    {
      "test_id": "greeting_001",
      "category": "greeting",
      "winner": "A",
      "scores": {
        "model_a": 8.5,
        "model_b": 7.8
      },
      "criteria_scores": {
        "accuracy": {"model_a": 9, "model_b": 8},
        "helpfulness": {"model_a": 8, "model_b": 7}
      },
      "swap_result": {
        "winner": "B",
        "final_winner": "A"
      },
      "reasoning": "Model A provided more detailed and helpful response..."
    }
  ]
}
```

### Metrics Provided

- **Win/Loss/Tie counts**: Overall comparison statistics
- **Average scores**: Mean score across all test cases
- **Criteria breakdown**: Performance on each evaluation criterion
- **Latency**: Average response time per model
- **Per-test details**: Individual test case results with reasoning

### Analyzing Results

```bash
# View summary
cat results/benchmark_*.json | jq '.summary'

# Check specific criteria
cat results/benchmark_*.json | jq '.summary.criteria_breakdown'

# See individual test results
cat results/benchmark_*.json | jq '.results[] | select(.winner == "A")'
```

## Best Practices

1. **Test Case Count**: Use 50-100 test cases for reliable results
2. **Domain Selection**: Choose domains matching your use case
3. **Judge Model**: Use a strong model (GPT-4o, Claude-4.5-Opus) as judge
4. **Evaluation Criteria**: Define clear, measurable criteria in domain.yaml
5. **Swap Mitigation**: Always enabled by default - reduces position bias
6. **Seed Control**: Use same seed for reproducible test case generation

## Troubleshooting

### Common Issues

**Issue**: Domain not found
```bash
# Solution: List available domains
domainbench chat domains

# Or use full path
domainbench chat run --domain ./path/to/domain/
```

**Issue**: Generation fails
```bash
# Solution: Check generator.py exists
ls domainbench/domains/your_domain/generator.py

# Or create domain first
domainbench chat create-domain "your domain"
```

**Issue**: Judge API errors
```bash
# Solution: Check API key is set
echo $OPENAI_API_KEY

# Try different judge model
domainbench chat run --judge gpt-4o ...
```

## Advanced Usage

### Custom Config File

Create `benchmark.yaml`:

```yaml
name: "My Benchmark"
models:
  - provider: openai
    model: gpt-4o
    alias: "GPT-4o"
  - provider: gemini
    model: gemini-2.5-pro
    alias: "Gemini-2.5-Pro"

domain: restaurant_waiter

judge:
  model: gpt-4o
  temperature: 0.0

settings:
  max_items: 50
  shuffle: true

output:
  directory: "./results"
  format: json
```

Run with:
```bash
domainbench chat run --config benchmark.yaml -d dataset.jsonl
```

---

For more information, see the [main README](../../../README.md) or [OCR capability documentation](../ocr/README.md).
