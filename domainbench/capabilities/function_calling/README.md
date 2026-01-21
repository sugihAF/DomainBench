# Function Calling Benchmark

LLM tool use accuracy benchmarks with AST-based validation adapted from Berkeley Function Call Leaderboard (BFCL).

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Commands](#commands)
- [Usage Examples](#usage-examples)
- [Dataset Format](#dataset-format)
- [Evaluation](#evaluation)
- [Results](#results)
- [Categories](#categories)

## Overview

The Function Calling capability benchmarks LLM tool use accuracy across different complexity levels:

- **Simple**: Single function call validation
- **Parallel**: Multiple independent function calls (order doesn't matter)
- **Multiple**: Same function called multiple times (order matters)
- **Multi-turn**: Sequential conversation with state tracking
- **Agentic**: Complex multi-step tasks with text response validation

### Key Features

- **Single model evaluation**: Test one model against ground truth
- **Head-to-head comparison**: Compare two models side-by-side
- **AST-based validation**: Parse and validate Python function call syntax
- **Category-based scoring**: Separate accuracy metrics per category
- **Pre-built domains**: Weather API, task manager, and custom domains
- **Flexible matching**: Strict or lenient parameter validation

### Evaluation Method

Function calling uses **AST-based deterministic validation** against ground truth:

- **Function Name Matching**: Exact match required
- **Parameter Validation**: Type-aware comparison with optional leniency
- **Value Matching**: Supports string normalization, numeric tolerance, and nested structures
- **Order Sensitivity**: Configurable for parallel vs sequential calls

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                      User Query                              │
│                                                              │
│    "What's the weather in New York and Los Angeles?"        │
└─────────────────────────────────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼                               ▼
    ┌─────────────┐                 ┌─────────────┐
    │   Model A   │                 │   Model B   │
    │   (GPT-4o)  │                 │  (Claude)   │
    └─────────────┘                 └─────────────┘
           │                               │
           ▼                               ▼
    ┌─────────────┐                 ┌─────────────┐
    │ Tool Calls  │                 │ Tool Calls  │
    │ Extracted   │                 │ Extracted   │
    └─────────────┘                 └─────────────┘
           │                               │
           └───────────────┬───────────────┘
                           ▼
                  ┌─────────────────┐
                  │  AST Checker    │
                  │ vs Ground Truth │
                  └─────────────────┘
                           │
                           ▼
              Function Name + Parameters
                   = Accuracy Score
```

### Evaluation Process

1. **Load test case**: Query + function definitions + expected calls
2. **Model inference**: LLM generates function/tool calls
3. **Extract tool calls**: Parse provider response format
4. **AST validation**: Compare against ground truth
   - Function name matching
   - Required parameters present
   - Parameter values match
   - Type validation
5. **Calculate accuracy**: Binary correct/incorrect per test case
6. **Aggregate scores**: Per-category and overall accuracy

## Commands

### `domainbench func-call run`

Run a function calling benchmark.

**Syntax:**
```bash
domainbench func-call run [OPTIONS]
```

**Options:**

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--dataset` | `-d` | PATH | Yes | - | Path to dataset JSONL file |
| `--models` | `-m` | TEXT | Yes | - | 1 model (single eval) or 2 models (comparison) |
| `--category` | `-c` | TEXT | - | auto | Category: simple, parallel, multiple, multi_turn, agentic |
| `--output` | `-o` | PATH | - | `./results` | Output directory for results |
| `--max-items` | - | INTEGER | - | - | Maximum number of test cases to run |
| `--strict` | - | FLAG | - | `True` | Strict parameter matching |
| `--lenient` | - | FLAG | - | `False` | Lenient parameter matching |
| `--verbose` | `-v` | FLAG | - | `True` | Show detailed progress |
| `--quiet` | `-q` | FLAG | - | `False` | Suppress progress output |

**Examples:**

```bash
# Single model evaluation
domainbench func-call run \
  -d weather_tests.jsonl \
  -m openai/gpt-4o

# Head-to-head comparison
domainbench func-call run \
  -d weather_tests.jsonl \
  -m openai/gpt-4o \
  -m anthropic/claude-sonnet-4

# Filter by category
domainbench func-call run \
  -d mixed_dataset.jsonl \
  -m openai/gpt-4o \
  -c parallel

# Lenient matching
domainbench func-call run \
  -d dataset.jsonl \
  -m gemini/gemini-2.5-flash \
  --lenient
```

### `domainbench func-call generate`

Generate function calling test cases from a domain.

**Syntax:**
```bash
domainbench func-call generate [OPTIONS]
```

**Options:**

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--domain` | `-d` | TEXT | Yes | - | Domain name (e.g., weather_api, task_manager) |
| `--count` | `-n` | INTEGER | - | `100` | Number of test cases to generate |
| `--category` | `-c` | TEXT | - | `simple` | Category to generate |
| `--output` | `-o` | PATH | - | `dataset.jsonl` | Output JSONL file path |
| `--seed` | `-s` | INTEGER | - | `42` | Random seed for reproducibility |

**Examples:**

```bash
# Generate simple test cases
domainbench func-call generate \
  -d weather_api \
  -n 100 \
  -o weather_tests.jsonl

# Generate parallel test cases
domainbench func-call generate \
  -d weather_api \
  -n 50 \
  -c parallel \
  -o parallel_tests.jsonl

# Generate mixed categories
domainbench func-call generate \
  -d weather_api \
  -n 100 \
  -c all \
  -o mixed_tests.jsonl
```

### `domainbench func-call domains`

List available function calling domains.

```bash
domainbench func-call domains
```

## Usage Examples

### Single Model Evaluation

Test one model's function calling accuracy:

```bash
# Evaluate GPT-4o on weather API calls
domainbench func-call run \
  -d weather_tests.jsonl \
  -m openai/gpt-4o
```

### Head-to-Head Comparison

Compare two models on the same dataset:

```bash
# Compare GPT-4o vs Claude
domainbench func-call run \
  -d weather_tests.jsonl \
  -m openai/gpt-4o \
  -m anthropic/claude-sonnet-4
```

### Category-Specific Testing

Test specific function calling patterns:

```bash
# Test parallel function calling
domainbench func-call run \
  -d dataset.jsonl \
  -m openai/gpt-4o \
  -c parallel

# Test multi-turn conversations
domainbench func-call run \
  -d multi_turn_tests.jsonl \
  -m openai/gpt-4o \
  -c multi_turn
```

### Quick Testing

Limit test cases for faster iteration:

```bash
domainbench func-call run \
  -d large_dataset.jsonl \
  -m openai/gpt-4o \
  --max-items 10
```

## Dataset Format

### JSONL Dataset

One test case per line in JSONL format:

```jsonl
{"id": "fc_001", "category": "simple", "query": "What's the weather?", "functions": [...], "ground_truth": "get_weather(city='NYC')"}
{"id": "fc_002", "category": "parallel", "query": "Weather in NYC and LA?", "functions": [...], "ground_truth": ["get_weather(city='NYC')", "get_weather(city='LA')"]}
```

### Test Case Structure

#### Simple Category

Single function call expected:

```json
{
  "id": "fc_001",
  "category": "simple",
  "query": "What's the weather in New York?",
  "functions": [
    {
      "name": "get_weather",
      "description": "Get current weather for a city",
      "parameters": {
        "type": "object",
        "properties": {
          "city": {"type": "string", "description": "City name"},
          "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["city"]
      }
    }
  ],
  "ground_truth": "get_weather(city='New York')"
}
```

#### Parallel Category

Multiple independent function calls (order doesn't matter):

```json
{
  "id": "fc_002",
  "category": "parallel",
  "query": "What's the weather in New York and Los Angeles?",
  "functions": [...],
  "ground_truth": [
    "get_weather(city='New York')",
    "get_weather(city='Los Angeles')"
  ]
}
```

#### Multiple Category

Same function called multiple times (order matters):

```json
{
  "id": "fc_003",
  "category": "multiple",
  "query": "Get 3-day, 5-day, and 7-day forecasts for Tokyo",
  "functions": [...],
  "ground_truth": [
    "get_forecast(city='Tokyo', days=3)",
    "get_forecast(city='Tokyo', days=5)",
    "get_forecast(city='Tokyo', days=7)"
  ]
}
```

#### Multi-turn Category

Sequential conversation with state tracking:

```json
{
  "id": "fc_004",
  "category": "multi_turn",
  "functions": [...],
  "initial_state": {"tasks": []},
  "turns": [
    {
      "query": "Create a task called 'Buy groceries'",
      "expected_calls": ["create_task(name='Buy groceries')"],
      "expected_state": {"tasks": [{"id": 1, "name": "Buy groceries"}]}
    },
    {
      "query": "Mark that task as complete",
      "expected_calls": ["update_task(id=1, status='complete')"],
      "expected_state": {"tasks": [{"id": 1, "name": "Buy groceries", "status": "complete"}]}
    }
  ]
}
```

#### Agentic Category

Complex tasks with text response validation:

```json
{
  "id": "fc_005",
  "category": "agentic",
  "query": "What is the capital of France?",
  "expected_response": "Paris",
  "match_mode": "contains"
}
```

### Function Definition Format

Functions use OpenAI-compatible schema:

```json
{
  "name": "function_name",
  "description": "What the function does",
  "parameters": {
    "type": "object",
    "properties": {
      "param1": {
        "type": "string",
        "description": "Parameter description"
      },
      "param2": {
        "type": "integer",
        "description": "Another parameter"
      },
      "param3": {
        "type": "string",
        "enum": ["option1", "option2", "option3"]
      }
    },
    "required": ["param1", "param2"]
  }
}
```

## Evaluation

### Scoring System

Function calling uses **binary accuracy scoring**:

- **Correct (1.0)**: Function name and all parameters match
- **Incorrect (0.0)**: Any mismatch in name or required parameters

Accuracy is calculated per category and aggregated:

```
Category Accuracy = Correct Tests / Total Tests × 100%
Overall Accuracy = Sum(Category Correct) / Sum(Category Total) × 100%
```

### Validation Rules

#### Function Name Matching

- Exact string match required
- Case-sensitive comparison

```
Expected: get_weather
Actual:   get_weather  → ✅ Match
Actual:   getWeather   → ❌ No match
Actual:   GET_WEATHER  → ❌ No match
```

#### Parameter Validation

**Strict mode (default):**
- All expected parameters must be present
- Parameter values must match exactly
- Extra parameters cause failure

**Lenient mode:**
- Required parameters must be present
- String comparison is case-insensitive
- Extra parameters are ignored

#### Value Matching

**Strings:**
```python
# Strict mode
"New York" == "New York"     # ✅ Match
"New York" == "new york"     # ❌ No match

# Lenient mode
"New York" == "new york"     # ✅ Match (normalized)
"NYC"      == "New York"     # ❌ No match (different values)
```

**Numbers:**
```python
42 == 42      # ✅ Match
42 == 42.0    # ✅ Match (type coercion)
42 == 43      # ❌ No match
```

**Lists:**
```python
[1, 2, 3] == [1, 2, 3]  # ✅ Match
[1, 2, 3] == [3, 2, 1]  # ❌ No match (order matters)
```

### Category-Specific Validation

#### Simple

- Exactly one function call expected
- First tool call in response is validated

#### Parallel

- Multiple function calls expected
- Order-independent matching (greedy best-match algorithm)
- All expected calls must be present

#### Multiple

- Multiple calls of same function
- Order-dependent validation
- Position-by-position comparison

#### Multi-turn

- Validates each turn sequentially
- Can track state across turns (if execution backend provided)
- Both calls and resulting state can be validated

#### Agentic

- Text response validation (not function calls)
- Supports exact, contains, and regex matching
- Case-insensitive comparison with normalization

## Results

### Single Model Result

```json
{
  "benchmark_type": "function_calling",
  "timestamp": "2025-01-12T10:30:00",
  "config": {
    "models": ["openai/gpt-4o"],
    "category": "simple",
    "strict_mode": true,
    "dataset": "weather_tests.jsonl",
    "test_count": 100
  },
  "summary": {
    "model_metrics": {
      "openai/gpt-4o": {
        "accuracy": 92.0,
        "correct": 92,
        "total": 100,
        "avg_time_ms": 850
      }
    },
    "category_scores": {
      "simple": {"accuracy": 92.0, "correct": 92, "total": 100},
      "overall": {"accuracy": 92.0, "correct": 92, "total": 100}
    }
  },
  "results": [
    {
      "test_id": "fc_001",
      "category": "simple",
      "is_correct": true,
      "score": 100.0,
      "errors": []
    },
    {
      "test_id": "fc_002",
      "category": "simple",
      "is_correct": false,
      "score": 0.0,
      "errors": ["Missing parameter: unit"]
    }
  ]
}
```

### Comparison Result

```json
{
  "benchmark_type": "function_calling",
  "timestamp": "2025-01-12T10:30:00",
  "config": {
    "models": ["openai/gpt-4o", "anthropic/claude-sonnet-4"],
    "category": "mixed",
    "test_count": 100
  },
  "summary": {
    "overall_winner": "openai/gpt-4o",
    "comparison": {
      "A_wins": 45,
      "B_wins": 35,
      "ties": 20
    },
    "model_metrics": {
      "openai/gpt-4o": {
        "accuracy": 92.0,
        "correct": 92,
        "total": 100,
        "avg_time_ms": 850
      },
      "anthropic/claude-sonnet-4": {
        "accuracy": 88.0,
        "correct": 88,
        "total": 100,
        "avg_time_ms": 920
      }
    },
    "category_scores": {
      "simple": {"accuracy": 95.0, "correct": 38, "total": 40},
      "parallel": {"accuracy": 88.0, "correct": 44, "total": 50},
      "multiple": {"accuracy": 100.0, "correct": 10, "total": 10},
      "overall": {"accuracy": 92.0, "correct": 92, "total": 100}
    }
  },
  "results": [...]
}
```

### Metrics Provided

**Per-model metrics:**
- Accuracy (percentage)
- Correct count
- Total count
- Average latency

**Per-category metrics:**
- Category accuracy
- Category correct/total counts

**Comparison metrics (2 models):**
- Win/loss/tie counts
- Overall winner
- Per-test comparison details

## Categories

### Simple

**Purpose**: Validate basic single function calling ability.

**Complexity**: Low

**Example Query**: "What's the weather in Tokyo?"

**Expected Output**: Single function call with correct parameters.

```
get_weather(city='Tokyo')
```

### Parallel

**Purpose**: Test ability to make multiple independent function calls simultaneously.

**Complexity**: Medium

**Example Query**: "Get the weather in Tokyo and the forecast for Paris."

**Expected Output**: Multiple function calls (order doesn't matter).

```
get_weather(city='Tokyo')
get_forecast(city='Paris', days=7)
```

**Validation**: Uses greedy best-match algorithm to pair actual calls with expected calls.

### Multiple

**Purpose**: Test ability to call the same function multiple times with different arguments.

**Complexity**: Medium

**Example Query**: "Get 1-day, 3-day, and 7-day forecasts for London."

**Expected Output**: Same function called multiple times in order.

```
get_forecast(city='London', days=1)
get_forecast(city='London', days=3)
get_forecast(city='London', days=7)
```

**Validation**: Position-by-position comparison (order matters).

### Multi-turn

**Purpose**: Test stateful conversation with sequential function calls.

**Complexity**: High

**Example Conversation**:
```
User: Create a new task called "Buy milk"
Model: create_task(name='Buy milk')

User: Now mark it as complete
Model: update_task(id=1, status='complete')
```

**Validation**: Each turn validated independently, optional state tracking.

### Agentic

**Purpose**: Test complex reasoning where final answer is text, not function calls.

**Complexity**: High

**Example Query**: "Using the search function, find the capital of France."

**Expected Output**: Text containing "Paris"

**Validation**: Text matching with normalization (exact, contains, or regex).

## Pre-built Domains

### Weather API

A comprehensive weather-related function calling domain.

**Functions**:
- `get_current_weather(city, country?, unit?)`
- `get_forecast(city, days, unit?)`
- `get_weather_alerts(region, severity?)`
- `compare_weather(city1, city2, metric?)`

**Categories**: simple, parallel, multiple

**Usage**:
```bash
domainbench func-call generate -d weather_api -n 100 -o tests.jsonl
domainbench func-call run -d tests.jsonl -m openai/gpt-4o
```

### Creating Custom Domains

Create a new domain under `domainbench/domains/builtin/function_calling/`:

```
my_domain/
├── __init__.py
├── domain.yaml
└── generator.py
```

**domain.yaml**:
```yaml
domain:
  name: "My Custom Domain"
  description: "Description of the domain"
  capability: "function_calling"
  categories:
    - simple
    - parallel

  functions:
    - name: "my_function"
      description: "What it does"
      parameters:
        type: "object"
        properties:
          param1:
            type: "string"
            description: "Parameter description"
        required:
          - param1
```

**generator.py**:
```python
def generate_test_cases(count: int, seed: int = 42, category: str = "simple"):
    """Generate test cases for the domain."""
    import random
    rng = random.Random(seed)

    items = []
    for i in range(count):
        items.append({
            "id": f"test_{i:04d}",
            "category": category,
            "query": "User query here",
            "functions": [...],
            "ground_truth": "my_function(param1='value')",
        })

    return items
```

## Best Practices

### Dataset Creation

1. **Diverse queries**: Include variations in phrasing and complexity
2. **Edge cases**: Test optional parameters, boundary values
3. **Realistic scenarios**: Match actual use cases
4. **Balanced categories**: Include mix of simple and complex tests

### Function Definitions

1. **Clear descriptions**: Help models understand function purpose
2. **Explicit types**: Use proper JSON Schema types
3. **Required vs optional**: Only mark truly essential parameters
4. **Enums when appropriate**: Constrain values where possible

### Evaluation Strategy

| Scenario | Recommendation |
|----------|----------------|
| **Initial testing** | Start with simple category, small dataset |
| **Production eval** | Use all categories, 100+ test cases |
| **Model comparison** | Same dataset, same conditions |
| **Debugging** | Enable verbose output, check error details |

### Model Selection

**GPT-4o / GPT-4.1**: Best overall function calling accuracy
**Claude Sonnet 4**: Strong accuracy, good parameter handling
**Gemini 2.5 Flash**: Fast, cost-effective, good for simple cases

## Troubleshooting

### Common Issues

**Issue**: Function calls not extracted from response
```bash
# Check provider response format
# Ensure model supports function calling
# Verify functions are passed correctly
```

**Issue**: Parameter mismatch despite correct values
```bash
# Try lenient mode
domainbench func-call run ... --lenient

# Check string normalization
# "New York" vs "new york" - use lenient mode
```

**Issue**: Wrong function called
```bash
# Check function descriptions are clear
# Verify query is unambiguous
# Review ground truth accuracy
```

**Issue**: Parallel calls matched incorrectly
```bash
# Check for unique function signatures
# Verify ground truth order doesn't matter
# Review greedy matching results
```

## Algorithm Details

### AST-based Validation

Function calls are parsed using Python's AST module:

```python
# Input: "get_weather(city='NYC', unit='celsius')"
# Output: {"name": "get_weather", "arguments": {"city": "NYC", "unit": "celsius"}}
```

### Parallel Matching Algorithm

1. Parse all expected and actual function calls
2. For each expected call, find best matching actual call
3. Use greedy matching by parameter similarity score
4. Mark matches and track unmatched calls
5. Calculate accuracy based on matched pairs

### Multi-turn State Tracking

1. Initialize state from `initial_state`
2. For each turn:
   - Execute model's function calls (if execution backend provided)
   - Update state based on execution
   - Compare actual state with `expected_state`
3. Score based on correct turns / total turns

---

For more information, see the [main README](../../../README.md) or [Chat Completion capability documentation](../chat_completion/README.md).
