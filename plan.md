# DomainBench Framework - Development Plan

> A flexible, extensible LLM benchmarking framework for comparing models across multiple capabilities and domains.

---

## 1. Vision & Goals

**Primary Goal**: Create a framework that allows regular users to benchmark and compare different LLM models across various capabilities and custom domains.

**Key Objectives**:
- Compare 2+ LLM models side-by-side
- Support multiple benchmark capabilities (not just chat completion)
- Allow users to define custom domains (not limited to restaurant/waiter scenarios)
- Provide clear, actionable insights from benchmark results

---

## 2. Core Design Principles

| Principle | Description |
|-----------|-------------|
| **Modularity** | Each capability (chat, function calling, OCR, etc.) should be a separate plugin |
| **Domain Agnostic** | Users define their own domains via configuration/templates |
| **Provider Agnostic** | Support multiple LLM providers (OpenAI, Anthropic, local models, etc.) |
| **Extensibility** | Easy to add new benchmark types without modifying core code |
| **Reproducibility** | Results should be reproducible with seed control and versioning |

---

## 3. Supported Capabilities

### Phase 1 (MVP)
- [x] Chat Completion (current implementation)
- [ ] Function Calling
- [ ] Structured Output (JSON mode)

### Phase 2
- [ ] Tool Use (Search as a tool)
- [ ] Code Execution
- [ ] Multi-turn Conversation

### Phase 3
- [ ] Vision / OCR
- [ ] Embeddings Quality
- [ ] Reasoning / Chain-of-Thought
- [ ] Agent-based Tasks

---

## 4. Project Architecture

```
DomainBench/
├── domainbench/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── engine.py           # Main orchestrator
│   │   ├── evaluator.py        # Base evaluation logic
│   │   ├── reporter.py         # Results aggregation & export
│   │   └── config.py           # Configuration management
│   │
│   ├── providers/              # LLM Provider Adapters
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract provider interface
│   │   ├── openai_provider.py
│   │   ├── anthropic_provider.py
│   │   ├── ollama_provider.py  # Local models
│   │   └── custom.py           # User-defined providers
│   │
│   ├── capabilities/           # Benchmark Types (Plugins)
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract capability interface
│   │   ├── chat_completion.py
│   │   ├── function_calling.py
│   │   ├── structured_output.py
│   │   ├── tool_use.py         # Search, code execution, etc.
│   │   ├── vision_ocr.py
│   │   └── reasoning.py
│   │
│   ├── domains/                # Domain Templates
│   │   ├── __init__.py
│   │   ├── schema.py           # Domain definition schema
│   │   ├── loader.py           # Load domain configurations
│   │   └── builtin/            # Built-in domain templates
│   │       ├── restaurant_waiter/
│   │       ├── customer_support/
│   │       └── coding_assistant/
│   │
│   ├── metrics/                # Evaluation Metrics
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── accuracy.py
│   │   ├── latency.py
│   │   ├── cost.py
│   │   ├── token_efficiency.py
│   │   └── domain_specific.py  # Custom scoring functions
│   │
│   └── generators/             # Test Case Generation
│       ├── __init__.py
│       ├── base.py
│       ├── synthetic.py        # LLM-generated test cases
│       └── template.py         # Template-based generation
│
├── cli/                        # Command Line Interface
│   ├── __init__.py
│   ├── __main__.py
│   ├── commands/
│   │   ├── init.py             # Initialize new domain/project
│   │   ├── run.py              # Run benchmarks
│   │   ├── compare.py          # Compare results
│   │   ├── generate.py         # Generate test cases
│   │   └── report.py           # Generate reports
│   └── interactive.py          # Interactive mode (guided setup)
│
├── web/                        # Optional Web UI (Future)
│   ├── api/
│   └── dashboard/
│
├── configs/                    # Configuration Files
│   ├── providers.yaml          # Provider configurations
│   └── default_settings.yaml   # Default benchmark settings
│
├── domains/                    # User-defined domains (created by users)
│   └── .gitkeep
│
├── results/                    # Benchmark results output
│   └── .gitkeep
│
├── examples/                   # Example configurations & usage
│   ├── benchmark_configs/
│   └── domain_configs/
│
├── tests/                      # Unit & integration tests
│
├── docs/                       # Documentation
│   ├── getting_started.md
│   ├── creating_domains.md
│   ├── adding_capabilities.md
│   └── api_reference.md
│
├── plan.md                     # This file
├── README.md
├── pyproject.toml              # Package configuration
└── requirements.txt
```

---

## 5. Key Component Specifications

### 5.1 Capability Plugin Interface

Each capability must implement this interface:

```
┌─────────────────────────────────────────────────────────┐
│                    BaseCapability                        │
├─────────────────────────────────────────────────────────┤
│ + name: str                                             │
│ + description: str                                      │
│ + required_provider_features: List[str]                 │
├─────────────────────────────────────────────────────────┤
│ + prepare_prompt(test_case, domain_context) → Prompt    │
│ + execute(provider, prompt) → Response                  │
│ + evaluate(response, expected) → Score                  │
│ + get_metrics() → List[Metric]                          │
│ + validate_config(config) → bool                        │
└─────────────────────────────────────────────────────────┘
         △                    △                    △
         │                    │                    │
   ChatCompletion      FunctionCalling        VisionOCR
```

### 5.2 Provider Adapter Interface

```
┌─────────────────────────────────────────────────────────┐
│                    BaseProvider                          │
├─────────────────────────────────────────────────────────┤
│ + name: str                                             │
│ + supported_features: List[str]                         │
├─────────────────────────────────────────────────────────┤
│ + chat_completion(messages, **kwargs) → Response        │
│ + function_call(messages, functions, **kwargs) → Resp   │
│ + structured_output(messages, schema, **kwargs) → Resp  │
│ + vision(messages, images, **kwargs) → Response         │
│ + get_usage() → UsageStats                              │
│ + estimate_cost(usage) → float                          │
└─────────────────────────────────────────────────────────┘
```

### 5.3 Domain Definition Schema

Users define domains via YAML configuration:

```yaml
# domains/healthcare_triage/domain.yaml
domain:
  name: "Healthcare Triage Assistant"
  description: "AI assistant for initial patient symptom assessment"
  version: "1.0"
  
  context:
    system_prompt: |
      You are a medical triage assistant. Your role is to:
      - Gather information about patient symptoms
      - Assess urgency level
      - Provide appropriate guidance
      - Always recommend professional consultation for serious symptoms
    
  personas:
    - id: "anxious_patient"
      name: "Anxious Patient"
      traits: ["verbose", "worried", "asks_many_questions"]
      description: "A patient who is very worried and provides lots of detail"
      
    - id: "elderly_patient"
      name: "Elderly Patient"  
      traits: ["brief", "may_forget_details", "polite"]
      description: "An older patient who may need follow-up questions"

  test_scenarios:
    - id: "symptom_assessment"
      category: "core_functionality"
      difficulty: "easy"
      templates:
        - "I've been experiencing {symptom} for {duration}"
        - "My {body_part} has been {condition} since {timeframe}"
      variables:
        symptom: ["headache", "chest pain", "dizziness", "nausea"]
        duration: ["2 days", "a week", "several hours"]
        body_part: ["head", "stomach", "back", "chest"]
        condition: ["hurting", "aching", "bothering me"]
        timeframe: ["yesterday", "this morning", "last week"]
        
  evaluation_criteria:
    - metric: "safety_compliance"
      weight: 0.4
      description: "Did the AI appropriately handle safety-critical situations?"
      
    - metric: "information_gathering"
      weight: 0.3
      description: "Did the AI ask relevant follow-up questions?"
      
    - metric: "appropriate_urgency"
      weight: 0.3
      description: "Did the AI correctly assess the urgency level?"

  # Optional: Define functions for function-calling capability tests
  functions:
    - name: "schedule_appointment"
      description: "Schedule a medical appointment"
      parameters:
        urgency: ["routine", "urgent", "emergency"]
        department: ["general", "cardiology", "neurology"]
        
    - name: "lookup_symptoms"
      description: "Look up symptom information"
      parameters:
        symptom: "string"
```

### 5.4 Benchmark Configuration

```yaml
# benchmark_config.yaml
benchmark:
  name: "GPT-4o vs Claude Sonnet - Healthcare Triage"
  description: "Comparing model performance on medical triage scenarios"
  
  models:
    - provider: openai
      model: gpt-4o
      alias: "GPT-4o"
      temperature: 0.7
      max_tokens: 1000
      
    - provider: anthropic
      model: claude-sonnet-4-20250514
      alias: "Claude Sonnet"
      temperature: 0.7
      max_tokens: 1000
  
  capabilities:
    - chat_completion
    - function_calling
    - structured_output
  
  domain: healthcare_triage
  
  settings:
    runs_per_test: 3           # Run each test N times for consistency
    parallel_execution: true    # Run tests in parallel
    save_raw_responses: true    # Save full API responses
    seed: 42                    # For reproducibility
    
  metrics:
    - accuracy
    - latency_p50
    - latency_p95
    - latency_p99
    - cost_per_1k_tokens
    - tokens_per_response
    - first_token_latency

  output:
    format: ["json", "markdown", "html"]
    directory: "./results"
```

---

## 6. User Interface

### 6.1 Primary: Command Line Interface (CLI)

**Why CLI First?**
- Developer-friendly and familiar
- Easy CI/CD integration
- Scriptable and automatable
- Lower development overhead
- Works in any environment

**CLI Commands:**

```bash
# Installation
pip install domainbench

# Initialize a new domain
domainbench init domain --name "my_custom_domain"

# Generate test cases for a domain
domainbench generate --domain restaurant_waiter --count 100 --output tests.jsonl

# Run a benchmark
domainbench run --config benchmark.yaml

# Run with inline options
domainbench run \
  --models openai/gpt-4o anthropic/claude-sonnet-4-20250514 \
  --domain restaurant_waiter \
  --capabilities chat function_calling \
  --runs 3

# Compare two result files
domainbench compare results1.json results2.json

# Generate a report
domainbench report --input results.json --format html --output report.html

# List available domains
domainbench domains list

# List available capabilities
domainbench capabilities list

# Validate a domain configuration
domainbench validate --domain ./my_domain/

# Interactive mode (guided setup)
domainbench interactive
```

**Interactive Mode Flow:**
```
$ domainbench interactive

Welcome to DomainBench Interactive Setup!

? Select or create a domain:
  > restaurant_waiter (built-in)
    customer_support (built-in)
    healthcare_triage (built-in)
    + Create new domain

? Select capabilities to benchmark:
  [x] Chat Completion
  [x] Function Calling
  [ ] Structured Output
  [ ] Vision/OCR

? Add models to compare:
  Model 1: openai/gpt-4o
  Model 2: anthropic/claude-sonnet-4-20250514
  Add another? (y/N): n

? Number of test runs per case: 3

? Ready to run benchmark? (Y/n): y

Running benchmark...
```

### 6.2 Secondary: Web Dashboard (Phase 2+)

Future enhancement for:
- Real-time benchmark progress visualization
- Interactive comparison charts
- Historical result tracking
- Shareable reports and dashboards
- Team collaboration features

---

## 7. Output & Reporting

### 7.1 Results Structure

```json
{
  "benchmark_id": "550e8400-e29b-41d4-a716-446655440000",
  "benchmark_name": "GPT-4o vs Claude Sonnet - Healthcare Triage",
  "timestamp": "2024-01-15T10:30:00Z",
  "config_hash": "abc123def456",
  "domain": "healthcare_triage",
  "models": [
    {
      "alias": "GPT-4o",
      "provider": "openai",
      "model": "gpt-4o"
    },
    {
      "alias": "Claude Sonnet",
      "provider": "anthropic", 
      "model": "claude-sonnet-4-20250514"
    }
  ],
  "capabilities_tested": ["chat_completion", "function_calling"],
  "results": {
    "chat_completion": {
      "GPT-4o": {
        "accuracy": 0.92,
        "latency_p50_ms": 1200,
        "latency_p95_ms": 2100,
        "avg_tokens": 156,
        "total_cost_usd": 0.045,
        "tests_passed": 92,
        "tests_total": 100
      },
      "Claude Sonnet": {
        "accuracy": 0.89,
        "latency_p50_ms": 900,
        "latency_p95_ms": 1600,
        "avg_tokens": 142,
        "total_cost_usd": 0.038,
        "tests_passed": 89,
        "tests_total": 100
      }
    },
    "function_calling": {
      "GPT-4o": { ... },
      "Claude Sonnet": { ... }
    }
  },
  "comparison_summary": {
    "overall_winner": "GPT-4o",
    "winner_by_accuracy": "GPT-4o",
    "winner_by_latency": "Claude Sonnet",
    "winner_by_cost": "Claude Sonnet",
    "recommendation": "GPT-4o for accuracy-critical applications, Claude Sonnet for cost-sensitive or latency-sensitive use cases"
  },
  "detailed_results": [
    {
      "test_id": "test_001",
      "capability": "chat_completion",
      "scenario": "symptom_assessment",
      "input": "I've been experiencing headaches for 2 days",
      "results": {
        "GPT-4o": {
          "response": "...",
          "latency_ms": 1150,
          "tokens": 145,
          "score": 0.95,
          "evaluation_notes": "..."
        },
        "Claude Sonnet": {
          "response": "...",
          "latency_ms": 870,
          "tokens": 132,
          "score": 0.90,
          "evaluation_notes": "..."
        }
      }
    }
  ]
}
```

### 7.2 Report Formats

| Format | Use Case |
|--------|----------|
| **JSON/JSONL** | Machine-readable, CI/CD integration, further analysis |
| **Markdown** | GitHub-friendly, documentation, quick sharing |
| **HTML** | Visual dashboard, standalone reports, presentations |
| **CSV** | Spreadsheet analysis, data export |

---

## 8. Evaluation Methodology

### 8.1 Evaluation Approaches

| Approach | Description | Use Case |
|----------|-------------|----------|
| **Rule-based** | Regex, exact match, keyword presence | Structured outputs, specific requirements |
| **LLM-as-Judge** | Another model scores responses | Complex quality assessment |
| **Semantic Similarity** | Embedding-based comparison | Open-ended responses |
| **Human-in-the-loop** | Manual scoring for ground truth | Calibration, edge cases |
| **Composite** | Combination of above | Production benchmarks |

### 8.2 Default Metrics

**Performance Metrics:**
- Latency (p50, p95, p99, avg)
- First token latency (streaming)
- Tokens per second
- Total response tokens

**Quality Metrics:**
- Accuracy (task completion)
- Relevance score
- Safety compliance
- Instruction following

**Cost Metrics:**
- Input tokens
- Output tokens
- Cost per request
- Cost per 1K tokens

**Capability-Specific:**
- Function call accuracy (correct function + parameters)
- JSON schema compliance (structured output)
- OCR accuracy (character error rate)

---

## 9. Implementation Phases

### Phase 1: MVP (Weeks 1-3)
**Goal**: Working CLI with core functionality

- [ ] Project structure setup
- [ ] Core engine implementation
- [ ] Provider adapters (OpenAI, Anthropic)
- [ ] Chat completion capability
- [ ] Basic metrics (accuracy, latency, cost)
- [ ] CLI commands: run, compare, report
- [ ] JSON/Markdown output
- [ ] Migrate existing waiter benchmark
- [ ] Basic documentation

### Phase 2: Capability Expansion (Weeks 4-6)
**Goal**: Multiple benchmark types

- [ ] Function calling capability
- [ ] Structured output capability
- [ ] Tool use capability
- [ ] Enhanced evaluation (LLM-as-judge)
- [ ] Test case generator
- [ ] Interactive CLI mode
- [ ] Additional providers (Ollama, Azure)

### Phase 3: Advanced Features (Weeks 7-9)
**Goal**: Production-ready framework

- [ ] Vision/OCR capability
- [ ] Code execution capability
- [ ] Multi-turn conversation benchmarks
- [ ] HTML report generation
- [ ] Custom metric plugins
- [ ] Performance optimization
- [ ] Comprehensive test suite

### Phase 4: Community & Polish (Weeks 10+)
**Goal**: Community adoption

- [ ] Web dashboard (optional)
- [ ] Domain template marketplace/sharing
- [ ] Public leaderboards
- [ ] Plugin system documentation
- [ ] Video tutorials
- [ ] PyPI package publication

---

## 10. Open Questions & Decisions

### To Be Decided:

1. **Evaluation Strategy Priority**
   - [ ] Rule-based only for MVP?
   - [ ] Include LLM-as-judge from start?
   - [ ] Which judge model to use?

2. **Test Case Generation**
   - [ ] Template-based only for MVP?
   - [ ] Allow LLM-generated synthetic cases?
   - [ ] Support importing real-world datasets?

3. **Scoring Philosophy**
   - [ ] Single aggregate score vs multi-dimensional?
   - [ ] How to weight different metrics?
   - [ ] Domain-specific vs universal scoring?

4. **Deployment Target**
   - [ ] Pure Python package (pip install)?
   - [ ] Docker support needed?
   - [ ] Cloud-hosted option later?

5. **Provider Priority**
   - [ ] Which providers for MVP?
   - [ ] Local model support priority?

---

## 11. Technical Stack

**Core:**
- Python 3.10+
- Pydantic (configuration & validation)
- Rich (CLI formatting)
- Click or Typer (CLI framework)

**Providers:**
- openai (OpenAI API)
- anthropic (Anthropic API)
- httpx (HTTP client for custom APIs)

**Evaluation:**
- numpy (statistics)
- scikit-learn (optional, for advanced metrics)

**Reporting:**
- Jinja2 (HTML templates)
- matplotlib/plotly (charts)

**Testing:**
- pytest
- pytest-asyncio

---

## 12. Success Criteria

### MVP Success:
- [ ] Can benchmark 2 models on chat completion
- [ ] Produces clear comparison results
- [ ] Works with custom domain configuration
- [ ] Installation via pip works
- [ ] Documentation covers basic usage

### Full Release Success:
- [ ] 5+ capabilities supported
- [ ] 3+ providers supported  
- [ ] 10+ built-in domain templates
- [ ] Active community contributions
- [ ] Used in production by external users

---

## Appendix A: Example Domain - Restaurant Waiter (Current)

This is the migration target for the existing implementation:

```yaml
domain:
  name: "Restaurant Waiter Assistant"
  description: "AI assistant acting as a restaurant waiter"
  
  context:
    system_prompt: |
      You are a helpful and friendly restaurant waiter...
      
  test_scenarios:
    - category: "order_taking"
    - category: "menu_questions"  
    - category: "special_requests"
    - category: "complaint_handling"
```

---

## Appendix B: References

- [HELM Benchmark](https://crfm.stanford.edu/helm/)
- [LMSys Chatbot Arena](https://chat.lmsys.org/)
- [OpenAI Evals](https://github.com/openai/evals)
- [Anthropic Model Card](https://docs.anthropic.com/)

---

*Last Updated: December 30, 2024*
*Version: 0.1-draft*
