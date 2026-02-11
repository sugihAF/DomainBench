# Voice Agent Benchmark

Multi-turn voice agent evaluation with **LLM-as-Judge** and **two-phase realignment scoring**, based on the [aiewf-eval](https://github.com/kwindla/aiewf-eval) methodology.

## Overview

The voice benchmark evaluates LLMs acting as voice agents in realistic, multi-turn conversations. Each scenario includes a system prompt, knowledge base, tool definitions, and 15-30 conversation turns with golden (expected) responses.

```
Scenario (30 turns)
    ↓
Sequential Execution (turn-by-turn with tool interception)
    ↓
LLM-as-Judge (Claude/GPT-4o scores each turn)
    ↓
Two-Phase Realignment (adjusts for shifted tool calls)
    ↓
Pass Rate (% of turns passing all dimensions)
```

### What It Tests

| Dimension | What It Measures |
|---|---|
| **Tool Use Correctness** | Did the model call the right function with semantically equivalent arguments? |
| **Instruction Following** | Did the model answer the question, advance the task, or properly deflect? |
| **KB Grounding** | Is the response factually consistent with the knowledge base? |
| **Turn-Taking** | Audio timing analysis — speech overlap, missing audio, latency (audio pipelines only) |

All scores are **binary (pass/fail)** per turn. The overall pass rate is the percentage of passing dimension-turn pairs.

## Quick Start

### 1. Generate a test scenario

```bash
# Use built-in hotel concierge template
domainbench voice generate --builtin hotel_concierge -o scenario.jsonl

# Or generate with AI for a custom domain
domainbench voice generate \
  -n "Tech Support" \
  -d "IT helpdesk agent that handles password resets, VPN issues, and software installation" \
  --turns 25 \
  -o tech_support.jsonl
```

### 2. Run the benchmark

```bash
# Single model evaluation
domainbench voice run -d scenario.jsonl -m openai/gpt-4o

# Compare two models
domainbench voice run -d scenario.jsonl -m openai/gpt-4o -m anthropic/claude-sonnet-4

# Multiple runs for consistency
domainbench voice run -d scenario.jsonl -m openai/gpt-4o --runs 5
```

### 3. View results

Results are saved to `./results/voice_*.json` with per-turn breakdowns, dimension scores, and latency statistics.

## Commands

### `domainbench voice run`

Run the voice agent benchmark.

| Option | Description | Default |
|---|---|---|
| `-d, --dataset` | Path to JSONL dataset | **Required** |
| `-m, --model` | Model(s) to evaluate (repeatable) | **Required** |
| `-p, --pipeline` | Pipeline config YAML (for cascaded/S2S) | None (text mode) |
| `-j, --judge` | Judge model for scoring | `openai/gpt-4o` |
| `--runs` | Number of repeated runs per scenario | `1` |
| `-o, --output` | Output directory | `./results` |
| `--max-scenarios` | Limit scenarios from dataset | All |
| `-v, --verbose` | Print detailed progress | Off |

### `domainbench voice generate`

Generate evaluation scenarios.

| Option | Description | Default |
|---|---|---|
| `--builtin, -b` | Use built-in template (e.g., `hotel_concierge`) | None |
| `-n, --name` | Domain name (for AI generation) | None |
| `-d, --description` | Domain description (for AI generation) | None |
| `--turns, -t` | Target conversation turns | `20` |
| `--scenarios` | Number of scenarios to generate | `1` |
| `-m, --model` | Model for AI generation | `openai/gpt-4o` |
| `-o, --output` | Output JSONL path | `./voice_dataset.jsonl` |
| `--seed` | Random seed for built-in generators | `42` |

### `domainbench voice domains`

List available built-in voice domains.

## Pipeline Types

### Text Mode (Default)

Tests the LLM only — no audio processing. Three dimensions scored (turn-taking auto-passes).

```bash
domainbench voice run -d scenario.jsonl -m openai/gpt-4o
```

### Cascaded Pipeline

Tests the full STT → LLM → TTS chain. Configure via YAML:

```yaml
# cascaded_config.yaml
type: cascaded

stt:
  provider: deepgram
  model: nova-2
  api_key_env: DEEPGRAM_API_KEY

llm:
  provider: openai
  model: gpt-4o-mini
  api_key_env: OPENAI_API_KEY

tts:
  provider: elevenlabs
  model: eleven_flash_v2_5
  api_key_env: ELEVENLABS_API_KEY
  voice_id: FGY2WhTYpPnrIDTdsKH5
  params:
    stability: 0.7
    similarity_boost: 0.8
    speed: 1.0
```

```bash
domainbench voice run -d scenario.jsonl -p cascaded_config.yaml
```

### Cascaded Pipeline — How It Works

```
Text Input → [TTS] → Audio → [STT] → Transcribed Text → [LLM] → Response Text → [TTS] → Audio
                                ↑                            ↑                       ↑
                          STT latency                  LLM latency              TTS latency
```

The benchmark synthesizes user audio from text (using the configured TTS or a default OpenAI TTS), sends it through STT to test transcription accuracy, then feeds the transcribed text to the LLM. This reveals how STT errors degrade overall pipeline quality. If scenario turns have `audio_file` paths, pre-recorded audio is used instead.

**Supported STT providers**: `whisper` (OpenAI), `deepgram`, `google`
**Supported TTS providers**: `openai`, `elevenlabs`, `google`, `cartesia`

### Speech-to-Speech Pipeline

Tests end-to-end audio models that accept audio in and produce audio + text out:

```yaml
# s2s_config.yaml
type: speech_to_speech

model:
  provider: openai
  model: gpt-4o-audio-preview
  api_key_env: OPENAI_API_KEY
  voice: alloy
  audio_format: wav

# Optional: TTS for synthesizing user audio from text
input_tts:
  provider: openai
  model: tts-1
  voice: alloy
```

```bash
domainbench voice run -d scenario.jsonl -p s2s_config.yaml
```

**Supported S2S models**: OpenAI (`gpt-4o-audio-preview`), Gemini (`gemini-2.0-flash` with native audio)

For S2S mode, `-m` is optional — the model comes from the pipeline YAML.

### Comparing Pipelines

Run the same scenario with all three pipeline types to compare:

```bash
# Text only (LLM ceiling)
domainbench voice run -d scenario.jsonl -m openai/gpt-4o

# Cascaded (STT -> LLM -> TTS)
domainbench voice run -d scenario.jsonl -p pipeline_cascaded.yaml

# Speech-to-speech (end-to-end)
domainbench voice run -d scenario.jsonl -p pipeline_s2s.yaml
```

All three produce the same 4-dimension scores, revealing:
- Does cascaded lose accuracy from STT errors?
- Is speech-to-speech faster but less accurate on tool calls?
- Is text-only the theoretical ceiling for content quality?

## Dataset Format

Each JSONL line is one complete scenario:

```json
{
  "id": "voice_hotel_concierge_001",
  "domain": "hotel_concierge",
  "system_prompt": "You are a friendly hotel concierge...",
  "knowledge_base": "Hotel Grand Horizon - Check-in: 3PM...",
  "tools": [
    {
      "name": "book_restaurant",
      "description": "Book a restaurant reservation",
      "parameters": {
        "type": "object",
        "properties": {
          "guest_name": {"type": "string", "description": "Guest name"},
          "restaurant": {"type": "string", "description": "Restaurant name"},
          "time": {"type": "string", "description": "Time in HH:MM"}
        },
        "required": ["guest_name", "restaurant", "time"]
      }
    }
  ],
  "turns": [
    {
      "input": "Hi, what restaurants are available?",
      "golden_text": "We have three dining options...",
      "required_function_call": null,
      "function_call_response": null
    },
    {
      "input": "Book a table at the Italian place for 7pm.",
      "golden_text": "I'll book that for you right away.",
      "required_function_call": {
        "name": "book_restaurant",
        "args": {"guest_name": "Sarah", "restaurant": "La Trattoria", "time": "19:00"}
      },
      "function_call_response": {"status": "confirmed", "id": "RES-123"}
    }
  ]
}
```

## Scoring Algorithm

### Two-Phase Realignment

The key innovation preventing cascading false negatives:

**Phase 1**: Score each turn independently against the golden reference.

**Phase 2**: Detect function-call timing shifts and adjust:

| Scenario | Expected Turn | Actual Turn | Expected Turn Score | Actual Turn Score |
|---|---|---|---|---|
| On time | 5 | 5 | PASS | PASS |
| Early call | 5 | 3 | PASS (not penalized) | PASS (credit) |
| Late call | 5 | 7 | FAIL | PASS (credit) |
| Never called | 5 | — | FAIL | — |

### Pass Rate Formula

```
Pass Rate = sum(tool_correct + instruction_following + kb_grounding) / (turns x 3) x 100%
```

For audio pipelines, turn-taking is included as a 4th dimension (denominator becomes turns x 4).

### Multi-Run Aggregation

When using `--runs N`, DomainBench reports:
- **Mean pass rate**: Average across all runs
- **Median pass rate**: Typical performance (robust to outliers)
- **Min/Max**: Performance range

## Evaluation Details

### Tool Use Correctness

- Arguments compared for **semantic equivalence** (not verbatim)
- `"a session about open telemetry"` ≈ `"Open Telemetry session"` → PASS
- IDs and identifiers must match **exactly**
- Extra tool calls (not expected) are noted but not penalized
- Missing expected calls are penalized unless caught by realignment

### Instruction Following

- Directly answering the question → PASS
- Properly deflecting out-of-scope → PASS
- Contradicting own actions → FAIL
- Lenient scoring when turn-taking fails (audio garbling)

### Knowledge Base Grounding

- Only fails on **explicit factual errors**
- Additional correct information → PASS
- Vague/general answers without errors → PASS
- Wrong dates, times, locations, names → FAIL

## Results Format

```json
{
  "benchmark_type": "voice",
  "timestamp": "2025-06-15T14:30:00",
  "config": {
    "models": ["openai/gpt-4o"],
    "judge": "openai/gpt-4o",
    "num_runs": 3,
    "pipeline_type": "text"
  },
  "results": {
    "openai/gpt-4o": {
      "aggregated": [{
        "model_name": "openai/gpt-4o",
        "scenario_id": "voice_hotel_concierge_001",
        "num_runs": 3,
        "pass_rate_mean": 93.3,
        "pass_rate_median": 95.0,
        "dimension_scores": {
          "tool_use_correct": 90.0,
          "instruction_following": 95.0,
          "kb_grounding": 100.0,
          "turn_taking": 100.0
        },
        "latency": {
          "ttfb_median_ms": 850
        }
      }],
      "runs": [...]
    }
  }
}
```

## Built-in Domains

| Domain | Turns | Tool Calls | Description |
|---|---|---|---|
| `hotel_concierge` | 20 | 7 | Hotel front desk — restaurant bookings, spa, transportation, room service, local info |

Create custom domains with AI:
```bash
domainbench voice generate -n "Airline Support" \
  -d "Airline customer service handling flight changes, seat upgrades, baggage issues, and loyalty programs" \
  --turns 25 -o airline.jsonl
```

## Best Practices

1. **Multiple runs**: Use `--runs 3-5` for reliable results; LLM responses are non-deterministic.
2. **Judge model**: Use a strong model (GPT-4o, Claude Sonnet 4) for consistent judging.
3. **Scenario length**: 15-30 turns is ideal. Shorter may not exercise all dimensions; longer increases cost.
4. **Tool calls**: Aim for 5-8 tool calls per scenario to adequately test tool use.
5. **Knowledge base**: Include specific, verifiable facts (dates, prices, names) for grounding checks.
6. **Deflection turns**: Include 1-2 out-of-scope questions to test instruction adherence.

## References

- **aiewf-eval**: [github.com/kwindla/aiewf-eval](https://github.com/kwindla/aiewf-eval) — Original benchmark methodology
- **Daily Blog**: [Benchmarking LLMs for Voice Agent Use Cases](https://www.daily.co/blog/benchmarking-llms-for-voice-agent-use-cases/)
- **MT-Bench**: Multi-turn evaluation methodology (adapted for voice)
- **BFCL**: Berkeley Function-Calling Leaderboard (tool use evaluation patterns)
