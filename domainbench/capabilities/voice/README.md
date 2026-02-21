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
| `--save-audio` | Save intermediate audio files to disk | Off |
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

## Latency Metrics

DomainBench tracks latency at multiple granularities, inspired by the [aiewf-eval](https://github.com/kwindla/aiewf-eval) methodology. These metrics are available for **cascaded** and **speech-to-speech** pipelines (text mode only reports LLM latency).

### Per-Stage Latency

For cascaded pipelines, each turn records individual stage timings:

| Metric | Description |
|---|---|
| `stt_latency_ms` | Time for STT to transcribe user audio |
| `llm_latency_ms` | Time for the LLM to generate a response |
| `tts_latency_ms` | Time for TTS to synthesize response audio |
| `input_synthesis_latency_ms` | Time to synthesize user input audio (not counted in pipeline latency) |

For speech-to-speech pipelines, only `model_latency` and `input_synthesis_latency_ms` are recorded (the model handles everything end-to-end).

### Aggregate Latency Statistics

Across all turns in a run, DomainBench computes median, mean, p95, and max for:

| Metric | Description |
|---|---|
| `ttfb_*_ms` | Time to first byte — how quickly the pipeline starts responding |
| `latency_*_ms` | Total pipeline latency (STT + LLM + TTS, or model latency for S2S) |

### Voice-to-Voice (V2V) Latency

V2V measures the complete time from when the user finishes speaking to when the model's response audio actually begins — including any leading silence in the output audio. This is the metric users *feel* in a real conversation.

```
V2V = Pipeline Processing Time + Silence Padding
```

| Pipeline | V2V Formula |
|---|---|
| **Cascaded** | `stt_latency + llm_latency + tts_latency + silence_pad` |
| **Speech-to-Speech** | `model_latency + silence_pad` |
| **Text** | Not applicable (no audio) |

Reported statistics:

| Metric | Description |
|---|---|
| `v2v_median_ms` | Median V2V across all turns |
| `v2v_mean_ms` | Mean V2V across all turns |
| `v2v_p95_ms` | 95th percentile V2V |
| `v2v_max_ms` | Maximum V2V observed |

### Silence Padding

Silence padding is the duration of leading silence in the model's response audio — the quiet gap before speech actually begins. TTS engines and S2S models often prepend a short silent buffer (typically 40-120ms) to their output. This "dead air" adds to perceived latency even though the response bytes have already arrived.

DomainBench detects silence padding by analyzing the WAV response audio:
1. Audio is divided into 10ms windows
2. RMS energy is computed for each window
3. The first window exceeding a normalized threshold (default 0.02) marks the start of speech
4. Everything before that point is silence padding

| Metric | Description |
|---|---|
| `silence_pad_median_ms` | Median silence padding across turns |
| `silence_pad_mean_ms` | Mean silence padding across turns |
| `silence_pad_p95_ms` | 95th percentile silence padding |
| `silence_pad_max_ms` | Maximum silence padding observed |

Silence padding is only measured on WAV audio. Non-WAV formats (mp3, ogg) return `null`.

### Tool vs Non-Tool Latency Separation

Turns that trigger tool calls typically have higher latency than simple conversational turns (the model must generate structured function arguments, wait for the tool response, then generate a follow-up). DomainBench separates latency statistics by whether a turn involved tool calls, matching the aiewf-eval approach:

| Metric | Description |
|---|---|
| `non_tool_v2v_median_ms` | Median V2V for turns **without** tool calls |
| `non_tool_v2v_max_ms` | Max V2V for non-tool turns |
| `tool_v2v_mean_ms` | Mean V2V for turns **with** tool calls |
| `tool_v2v_max_ms` | Max V2V for tool turns |
| `non_tool_latency_mean_ms` | Mean total latency for non-tool turns |
| `tool_latency_mean_ms` | Mean total latency for tool turns |

This separation helps identify whether latency issues are specific to tool-calling turns or affect the entire pipeline.

### Comparison with aiewf-eval

| aiewf-eval Metric | DomainBench Equivalent |
|---|---|
| Non-Tool V2V Med | `non_tool_v2v_median_ms` |
| Non-Tool V2V Max | `non_tool_v2v_max_ms` |
| Tool V2V Mean | `tool_v2v_mean_ms` |
| Silence Pad Mean | `silence_pad_mean_ms` |

DomainBench additionally provides per-stage breakdown (STT/LLM/TTS latency), p95 percentiles, and per-turn granularity that aiewf-eval does not track.

### CLI Output

The results table includes V2V and Silence Pad columns for audio pipelines:

```
┌──────────────────┬───────────┬──────────┬─────────────┬───────────┬───────────┬──────────┬─────────┬──────┐
│ Model            │ Pass Rate │ Tool Use │ Instruction │ KB Ground │ TTFB (med)│ V2V (med)│ Sil.Pad │ Runs │
├──────────────────┼───────────┼──────────┼─────────────┼───────────┼───────────┼──────────┼─────────┼──────┤
│ openai/gpt-4o    │ 88.5%     │ 76.0%    │ 80.0%       │ 98.0%     │ 3920ms    │ 4970ms   │ 50ms    │ 25   │
└──────────────────┴───────────┴──────────┴─────────────┴───────────┴───────────┴──────────┴─────────┴──────┘
```

### Viewer Dashboard

The web viewer (`domainbench viewer`) displays:
- **Model card**: V2V mean, Silence Pad mean, Non-Tool V2V, Tool V2V
- **Per-turn detail**: V2V and Silence Pad for each individual turn
- **Pipeline Latency chart**: Stacked bar chart with per-stage breakdown and component names

## Audio Persistence

Use `--save-audio` to save intermediate audio files for debugging and auditing:

```bash
domainbench voice run -d scenario.jsonl -p cascaded_config.yaml --save-audio
```

Audio files are organized under `results/audio/`:

```
results/
├── voice_gpt-4o-mini_20260211_161703.json
└── audio/
    └── voice_gpt-4o-mini_20260211_161703/
        └── voice_hotel_concierge_001/
            ├── run_0/
            │   ├── turn_0_input_tts.wav
            │   ├── turn_0_response_tts.wav
            │   ├── turn_1_input_tts.wav
            │   └── turn_1_response_tts.wav
            └── run_1/
                └── ...
```

| Audio File | Description |
|---|---|
| `turn_N_input_tts.wav` | Synthesized user input audio (sent to STT or S2S model) |
| `turn_N_response_tts.wav` | Synthesized assistant response audio (cascaded pipeline) |
| `turn_N_s2s_output.wav` | Raw S2S model output audio (speech-to-speech pipeline) |

Saved audio is playable directly in the web viewer via inline audio players in the turn detail cards.

## Results Format

```json
{
  "benchmark_type": "voice",
  "timestamp": "2025-06-15T14:30:00",
  "config": {
    "models": ["openai/gpt-4o"],
    "judge": "openai/gpt-4o",
    "num_runs": 3,
    "pipeline_type": "cascaded",
    "audio_dir": "results/audio/voice_gpt-4o_20250615_143000",
    "pipeline_components": {
      "stt": "deepgram/nova-2",
      "llm": "openai/gpt-4o",
      "tts": "elevenlabs/eleven_flash_v2_5"
    }
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
          "ttfb_median_ms": 850,
          "latency_median_ms": 1200,
          "v2v_median_ms": 1250,
          "silence_pad_mean_ms": 52.0,
          "non_tool_v2v_median_ms": 1100,
          "non_tool_v2v_max_ms": 1800,
          "tool_v2v_mean_ms": 1600
        }
      }],
      "runs": [...]
    }
  }
}
```

Each turn result in `runs[].turn_results[]` includes per-turn values:

```json
{
  "turn_index": 0,
  "user_input": "Hi, what restaurants are available?",
  "assistant_text": "We have three dining options...",
  "golden_text": "We have three dining options...",
  "tool_calls": [],
  "stt_latency_ms": 320.5,
  "llm_latency_ms": 850.2,
  "tts_latency_ms": 280.1,
  "latency_ms": 1450.8,
  "ttfb_ms": 1170.7,
  "silence_pad_ms": 48.0,
  "v2v_ms": 1498.8,
  "audio_files": {
    "input_tts": "voice_hotel_concierge_001/run_0/turn_0_input_tts.wav",
    "stt_input": "voice_hotel_concierge_001/run_0/turn_0_input_tts.wav",
    "response_tts": "voice_hotel_concierge_001/run_0/turn_0_response_tts.wav"
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
