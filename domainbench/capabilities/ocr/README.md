# OCR/Vision Extraction Benchmark

Document and image extraction benchmarks with schema-aware fuzzy matching and structural validation.

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Commands](#commands)
- [Usage Examples](#usage-examples)
- [Dataset Format](#dataset-format)
- [Evaluation](#evaluation)
- [Results](#results)
- [Supported Formats](#supported-formats)

## Overview

The OCR/Vision capability benchmarks vision models on structured data extraction tasks such as:
- Menu extraction from restaurant images
- Receipt parsing and itemization
- Document OCR with structured fields
- Custom structured data extraction

### Key Features

- **Single model evaluation**: Test one model against ground truth
- **Head-to-head comparison**: Compare two models side-by-side
- **PDF support**: Automatic PDF to image conversion (per-page)
- **Schema-aware scoring**: Validates both structure and content
- **Fuzzy matching**: Configurable text similarity thresholds
- **Multiple formats**: Support for PNG, JPG, PDF, and more

### Evaluation Method

Unlike chat benchmarks that use LLM-as-Judge, OCR benchmarks use **deterministic fuzzy matching** against ground truth data:

- **Structure Score** (35%): Schema validity, required fields, type correctness
- **Content Score** (65%): Value similarity with fuzzy text matching
- **Overall Score**: Weighted combination of structure and content

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                   Input Image/PDF                            │
│                                                              │
│    [Menu Image]  or  [Receipt PDF]  or  [Document]          │
└─────────────────────────────────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼                               ▼
    ┌─────────────┐                 ┌─────────────┐
    │   Model A   │                 │   Model B   │
    │   (GPT-4o)  │                 │  (Gemini)   │
    └─────────────┘                 └─────────────┘
           │                               │
           ▼                               ▼
    ┌─────────────┐                 ┌─────────────┐
    │  Parse JSON │                 │  Parse JSON │
    └─────────────┘                 └─────────────┘
           │                               │
           └───────────────┬───────────────┘
                           ▼
                  ┌─────────────────┐
                  │ Fuzzy Matcher   │
                  │ vs Ground Truth │
                  └─────────────────┘
                           │
                           ▼
            Structure Score + Content Score
                    = Overall Score
```

### Evaluation Process

1. **Load test case**: Image/PDF + ground truth + optional schema
2. **Model inference**: Vision model extracts structured data
3. **Parse response**: Extract JSON from model output
4. **Structure validation**: Check schema compliance
   - Required fields present
   - Correct data types
   - Schema constraints met
5. **Content matching**: Compare values with ground truth
   - Fuzzy string matching
   - Numeric tolerance
   - Identity-based list matching
6. **Calculate scores**: Combine structure + content scores

## Commands

### `domainbench ocr run`

Run an OCR/Vision extraction benchmark.

**Syntax:**
```bash
domainbench ocr run [OPTIONS]
```

**Options:**

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--dataset` | `-d` | PATH | ✅ | - | Path to dataset JSONL file, or single image/PDF |
| `--models` | `-m` | TEXT | ✅ | - | 1 model (single eval) or 2 models (comparison) |
| `--ground-truth` | `-gt` | PATH | - | - | Ground truth JSON (required for single image/PDF) |
| `--schema-output` | `-so` | PATH | - | - | JSON schema file for expected output format |
| `--schema` | `-s` | TEXT | - | `menu` | Schema type: menu, receipt, document |
| `--output` | `-o` | PATH | - | `./results` | Output directory for results |
| `--max-items` | - | INTEGER | - | - | Maximum number of test cases to run |
| `--threshold` | `-t` | FLOAT | - | `0.7` | Fuzzy match threshold (0.0-1.0) |
| `--pdf-dpi` | - | INTEGER | - | `150` | DPI for PDF to image conversion |
| `--pdf-max-pages` | - | INTEGER | - | - | Max pages to process per PDF (all if not set) |
| `--verbose` | `-v` | FLAG | - | `True` | Show detailed progress |
| `--quiet` | `-q` | FLAG | - | `False` | Suppress progress output |

**Examples:**

```bash
# Single model evaluation with JSONL dataset
domainbench ocr run \
  -d menu_dataset.jsonl \
  -m openai/gpt-4o

# Single image with ground truth
domainbench ocr run \
  -d menu.png \
  -gt truth.json \
  -m openai/gpt-4o

# PDF with output schema
domainbench ocr run \
  -d document.pdf \
  -gt truth.json \
  -so schema.json \
  -m gemini/gemini-2.5-flash

# Head-to-head comparison
domainbench ocr run \
  -d receipt_dataset.jsonl \
  -m openai/gpt-4o \
  -m gemini/gemini-2.5-flash \
  --schema receipt

# Custom PDF processing
domainbench ocr run \
  -d large_document.pdf \
  -gt truth.json \
  -m openai/gpt-4o \
  --pdf-dpi 200 \
  --pdf-max-pages 10

# Quick testing with limited items
domainbench ocr run \
  -d dataset.jsonl \
  -m openai/gpt-4o \
  -m gemini/gemini-2.0-flash \
  --max-items 5
```

## Usage Examples

### Single Model Evaluation

Test one model's extraction accuracy against ground truth:

```bash
# Evaluate GPT-4o on menu extraction
domainbench ocr run \
  -d menus.jsonl \
  -m openai/gpt-4o \
  --schema menu
```

### Head-to-Head Comparison

Compare two models on the same dataset:

```bash
# Compare GPT-4o vs Gemini-2.5-Flash
domainbench ocr run \
  -d receipts.jsonl \
  -m openai/gpt-4o \
  -m gemini/gemini-2.5-flash \
  --schema receipt
```

### Single Image Quick Test

Test extraction on a single image:

```bash
# Process single menu image
domainbench ocr run \
  -d restaurant_menu.png \
  -gt menu_truth.json \
  -so menu_schema.json \
  -m openai/gpt-4o
```

### PDF Document Processing

Process multi-page PDFs:

```bash
# Extract from PDF document
domainbench ocr run \
  -d invoice.pdf \
  -gt invoice_truth.json \
  -m gemini/gemini-2.5-flash \
  --pdf-dpi 200 \
  --pdf-max-pages 5
```

### Custom Schema with High Threshold

Use custom extraction schema with strict matching:

```bash
domainbench ocr run \
  -d custom_docs.jsonl \
  -m openai/gpt-4o \
  -so custom_schema.json \
  --threshold 0.9
```

## Dataset Format

### JSONL Dataset

One test case per line in JSONL format:

```jsonl
{"id": "menu_001", "image_path": "images/menu1.png", "ground_truth": {"items": [...]}}
{"id": "menu_002", "pdf_path": "pdfs/menu2.pdf", "ground_truth": {"items": [...]}}
{"id": "menu_003", "image_paths": ["page1.png", "page2.png"], "ground_truth": {...}}
```

### Test Case Structure

**For images:**
```json
{
  "id": "unique_id",
  "image_path": "path/to/image.png",
  "ground_truth": {
    "items": [...],
    "metadata": {...}
  },
  "output_schema": {
    "type": "object",
    "properties": {...}
  }
}
```

**For PDFs:**
```json
{
  "id": "unique_id",
  "pdf_path": "path/to/document.pdf",
  "ground_truth": {...},
  "pdf_dpi": 200,
  "pdf_max_pages": 5
}
```

**For multiple images:**
```json
{
  "id": "unique_id",
  "image_paths": ["page1.png", "page2.png"],
  "ground_truth": {...}
}
```

### Ground Truth Format

Ground truth should match the expected extraction structure:

**Menu extraction:**
```json
{
  "items": [
    {
      "id": "item_1",
      "name": "Margherita Pizza",
      "price": 12.99,
      "description": "Fresh mozzarella and basil"
    }
  ],
  "categories": [
    {
      "name": "Pizza",
      "items": ["item_1"]
    }
  ]
}
```

**Receipt extraction:**
```json
{
  "items": [
    {
      "name": "Coffee",
      "quantity": 2,
      "price": 4.50
    }
  ],
  "totals": {
    "subtotal": 9.00,
    "tax": 0.90,
    "total": 9.90
  },
  "metadata": {
    "date": "2025-01-12",
    "merchant": "Coffee Shop"
  }
}
```

### Output Schema (Optional)

Define expected JSON structure to guide model extraction:

```json
{
  "type": "object",
  "properties": {
    "items": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "price": {"type": "number"}
        },
        "required": ["name", "price"]
      }
    }
  },
  "required": ["items"]
}
```

The schema is included in the prompt to help models produce correctly structured output.

## Evaluation

### Scoring System

OCR evaluation uses a **two-part scoring system**:

#### 1. Structure Score (35% default weight)

Validates schema compliance:
- ✅ Required fields present
- ✅ Correct data types
- ✅ Schema constraints met (minLength, minimum, etc.)
- ❌ Extra keys (if additionalProperties: false)

**Example:**
```
Schema requires: {"items": [...], "total": number}
Model output:     {"items": [...], "total": 42.50}
Structure score:  100% (all requirements met)
```

#### 2. Content Score (65% default weight)

Compares values with ground truth:

**String matching:**
- Fuzzy string similarity (SequenceMatcher)
- Normalized (lowercase, trim whitespace)
- Configurable threshold (default: 0.7)

**Number matching:**
- Absolute tolerance: ±0.01 (default)
- Relative tolerance: 0.0 (default)

**List matching:**
- Identity-based matching (by id, uuid, name)
- Unordered greedy matching
- Coverage-based scoring

**Example:**
```
Ground truth: "Margherita Pizza" → $12.99
Model output: "margherita pizza" → $12.99

String similarity: 100% (normalized match)
Number match:      100% (exact)
Content score:     100%
```

### Schema Types

Three built-in schema types are available:

#### Menu Schema
```python
{
  "items": {
    "name_field": "name",
    "weight": 0.5,
    "additional_fields": ["price", "description"]
  },
  "categories": {
    "name_field": "name",
    "weight": 0.2
  }
}
```

#### Receipt Schema
```python
{
  "items": {
    "name_field": "name",
    "weight": 0.5,
    "additional_fields": ["price", "quantity"]
  },
  "totals": {
    "name_field": "label",
    "weight": 0.3,
    "additional_fields": ["amount"]
  }
}
```

#### Document Schema
```python
{
  "fields": {
    "name_field": "field_name",
    "weight": 0.6,
    "additional_fields": ["value"]
  }
}
```

### Configurable Parameters

Control evaluation behavior:

```python
SchemaScoreConfig(
    structure_weight=0.35,        # Weight for structure score
    content_weight=0.65,          # Weight for content score
    default_list_mode="unordered", # List matching: ordered/unordered
    identity_keys=("id", "uuid", "name"),  # Keys for list matching
    abs_tol=0.01,                 # Numeric absolute tolerance
    rel_tol=0.0,                  # Numeric relative tolerance
    normalize_strings=True,        # Normalize before comparison
    enforce_additional_properties=True,  # Penalize extra keys
    unordered_min_match=0.30,     # Min similarity for list items
    path_weights=None             # Custom weights for JSON paths
)
```

### Fuzzy Matching Examples

**High similarity (>0.9):**
```
"Margherita Pizza" vs "margherita pizza"  → 1.0 (exact after normalization)
"Caesar Salad"     vs "Ceasar Salad"      → 0.92 (minor typo)
```

**Medium similarity (0.7-0.9):**
```
"Pepperoni Pizza" vs "Peperoni Pizza"     → 0.87 (spelling variant)
"French Fries"    vs "French Fried"       → 0.78 (partial match)
```

**Low similarity (<0.7):**
```
"Margherita Pizza" vs "Cheese Pizza"      → 0.45 (different items)
"Coffee"           vs "Cappuccino"        → 0.35 (related but different)
```

## Results

### Single Model Result

```json
{
  "benchmark_type": "ocr",
  "timestamp": "2025-01-12T10:30:00",
  "config": {
    "models": ["openai/gpt-4o"],
    "schema_type": "menu",
    "threshold": 0.7,
    "dataset": "menus.jsonl",
    "test_count": 50
  },
  "summary": {
    "model_metrics": {
      "openai/gpt-4o": {
        "avg_score": 87.5,
        "min_score": 65.2,
        "max_score": 98.3,
        "avg_time_ms": 1200
      }
    }
  },
  "results": [
    {
      "test_id": "menu_001",
      "score": 87.5,
      "metrics": {
        "scores": {
          "structure_0_to_100": 95.0,
          "content_0_to_100": 84.3,
          "overall_0_to_100": 87.5
        },
        "structure_diagnostics": {
          "missing_required": 0,
          "type_errors": 0,
          "extra_keys": 2
        }
      },
      "parsed_extraction": {...},
      "ground_truth": {...}
    }
  ]
}
```

### Comparison Result

```json
{
  "benchmark_type": "ocr",
  "timestamp": "2025-01-12T10:30:00",
  "config": {
    "models": ["openai/gpt-4o", "gemini/gemini-2.5-flash"],
    "schema_type": "menu",
    "threshold": 0.7,
    "test_count": 50
  },
  "summary": {
    "overall_winner": "openai/gpt-4o",
    "comparison": {
      "A_wins": 28,
      "B_wins": 15,
      "ties": 7
    },
    "model_metrics": {
      "openai/gpt-4o": {
        "avg_score": 87.5,
        "min_score": 65.2,
        "max_score": 98.3,
        "avg_time_ms": 1200
      },
      "gemini/gemini-2.5-flash": {
        "avg_score": 82.1,
        "min_score": 58.7,
        "max_score": 95.2,
        "avg_time_ms": 850
      }
    }
  },
  "results": [
    {
      "test_id": "menu_001",
      "winner": "A",
      "scores": {
        "openai/gpt-4o": 87.5,
        "gemini/gemini-2.5-flash": 82.3
      },
      "reasons": [
        "Model A achieved higher overall score (87.5% vs 82.3%)",
        "Structure: Model A had better schema compliance (95.0% vs 88.0%)",
        "Content: Model A had higher value accuracy (84.3% vs 79.8%)"
      ],
      "parsed_extractions": {
        "openai/gpt-4o": {...},
        "gemini/gemini-2.5-flash": {...}
      },
      "ground_truth": {...}
    }
  ]
}
```

### Output Files

Two files are generated:

1. **Full results** (`ocr_results_<timestamp>.json`)
   - Complete benchmark data
   - Detailed metrics
   - All parsed extractions

2. **Extractions only** (`ocr_extractions_<timestamp>.json`)
   - Just parsed outputs + ground truth
   - Easier to inspect and compare

### Metrics Provided

**Per-model metrics:**
- Average score (overall)
- Min/max scores
- Average latency

**Per-test metrics:**
- Structure score (0-100)
- Content score (0-100)
- Overall score (0-100)
- Structure diagnostics
  - Missing required fields
  - Type errors
  - Extra keys
  - Constraint violations

**Comparison metrics (2 models):**
- Win/loss/tie counts
- Winner determination
- Score differences
- Per-criteria comparison

## Supported Formats

### Image Formats

- **PNG** (.png)
- **JPEG** (.jpg, .jpeg)
- **GIF** (.gif)
- **WebP** (.webp)
- **BMP** (.bmp)

### Document Formats

- **PDF** (.pdf)
  - Automatic per-page conversion to images
  - Configurable DPI (default: 150)
  - Page limit support
  - Requires PyMuPDF (`pip install PyMuPDF`)

### PDF Processing

PDFs are automatically converted to images (one per page) before sending to the vision model:

```python
# In test case
{
  "pdf_path": "menu.pdf",
  "pdf_dpi": 200,          # Higher quality (default: 150)
  "pdf_max_pages": 3       # Limit pages (default: all)
}
```

Or via CLI:
```bash
domainbench ocr run \
  -d menu.pdf \
  -gt truth.json \
  -m openai/gpt-4o \
  --pdf-dpi 200 \
  --pdf-max-pages 3
```

## Best Practices

### Dataset Creation

1. **Representative samples**: Include variety in image quality, layouts, formats
2. **Accurate ground truth**: Double-check all values
3. **Consistent schema**: Use same structure across test cases
4. **Reasonable size**: 20-50 test cases for initial testing, 100+ for comprehensive evaluation

### Schema Definition

1. **Match extraction task**: Define schema that reflects the actual document structure
2. **Required vs optional**: Only mark truly essential fields as required
3. **Type consistency**: Use correct JSON types (string, number, array, object)
4. **Clear naming**: Use descriptive field names

### Threshold Selection

| Threshold | Use Case |
|-----------|----------|
| **0.9-1.0** | Exact matching (invoices, financial documents) |
| **0.7-0.9** | Standard matching (menus, receipts) |
| **0.5-0.7** | Lenient matching (handwritten, low quality) |

### Model Selection

**GPT-4o**: Best overall accuracy, slower, higher cost
**Gemini-2.5-Flash**: Good accuracy, faster, lower cost
**Gemini-2.0-Flash**: Fastest, lowest cost, slightly lower accuracy

## Troubleshooting

### Common Issues

**Issue**: PDF processing fails
```bash
# Solution: Install PyMuPDF
pip install PyMuPDF

# Or use images instead
convert menu.pdf menu_%d.png  # ImageMagick
```

**Issue**: Low scores despite correct extraction
```bash
# Solution: Check fuzzy matching threshold
domainbench ocr run ... --threshold 0.6

# Or check string normalization
# Model: "PIZZA MARGHERITA" → "pizza margherita"
# Truth: "Pizza Margherita" → "pizza margherita"
# Match: 100%
```

**Issue**: Structure score is 0
```bash
# Solution: Check JSON parsing
cat results/ocr_extractions_*.json | jq '.[] | .extraction'

# Verify schema requirements
cat schema.json | jq '.required'
```

**Issue**: Out of memory with large PDFs
```bash
# Solution: Limit pages or reduce DPI
domainbench ocr run \
  -d large.pdf \
  --pdf-max-pages 10 \
  --pdf-dpi 100
```

## Advanced Usage

### Custom Output Schema

Define complex extraction schemas:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "menu_structure": {
      "type": "object",
      "properties": {
        "items": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "id": {"type": "string"},
              "name": {"type": "string", "minLength": 1},
              "price": {"type": "number", "minimum": 0},
              "description": {"type": "string"},
              "dietary_tags": {
                "type": "array",
                "items": {"type": "string"}
              }
            },
            "required": ["id", "name", "price"]
          }
        },
        "categories": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "name": {"type": "string"},
              "item_ids": {
                "type": "array",
                "items": {"type": "string"}
              }
            },
            "required": ["name"]
          }
        }
      },
      "required": ["items"]
    }
  },
  "required": ["menu_structure"]
}
```

Use it:
```bash
domainbench ocr run \
  -d menus.jsonl \
  -m openai/gpt-4o \
  -so complex_schema.json
```

### Path-Based Weighting

Weight specific JSON paths higher:

```python
from domainbench.capabilities.ocr import SchemaScoreConfig

config = SchemaScoreConfig(
    path_weights={
        "$.items[].price": 2.0,      # Prices are 2x important
        "$.totals.total": 3.0,       # Total is 3x important
        "$.metadata.date": 1.5       # Date is 1.5x important
    }
)
```

### Batch Processing

Process multiple files efficiently:

```bash
# Create dataset from directory
for img in images/*.png; do
  echo "{\"id\": \"$(basename $img .png)\", \"image_path\": \"$img\", \"ground_truth\": $(cat truth/$(basename $img .png).json)}" >> dataset.jsonl
done

# Run benchmark
domainbench ocr run -d dataset.jsonl -m openai/gpt-4o
```

---

For more information, see the [main README](../../../README.md) or [Chat Completion capability documentation](../chat_completion/README.md).
