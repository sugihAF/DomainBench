"""
Test Case Converter - Convert YAML/CSV formats to JSONL

Allows users to create test cases in more user-friendly formats
and convert them to the JSONL format used by the benchmark engine.
"""

import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Optional


def convert_yaml_to_jsonl(input_path: str, output_path: str) -> int:
    """
    Convert a YAML test cases file to JSONL format.
    
    Args:
        input_path: Path to input YAML file
        output_path: Path to output JSONL file
        
    Returns:
        Number of test cases converted
    """
    import yaml
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    # Handle both formats: list directly or under 'test_cases' key
    if isinstance(data, list):
        test_cases = data
    elif isinstance(data, dict) and 'test_cases' in data:
        test_cases = data['test_cases']
    else:
        raise ValueError("YAML must contain a list of test cases or a 'test_cases' key")
    
    items = []
    for i, tc in enumerate(test_cases):
        item = _normalize_test_case(tc, i + 1)
        items.append(item)
    
    # Write JSONL
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return len(items)


def convert_csv_to_jsonl(input_path: str, output_path: str) -> int:
    """
    Convert a CSV test cases file to JSONL format.
    
    Expected CSV columns:
    - id (optional): Test case ID
    - category (optional): Test category
    - turn_1, turn_2, turn_3, ... : User turns (at least turn_1 required)
    - safety_critical (optional): "yes"/"no" or "true"/"false"
    - population (optional): e.g., "adult", "pediatric", "pregnancy"
    - Any other columns become part of 'meta'
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output JSONL file
        
    Returns:
        Number of test cases converted
    """
    items = []
    
    with open(input_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        
        for i, row in enumerate(reader):
            item = _csv_row_to_test_case(row, i + 1)
            if item:  # Skip empty rows
                items.append(item)
    
    # Write JSONL
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return len(items)


def _normalize_test_case(tc: Dict[str, Any], index: int) -> Dict[str, Any]:
    """
    Normalize a test case from YAML format to internal format.
    
    Handles flexible input:
    - 'turns' can be a list of strings
    - 'id' is auto-generated if missing
    - 'category' defaults to 'general'
    - Other fields go into 'meta'
    """
    # Required: turns
    turns = tc.get('turns', [])
    if not turns:
        raise ValueError(f"Test case {index} is missing 'turns' field")
    
    # Ensure turns are strings
    turns = [str(t) for t in turns]
    
    # Build normalized item
    item = {
        "id": tc.get('id', f"tc_{index:04d}"),
        "category": tc.get('category', 'general'),
        "turns": turns,
    }
    
    # Build meta from remaining fields
    meta = {}
    known_fields = {'id', 'category', 'turns', 'meta'}
    
    for key, value in tc.items():
        if key not in known_fields:
            # Handle boolean-like strings
            if isinstance(value, str) and value.lower() in ('yes', 'true'):
                value = True
            elif isinstance(value, str) and value.lower() in ('no', 'false'):
                value = False
            meta[key] = value
    
    # Merge with explicit meta if provided
    if 'meta' in tc and isinstance(tc['meta'], dict):
        meta.update(tc['meta'])
    
    if meta:
        item['meta'] = meta
    
    return item


def _csv_row_to_test_case(row: Dict[str, str], index: int) -> Optional[Dict[str, Any]]:
    """
    Convert a CSV row to a test case.
    
    Extracts turns from turn_1, turn_2, ... columns.
    """
    # Collect turns from turn_N columns
    turns = []
    turn_num = 1
    while True:
        turn_key = f"turn_{turn_num}"
        if turn_key in row and row[turn_key].strip():
            turns.append(row[turn_key].strip())
            turn_num += 1
        else:
            break
    
    # Skip if no turns found
    if not turns:
        return None
    
    # Build item
    item = {
        "id": row.get('id', '').strip() or f"tc_{index:04d}",
        "category": row.get('category', '').strip() or 'general',
        "turns": turns,
    }
    
    # Build meta from other columns
    meta = {}
    known_columns = {'id', 'category'}
    turn_columns = {f"turn_{i}" for i in range(1, 20)}  # Skip turn columns
    
    for key, value in row.items():
        if key not in known_columns and key not in turn_columns and value.strip():
            # Handle boolean-like strings
            val = value.strip()
            if val.lower() in ('yes', 'true'):
                meta[key] = True
            elif val.lower() in ('no', 'false'):
                meta[key] = False
            else:
                meta[key] = val
    
    if meta:
        item['meta'] = meta
    
    return item


def detect_format(input_path: str) -> str:
    """
    Detect the format of an input file based on extension.
    
    Returns: 'yaml', 'csv', or 'jsonl'
    """
    path = Path(input_path)
    suffix = path.suffix.lower()
    
    # Handle .example suffix
    if suffix == '.example':
        suffix = path.with_suffix('').suffix.lower()
    
    if suffix in ('.yaml', '.yml'):
        return 'yaml'
    elif suffix == '.csv':
        return 'csv'
    elif suffix in ('.jsonl', '.json'):
        return 'jsonl'
    else:
        raise ValueError(f"Unknown file format: {suffix}. Supported: .yaml, .yml, .csv, .jsonl")


def convert_to_jsonl(input_path: str, output_path: Optional[str] = None) -> tuple[str, int]:
    """
    Auto-detect format and convert to JSONL.
    
    Args:
        input_path: Path to input file (YAML or CSV)
        output_path: Path to output JSONL file (auto-generated if not provided)
        
    Returns:
        Tuple of (output_path, count)
    """
    format_type = detect_format(input_path)
    
    # Auto-generate output path if not provided
    if output_path is None:
        output_path = str(Path(input_path).with_suffix('.jsonl'))
    
    if format_type == 'yaml':
        count = convert_yaml_to_jsonl(input_path, output_path)
    elif format_type == 'csv':
        count = convert_csv_to_jsonl(input_path, output_path)
    elif format_type == 'jsonl':
        raise ValueError("Input is already JSONL format, no conversion needed")
    else:
        raise ValueError(f"Unsupported format: {format_type}")
    
    return output_path, count
