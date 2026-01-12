"""
OCR Capability - Document/Image extraction benchmark

Supports benchmarking vision models on structured data extraction tasks
(e.g., menu parsing, receipt extraction, document OCR).
Can benchmark a single model or compare two models.
Supported file formats:
- Images: PNG, JPG, JPEG, GIF, WEBP, BMP
- Documents: PDF (automatically converted to images per page)

Scoring System:
- Schema-aware JSON similarity scoring
- Structure score: schema validity/structural correctness (35% default)
- Content score: value similarity with fuzzy matching (65% default)
- Identity-based list matching (by id, uuid, name, etc.)
- Path-based weighting for important fields
"""

import re
import json
import base64
import io
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path

from domainbench.capabilities.base import BaseCapability


# Type alias for JSON values
JSONType = Union[dict, list, str, int, float, bool, None]


# =============================================================================
# Schema Score Configuration
# =============================================================================

@dataclass
class SchemaScoreConfig:
    """
    Configuration for schema-aware JSON similarity scoring.
    
    Attributes:
        structure_weight: Weight for schema validity/structural correctness (0-1)
        content_weight: Weight for value similarity (0-1)
        default_list_mode: How to match lists - "ordered" or "unordered"
        identity_keys: Keys to use for identity-based list matching
        abs_tol: Absolute tolerance for numeric comparison
        rel_tol: Relative tolerance for numeric comparison
        normalize_strings: Whether to normalize strings before comparison
        enforce_additional_properties: Penalize extra keys when schema forbids them
        unordered_min_match: Minimum similarity for unordered list matching
        path_weights: Optional weights for specific JSON paths
    """
    structure_weight: float = 0.35
    content_weight: float = 0.65
    default_list_mode: str = "unordered"
    identity_keys: Tuple[str, ...] = ("id", "uuid", "key", "name")
    abs_tol: float = 0.01
    rel_tol: float = 0.0
    normalize_strings: bool = True
    enforce_additional_properties: bool = True
    unordered_min_match: float = 0.30
    path_weights: Optional[Dict[str, float]] = None


# =============================================================================
# String and Number Utilities
# =============================================================================

def _norm_str(s: str) -> str:
    """Normalize string for comparison."""
    s = s.strip().lower()
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*/\s*", " / ", s)
    return s


def string_sim(a: Optional[str], b: Optional[str], normalize: bool = True) -> float:
    """
    Calculate similarity score between two strings.
    
    Args:
        a: First string
        b: Second string
        normalize: Whether to normalize strings before comparison
        
    Returns:
        Similarity score between 0 and 1
    """
    a = "" if a is None else a
    b = "" if b is None else b
    if normalize:
        a = _norm_str(a)
        b = _norm_str(b)
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def number_sim(a: float, b: float, abs_tol: float = 0.01, rel_tol: float = 0.0) -> float:
    """
    Calculate similarity score between two numbers.
    
    Args:
        a: First number
        b: Second number
        abs_tol: Absolute tolerance
        rel_tol: Relative tolerance
        
    Returns:
        Similarity score between 0 and 1
    """
    if a == b:
        return 1.0
    diff = abs(a - b)
    tol = max(abs_tol, rel_tol * max(abs(a), abs(b)))
    if tol > 0 and diff <= tol:
        return 1.0
    denom = max(1.0, abs(a), abs(b))
    nd = diff / denom
    return 1.0 / (1.0 + nd)


def is_number(x: Any) -> bool:
    """Check if value is a number (int or float, but not bool)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def type_ok(value: Any, schema_type: Union[str, List[str]]) -> bool:
    """Check if value matches schema type."""
    allowed = schema_type if isinstance(schema_type, list) else [schema_type]
    for t in allowed:
        if t == "null" and value is None:
            return True
        if t == "string" and isinstance(value, str):
            return True
        if t == "number" and is_number(value):
            return True
        if t == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if t == "boolean" and isinstance(value, bool):
            return True
        if t == "object" and isinstance(value, dict):
            return True
        if t == "array" and isinstance(value, list):
            return True
    return False


# =============================================================================
# Path Utilities
# =============================================================================

def path_weight(path: str, cfg: SchemaScoreConfig) -> float:
    """
    Get weight for a JSON path.
    
    Supports array wildcard [] in path_weights config.
    E.g., "$.menu_structure.items[].price" matches "$.menu_structure.items[0].price"
    """
    if not cfg.path_weights:
        return 1.0
    # Normalize path: convert [123] -> []
    norm = re.sub(r"\[\d+\]", "[]", path)
    return cfg.path_weights.get(norm, 1.0)


def _join_path(base: str, key: str) -> str:
    """Join JSON path with key."""
    return f"{base}.{key}" if base != "$" else f"$.{key}"


def _idx_path(base: str, idx: int) -> str:
    """Join JSON path with array index."""
    return f"{base}[{idx}]"


# =============================================================================
# Structure Scoring (Schema Validation)
# =============================================================================

@dataclass
class StructureResult:
    """Result of structure scoring."""
    score: float  # 0..1
    missing_required: int = 0
    type_errors: int = 0
    constraint_errors: int = 0
    extra_keys: int = 0
    notes: List[str] = field(default_factory=list)


def score_structure(
    value: JSONType,
    schema: Dict[str, Any],
    cfg: SchemaScoreConfig,
    path: str = "$"
) -> StructureResult:
    """
    Score structural correctness of value against schema.
    
    Args:
        value: The JSON value to check
        schema: JSON Schema (draft-07 subset)
        cfg: Score configuration
        path: Current JSON path (for error reporting)
        
    Returns:
        StructureResult with score and diagnostics
    """
    notes: List[str] = []
    
    # Type check if present
    if "type" in schema:
        if not type_ok(value, schema["type"]):
            return StructureResult(
                score=0.0,
                type_errors=1,
                notes=[f"{path}: type mismatch, expected {schema['type']}, got {type(value).__name__}"]
            )
    
    # Constraints for strings
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            return StructureResult(0.0, constraint_errors=1, notes=[f"{path}: minLength violation"])
        return StructureResult(1.0, notes=[])
    
    # Constraints for numbers
    if is_number(value):
        if "minimum" in schema and value < schema["minimum"]:
            return StructureResult(0.0, constraint_errors=1, notes=[f"{path}: minimum violation"])
        return StructureResult(1.0, notes=[])
    
    # Objects
    if isinstance(value, dict):
        props = schema.get("properties", {})
        required = schema.get("required", [])
        addl = schema.get("additionalProperties", True)
        
        missing = [k for k in required if k not in value]
        missing_required = len(missing)
        if missing:
            notes.append(f"{path}: missing required keys {missing}")
        
        extra = []
        if cfg.enforce_additional_properties and addl is False:
            extra = [k for k in value.keys() if k not in props]
            if extra:
                notes.append(f"{path}: extra keys not allowed {extra[:20]}")
        extra_keys = len(extra)
        
        # Recurse into known props
        child_scores = []
        type_errors = 0
        constraint_errors = 0
        for k, subschema in props.items():
            if k not in value:
                continue
            r = score_structure(value[k], subschema, cfg, _join_path(path, k))
            child_scores.append(r.score)
            missing_required += r.missing_required
            type_errors += r.type_errors
            constraint_errors += r.constraint_errors
            extra_keys += r.extra_keys
            notes.extend(r.notes or [])
        
        base = sum(child_scores) / len(child_scores) if child_scores else 1.0
        
        # Penalize missing required + extra keys
        penalty = 1.0
        if required:
            penalty *= max(0.0, 1.0 - (len(missing) / max(1, len(required))))
        if cfg.enforce_additional_properties and addl is False and len(value) > 0:
            penalty *= max(0.0, 1.0 - (len(extra) / max(1, len(value))))
        
        score = max(0.0, min(1.0, base * penalty))
        
        return StructureResult(
            score=score,
            missing_required=missing_required,
            type_errors=type_errors,
            constraint_errors=constraint_errors,
            extra_keys=extra_keys,
            notes=notes
        )
    
    # Arrays
    if isinstance(value, list):
        item_schema = schema.get("items")
        if not item_schema:
            return StructureResult(1.0, notes=[])
        
        child_scores = []
        missing_required = type_errors = constraint_errors = extra_keys = 0
        for i, elem in enumerate(value):
            r = score_structure(elem, item_schema, cfg, _idx_path(path, i))
            child_scores.append(r.score)
            missing_required += r.missing_required
            type_errors += r.type_errors
            constraint_errors += r.constraint_errors
            extra_keys += r.extra_keys
            notes.extend(r.notes or [])
        
        base = sum(child_scores) / len(child_scores) if child_scores else 1.0
        return StructureResult(
            score=base,
            missing_required=missing_required,
            type_errors=type_errors,
            constraint_errors=constraint_errors,
            extra_keys=extra_keys,
            notes=notes
        )
    
    # Null / bool / others
    return StructureResult(1.0, notes=[])


# =============================================================================
# Content Similarity Scoring (Schema-Guided)
# =============================================================================

def element_identity(elem: Any, keys: Tuple[str, ...]) -> Optional[Tuple[str, str]]:
    """
    Get identity key-value pair from an element for list matching.
    
    Args:
        elem: Element to check
        keys: Identity keys to look for
        
    Returns:
        Tuple of (key_name, normalized_value) or None if no identity found
    """
    if not isinstance(elem, dict):
        return None
    for k in keys:
        if k in elem and elem[k] is not None:
            return (k, _norm_str(str(elem[k])))
    return None


def compare_content(
    pred: JSONType,
    gt: JSONType,
    schema: Dict[str, Any],
    cfg: SchemaScoreConfig,
    path: str = "$"
) -> float:
    """
    Compare content similarity between predicted and ground truth values.
    
    Uses schema to decide list matching strategy, types, and recursion.
    
    Args:
        pred: Predicted/extracted value
        gt: Ground truth value
        schema: JSON Schema for this value
        cfg: Score configuration
        path: Current JSON path
        
    Returns:
        Similarity score between 0 and 1
    """
    w = path_weight(path, cfg)
    
    # Type check
    if "type" in schema and not type_ok(pred, schema["type"]):
        return 0.0
    
    # Nulls
    if gt is None:
        return 1.0 if pred is None else 0.0
    
    # Booleans
    if isinstance(gt, bool):
        return 1.0 if pred == gt else 0.0
    
    # Numbers
    if is_number(gt):
        if not is_number(pred):
            return 0.0
        return number_sim(float(pred), float(gt), cfg.abs_tol, cfg.rel_tol)
    
    # Strings
    if isinstance(gt, str):
        if not isinstance(pred, str):
            return 0.0
        return string_sim(pred, gt, cfg.normalize_strings)
    
    # Objects
    if isinstance(gt, dict):
        if not isinstance(pred, dict):
            return 0.0
        
        props = schema.get("properties", {})
        scores = []
        weights = []
        
        for k, subschema in props.items():
            child_path = _join_path(path, k)
            cw = path_weight(child_path, cfg)
            
            if k not in gt and k not in pred:
                continue
            
            # Missing in pred => 0 for that field
            if k in gt and k not in pred:
                scores.append(0.0)
                weights.append(cw)
                continue
            
            # Extra in pred but not in GT
            if k not in gt and k in pred:
                if cfg.enforce_additional_properties and schema.get("additionalProperties", True) is False:
                    scores.append(0.0)
                    weights.append(cw)
                continue
            
            # Compare values
            scores.append(compare_content(pred.get(k), gt.get(k), subschema, cfg, child_path))
            weights.append(cw)
        
        if not scores:
            return 1.0
        
        return sum(s * wt for s, wt in zip(scores, weights)) / max(1e-9, sum(weights))
    
    # Arrays
    if isinstance(gt, list):
        if not isinstance(pred, list):
            return 0.0
        
        item_schema = schema.get("items", {})
        
        # Build identity maps for smarter matching
        pred_id_map = {}
        gt_id_map = {}
        
        for j, g in enumerate(gt):
            sid = element_identity(g, cfg.identity_keys)
            if sid:
                gt_id_map[sid] = j
        for i, p in enumerate(pred):
            sid = element_identity(p, cfg.identity_keys)
            if sid:
                pred_id_map[sid] = i
        
        matched_pred = set()
        matched_gt = set()
        sims = []
        
        # Match by identity where possible
        for sid, gj in gt_id_map.items():
            if sid in pred_id_map:
                pi = pred_id_map[sid]
                matched_pred.add(pi)
                matched_gt.add(gj)
                sims.append(compare_content(pred[pi], gt[gj], item_schema, cfg, f"{path}[{sid[0]}={sid[1]}]"))
        
        # Remaining elements
        rem_pred = [i for i in range(len(pred)) if i not in matched_pred]
        rem_gt = [j for j in range(len(gt)) if j not in matched_gt]
        
        if cfg.default_list_mode == "ordered" and not gt_id_map:
            # Positional compare
            n = max(len(pred), len(gt))
            if n == 0:
                return 1.0
            sims = []
            for idx in range(n):
                if idx >= len(gt) or idx >= len(pred):
                    sims.append(0.0)
                else:
                    sims.append(compare_content(pred[idx], gt[idx], item_schema, cfg, _idx_path(path, idx)))
            return sum(sims) / len(sims)
        
        # Unordered greedy matching for remainder
        pairs: List[Tuple[float, int, int]] = []
        for i in rem_pred:
            for j in rem_gt:
                s = compare_content(pred[i], gt[j], item_schema, cfg, _idx_path(path, j))
                if s >= cfg.unordered_min_match:
                    pairs.append((s, i, j))
        
        pairs.sort(reverse=True, key=lambda x: x[0])
        for s, i, j in pairs:
            if i in matched_pred or j in matched_gt:
                continue
            matched_pred.add(i)
            matched_gt.add(j)
            sims.append(s)
        
        # Coverage-based aggregation
        if len(gt) == 0:
            return 1.0 if len(pred) == 0 else (0.0 if cfg.enforce_additional_properties else 1.0)
        
        coverage = len(matched_gt) / len(gt)
        base = (sum(sims) / len(sims)) if sims else 0.0
        
        # Penalize extra predicted elements
        extra = len(pred) - len(matched_pred)
        if cfg.enforce_additional_properties and extra > 0:
            base *= max(0.0, 1.0 - extra / max(1, len(pred)))
        
        return base * coverage
    
    return 1.0


# =============================================================================
# High-Level Scoring API
# =============================================================================

def score_with_schema(
    pred: JSONType,
    gt: JSONType,
    schema: Dict[str, Any],
    cfg: Optional[SchemaScoreConfig] = None
) -> Dict[str, Any]:
    """
    Calculate comprehensive schema-aware similarity score.
    
    Args:
        pred: Predicted/extracted JSON
        gt: Ground truth JSON
        schema: JSON Schema for validation and comparison
        cfg: Optional score configuration
        
    Returns:
        Dict with structure score, content score, overall score, and diagnostics
    """
    if cfg is None:
        cfg = SchemaScoreConfig()
    
    identical = pred == gt
    
    # Structure correctness for pred
    st = score_structure(pred, schema, cfg, path="$")
    
    # Content similarity guided by schema
    content = compare_content(pred, gt, schema, cfg, path="$")
    
    overall = cfg.structure_weight * st.score + cfg.content_weight * content
    
    return {
        "identical_after_parse": identical,
        "scores": {
            "structure_0_to_100": round(st.score * 100, 2),
            "content_0_to_100": round(content * 100, 2),
            "overall_0_to_100": round(overall * 100, 2),
        },
        "structure_diagnostics": {
            "missing_required": st.missing_required,
            "type_errors": st.type_errors,
            "constraint_errors": st.constraint_errors,
            "extra_keys": st.extra_keys,
            "notes_sample": (st.notes or [])[:25],
        },
        "config": {
            "structure_weight": cfg.structure_weight,
            "content_weight": cfg.content_weight,
            "default_list_mode": cfg.default_list_mode,
            "identity_keys": cfg.identity_keys,
            "abs_tol": cfg.abs_tol,
            "rel_tol": cfg.rel_tol,
            "normalize_strings": cfg.normalize_strings,
            "enforce_additional_properties": cfg.enforce_additional_properties,
            "unordered_min_match": cfg.unordered_min_match,
            "path_weights": cfg.path_weights or {},
        },
    }


# =============================================================================
# Legacy Compatibility Functions
# =============================================================================

def normalize_ground_truth(ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize ground truth structure to handle nested formats.
    
    Handles cases like:
    - {"items": [...]} -> direct use
    - {"menu_structure": {"items": [...]}} -> unwrap
    - {"data": {"items": [...]}} -> unwrap
    
    Args:
        ground_truth: Ground truth data (may be nested)
        
    Returns:
        Normalized ground truth with components at root level
    """
    if not ground_truth:
        return {}
    
    # Common wrapper keys to check
    wrapper_keys = ["menu_structure", "data", "extraction", "result", "content", "output"]
    
    # Check if ground truth is wrapped
    for wrapper_key in wrapper_keys:
        if wrapper_key in ground_truth and isinstance(ground_truth[wrapper_key], dict):
            inner = ground_truth[wrapper_key]
            if any(key in inner for key in ["items", "categories", "modifiers", "fields"]):
                return inner
    
    # If items/categories exist at root level, use as-is
    if any(key in ground_truth for key in ["items", "categories", "modifiers", "fields"]):
        return ground_truth
    
    # If only one key and it's a dict, try unwrapping
    if len(ground_truth) == 1:
        only_key = list(ground_truth.keys())[0]
        if isinstance(ground_truth[only_key], dict):
            return ground_truth[only_key]
    
    return ground_truth


def infer_schema_from_ground_truth(ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    """
    Infer a JSON Schema from ground truth structure.
    
    Args:
        ground_truth: Ground truth data to infer schema from
        
    Returns:
        Inferred JSON Schema
    """
    def infer_type(value: Any) -> Dict[str, Any]:
        if value is None:
            return {"type": "null"}
        if isinstance(value, bool):
            return {"type": "boolean"}
        if isinstance(value, int):
            return {"type": "integer"}
        if isinstance(value, float):
            return {"type": "number"}
        if isinstance(value, str):
            return {"type": "string"}
        if isinstance(value, list):
            if not value:
                return {"type": "array", "items": {}}
            # Infer from first element
            return {"type": "array", "items": infer_type(value[0])}
        if isinstance(value, dict):
            props = {}
            for k, v in value.items():
                props[k] = infer_type(v)
            return {
                "type": "object",
                "properties": props,
                "required": list(value.keys()),
            }
        return {}
    
    return infer_type(ground_truth)


def calculate_extraction_accuracy(
    parsed_result: Dict[str, Any],
    ground_truth: Dict[str, Any],
    schema: Optional[Dict[str, Any]] = None,
    score_config: Optional[SchemaScoreConfig] = None,
) -> Dict[str, Any]:
    """
    Calculate comprehensive accuracy metrics for structured extraction.
    
    Uses schema-aware JSON similarity scoring with structure and content scores.
    
    Args:
        parsed_result: The extracted data from the model
        ground_truth: The ground truth data
        schema: Optional JSON Schema (will be inferred if not provided)
        score_config: Optional scoring configuration
        
    Returns:
        Dict containing structure score, content score, overall score, and diagnostics
    """
    # Normalize ground truth
    gt_normalized = normalize_ground_truth(ground_truth)
    pred_normalized = normalize_ground_truth(parsed_result)
    
    # Use provided schema or infer from ground truth
    if schema is None:
        schema = infer_schema_from_ground_truth(gt_normalized)
    
    # Use provided config or default
    cfg = score_config or SchemaScoreConfig()
    
    # Calculate scores
    result = score_with_schema(pred_normalized, gt_normalized, schema, cfg)
    
    # Add overall F1-like score for backward compatibility
    result["overall"] = {
        "f1_score": result["scores"]["overall_0_to_100"],
        "structure_score": result["scores"]["structure_0_to_100"],
        "content_score": result["scores"]["content_0_to_100"],
    }
    
    return result


def compare_model_results(
    metrics_a: Dict[str, Any],
    metrics_b: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare results from two models.
    
    Args:
        metrics_a: Metrics from model A
        metrics_b: Metrics from model B
        
    Returns:
        Comparison result with winner and score differences
    """
    score_a = metrics_a.get("overall", {}).get("f1_score", 0)
    score_b = metrics_b.get("overall", {}).get("f1_score", 0)
    
    # Determine winner (tie if within 2% threshold)
    diff = abs(score_a - score_b)
    if diff < 2.0:
        winner = "tie"
    elif score_a > score_b:
        winner = "A"
    else:
        winner = "B"
    
    # Structure score comparison
    struct_a = metrics_a.get("overall", {}).get("structure_score", 0)
    struct_b = metrics_b.get("overall", {}).get("structure_score", 0)
    
    # Content score comparison
    content_a = metrics_a.get("overall", {}).get("content_score", 0)
    content_b = metrics_b.get("overall", {}).get("content_score", 0)
    
    return {
        "winner": winner,
        "score_A": score_a,
        "score_B": score_b,
        "difference": round(diff, 2),
        "structure_comparison": {
            "score_A": struct_a,
            "score_B": struct_b,
            "winner": "A" if struct_a > struct_b + 2 else ("B" if struct_b > struct_a + 2 else "tie"),
        },
        "content_comparison": {
            "score_A": content_a,
            "score_B": content_b,
            "winner": "A" if content_a > content_b + 2 else ("B" if content_b > content_a + 2 else "tie"),
        },
    }


# =============================================================================
# Image Processing Utilities
# =============================================================================

def is_pdf_file(file_path: str) -> bool:
    """Check if file is a PDF."""
    return Path(file_path).suffix.lower() == ".pdf"


def convert_pdf_to_images(
    pdf_path: str,
    dpi: int = 150,
    max_pages: Optional[int] = None,
) -> List[Tuple[bytes, str]]:
    """
    Convert PDF pages to images using PyMuPDF (fitz).
    
    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution for rendering (default: 150)
        max_pages: Maximum number of pages to convert (None = all)
        
    Returns:
        List of tuples: (image_bytes, mime_type) for each page
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF is required for PDF support. "
            "Install it with: pip install PyMuPDF"
        )
    
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    images = []
    doc = fitz.open(pdf_path)
    
    try:
        num_pages = len(doc)
        if max_pages:
            num_pages = min(num_pages, max_pages)
        
        # Calculate zoom factor from DPI (72 is default PDF DPI)
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        
        for page_num in range(num_pages):
            page = doc[page_num]
            # Render page to pixmap (image)
            pix = page.get_pixmap(matrix=matrix)
            
            # Convert to PNG bytes
            img_bytes = pix.tobytes("png")
            images.append((img_bytes, "image/png"))
    finally:
        doc.close()
    
    return images


def load_image_as_base64(image_path: str) -> str:
    """
    Load an image file and encode it as base64.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_file_as_images(
    file_path: str,
    pdf_dpi: int = 150,
    pdf_max_pages: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """
    Load a file (image or PDF) and return base64 encoded images.
    
    For images: Returns single image as base64
    For PDFs: Converts each page to image and returns all as base64
    
    Args:
        file_path: Path to image or PDF file
        pdf_dpi: DPI for PDF rendering (default: 150)
        pdf_max_pages: Max pages to convert from PDF (None = all)
        
    Returns:
        List of tuples: (base64_string, mime_type) for each image/page
    """
    if is_pdf_file(file_path):
        # Convert PDF pages to images
        page_images = convert_pdf_to_images(file_path, pdf_dpi, pdf_max_pages)
        return [
            (base64.b64encode(img_bytes).decode("utf-8"), mime_type)
            for img_bytes, mime_type in page_images
        ]
    else:
        # Load regular image
        base64_img = load_image_as_base64(file_path)
        mime_type = get_image_mime_type(file_path)
        return [(base64_img, mime_type)]


def get_image_mime_type(image_path: str) -> str:
    """
    Get the MIME type of an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        MIME type string (e.g., "image/png", "image/jpeg")
    """
    ext = Path(image_path).suffix.lower()
    mime_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".pdf": "application/pdf",
    }
    return mime_types.get(ext, "image/png")


# =============================================================================
# OCR Capability Class
# =============================================================================

class OCRCapability(BaseCapability):
    """
    OCR/Vision Extraction capability for structured data extraction from images and PDFs.
    
    Supports:
    - Single model evaluation against ground truth
    - Two model comparison (head-to-head)
    - Fuzzy matching for accuracy calculation
    - Multiple extraction schemas (menu, receipt, document, custom)
    - PDF files (automatically converted to images per page)
    
    Test cases should have:
    - 'image_path' or 'image_paths': Path(s) to image or PDF file(s)
    - 'pdf_path' or 'pdf_paths': Alternative for PDF files specifically
    - 'ground_truth': Expected extraction result
    - 'schema': (optional) Extraction schema configuration
    
    PDF Options (in test_case):
    - 'pdf_dpi': Resolution for PDF rendering (default: 150)
    - 'pdf_max_pages': Maximum pages to process (default: all)
    """
    
    name = "ocr"
    description = "Vision-based structured data extraction benchmark (OCR/PDF)"
    required_provider_features = ["vision"]
    
    # Default extraction prompt
    DEFAULT_EXTRACTION_PROMPT = """Extract all structured information from this image.
Return the result as valid JSON matching the expected schema.
Be thorough and accurate - extract ALL visible items."""
    
    def __init__(
        self,
        schema_config: Optional[Dict[str, Any]] = None,
        pdf_dpi: int = 150,
        pdf_max_pages: Optional[int] = None,
    ):
        """
        Initialize OCR capability.
        
        Args:
            schema_config: Optional schema configuration for evaluation
            pdf_dpi: Default DPI for PDF rendering (default: 150)
            pdf_max_pages: Default max pages to process from PDFs (None = all)
        """
        self.schema_config = schema_config
        self.pdf_dpi = pdf_dpi
        self.pdf_max_pages = pdf_max_pages
    
    def build_messages(
        self,
        test_case: Dict[str, Any],
        system_prompt: str,
    ) -> List[Dict[str, Any]]:
        """
        Build messages with image/PDF content for vision API.
        
        Automatically handles:
        - Image files (PNG, JPG, etc.): Loaded directly
        - PDF files: Converted to images (one per page)
        
        Args:
            test_case: Dict with 'image_path'/'image_paths' or 'pdf_path'/'pdf_paths'
            system_prompt: System prompt from domain config
            
        Returns:
            List of messages for vision API (with image content)
        """
        messages = []
        
        # Add system prompt
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })
        
        # Get file paths (support both image_path(s) and pdf_path(s))
        file_paths = []
        
        # Check for image paths
        if test_case.get("image_paths"):
            file_paths.extend(test_case["image_paths"])
        elif test_case.get("image_path"):
            file_paths.append(test_case["image_path"])
        
        # Check for PDF paths (can be combined with images)
        if test_case.get("pdf_paths"):
            file_paths.extend(test_case["pdf_paths"])
        elif test_case.get("pdf_path"):
            file_paths.append(test_case["pdf_path"])
        
        # Get PDF conversion settings from test_case or use defaults
        pdf_dpi = test_case.get("pdf_dpi", self.pdf_dpi)
        pdf_max_pages = test_case.get("pdf_max_pages", self.pdf_max_pages)
        
        # Build content with images
        content = []
        
        # Add text prompt
        prompt = test_case.get("prompt", self.DEFAULT_EXTRACTION_PROMPT)
        
        # If schema is provided, include it in the prompt
        schema = test_case.get("schema") or test_case.get("output_schema")
        if schema:
            prompt += f"\n\nExpected output schema:\n{json.dumps(schema, indent=2)}"
        
        content.append({
            "type": "text",
            "text": prompt,
        })
        
        # Process all files (images and PDFs)
        for file_path in file_paths:
            try:
                # load_file_as_images handles both images and PDFs
                images = load_file_as_images(
                    file_path,
                    pdf_dpi=pdf_dpi,
                    pdf_max_pages=pdf_max_pages,
                )
                
                for base64_image, mime_type in images:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}",
                        },
                    })
                    
            except FileNotFoundError:
                # Log error but continue with other files
                content.append({
                    "type": "text",
                    "text": f"[File not found: {file_path}]",
                })
            except ImportError as e:
                # PyMuPDF not installed
                content.append({
                    "type": "text",
                    "text": f"[PDF processing error: {e}]",
                })
        
        messages.append({
            "role": "user",
            "content": content,
        })
        
        return messages
    
    def validate_test_case(self, test_case: Dict[str, Any]) -> bool:
        """
        Validate test case has required image/PDF and ground truth fields.
        
        Args:
            test_case: Test case data
            
        Returns:
            True if valid, False otherwise
        """
        # Must have image or PDF path(s)
        has_files = (
            "image_path" in test_case or 
            ("image_paths" in test_case and len(test_case["image_paths"]) > 0) or
            "pdf_path" in test_case or
            ("pdf_paths" in test_case and len(test_case["pdf_paths"]) > 0)
        )
        
        if not has_files:
            return False
        
        # Should have ground truth for evaluation
        # (not strictly required - can run without for pure inference)
        return True
    
    def get_required_fields(self) -> List[str]:
        """Required fields for OCR test cases"""
        return ["id"]  # image_path, image_paths, pdf_path, or pdf_paths
    
    def get_metrics(self) -> List[str]:
        """Metrics collected by OCR capability"""
        return [
            "structure_score",
            "content_score",
            "overall_score",
            "latency",
            "tokens",
        ]
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse model response into structured data.
        
        Args:
            response: Raw response string from model
            
        Returns:
            Parsed dictionary or empty dict on failure
        """
        # Try to extract JSON from potential markdown code blocks
        text = response.strip()
        
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in text
            try:
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        
        return {}
    
    def evaluate_single(
        self,
        response: str,
        ground_truth: Dict[str, Any],
        schema_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single model's extraction against ground truth.
        
        Args:
            response: Model's raw response
            ground_truth: Expected extraction result
            schema_config: Optional schema configuration
            
        Returns:
            Dict with accuracy metrics
        """
        parsed = self.parse_response(response)
        
        config = schema_config or self.schema_config
        metrics = calculate_extraction_accuracy(parsed, ground_truth, config)
        
        return {
            "parsed_result": parsed,
            "metrics": metrics,
            "overall_score": metrics.get("overall", {}).get("f1_score", 0),
        }
    
    def evaluate_pair(
        self,
        response_a: str,
        response_b: str,
        ground_truth: Dict[str, Any],
        schema_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate and compare two models' extractions.
        
        Args:
            response_a: Model A's raw response
            response_b: Model B's raw response
            ground_truth: Expected extraction result
            schema_config: Optional schema configuration
            
        Returns:
            Dict with comparison results including winner
        """
        # Evaluate each model
        eval_a = self.evaluate_single(response_a, ground_truth, schema_config)
        eval_b = self.evaluate_single(response_b, ground_truth, schema_config)
        
        # Compare results
        comparison = compare_model_results(eval_a["metrics"], eval_b["metrics"])
        
        return {
            "winner": comparison["winner"],
            "score_A": comparison["score_A"],
            "score_B": comparison["score_B"],
            "difference": comparison["difference"],
            "structure_comparison": comparison.get("structure_comparison", {}),
            "content_comparison": comparison.get("content_comparison", {}),
            "model_A": eval_a,
            "model_B": eval_b,
            "reasons": self._generate_comparison_reasons(eval_a, eval_b, comparison),
        }
    
    def _generate_comparison_reasons(
        self,
        eval_a: Dict[str, Any],
        eval_b: Dict[str, Any],
        comparison: Dict[str, Any],
    ) -> List[str]:
        """Generate human-readable reasons for the comparison result."""
        reasons = []
        
        score_a = comparison["score_A"]
        score_b = comparison["score_B"]
        winner = comparison["winner"]
        
        if winner == "tie":
            reasons.append(f"Models performed similarly (A: {score_a:.1f}%, B: {score_b:.1f}%)")
        else:
            winning_score = score_a if winner == "A" else score_b
            losing_score = score_b if winner == "A" else score_a
            reasons.append(f"Model {winner} achieved higher overall score ({winning_score:.1f}% vs {losing_score:.1f}%)")
        
        # Add structure-level insights
        struct_comp = comparison.get("structure_comparison", {})
        if struct_comp.get("winner") and struct_comp["winner"] != "tie":
            reasons.append(
                f"Structure: Model {struct_comp['winner']} had better schema compliance "
                f"({struct_comp['score_A']:.1f}% vs {struct_comp['score_B']:.1f}%)"
            )
        
        # Add content-level insights
        content_comp = comparison.get("content_comparison", {})
        if content_comp.get("winner") and content_comp["winner"] != "tie":
            reasons.append(
                f"Content: Model {content_comp['winner']} had higher value accuracy "
                f"({content_comp['score_A']:.1f}% vs {content_comp['score_B']:.1f}%)"
            )
        
        return reasons
    
    def format_result_for_display(self, result: Dict[str, Any]) -> str:
        """Format evaluation result for console display."""
        lines = []
        
        metrics = result.get("metrics", {})
        overall = metrics.get("overall", {})
        
        # Display scores breakdown
        lines.append(f"  {'STRUCTURE':15} | Score: {overall.get('structure_score', 0):5.1f}%")
        lines.append(f"  {'CONTENT':15} | Score: {overall.get('content_score', 0):5.1f}%")
        lines.append(f"  {'OVERALL':15} | Score: {overall.get('f1_score', 0):5.1f}%")
        
        # Display diagnostics if present
        diagnostics = metrics.get("structure_diagnostics", {})
        if diagnostics:
            issues = []
            if diagnostics.get("missing_required", 0) > 0:
                issues.append(f"{diagnostics['missing_required']} missing required")
            if diagnostics.get("type_errors", 0) > 0:
                issues.append(f"{diagnostics['type_errors']} type errors")
            if diagnostics.get("extra_keys", 0) > 0:
                issues.append(f"{diagnostics['extra_keys']} extra keys")
            if issues:
                lines.append(f"  {'ISSUES':15} | {', '.join(issues)}")
        
        return "\n".join(lines)


# =============================================================================
# Pre-built Schema Configurations
# =============================================================================

MENU_EXTRACTION_SCHEMA = {
    "components": {
        "items": {
            "name_field": "name",
            "weight": 0.5,
            "additional_fields": ["price", "description"],
        },
        "categories": {
            "name_field": "name",
            "weight": 0.2,
        },
        "modifier_lists": {
            "name_field": "name",
            "weight": 0.15,
        },
        "modifiers": {
            "name_field": "name",
            "weight": 0.15,
        },
    },
    "threshold": 0.7,
}

RECEIPT_EXTRACTION_SCHEMA = {
    "components": {
        "items": {
            "name_field": "name",
            "weight": 0.5,
            "additional_fields": ["price", "quantity"],
        },
        "totals": {
            "name_field": "label",
            "weight": 0.3,
            "additional_fields": ["amount"],
        },
        "metadata": {
            "name_field": "field",
            "weight": 0.2,
            "additional_fields": ["value"],
        },
    },
    "threshold": 0.7,
}

DOCUMENT_EXTRACTION_SCHEMA = {
    "components": {
        "fields": {
            "name_field": "field_name",
            "weight": 0.6,
            "additional_fields": ["value"],
        },
        "tables": {
            "name_field": "table_name",
            "weight": 0.4,
        },
    },
    "threshold": 0.7,
}


def get_schema_config(schema_type: str) -> Dict[str, Any]:
    """
    Get a predefined schema configuration by name.
    
    Args:
        schema_type: One of 'menu', 'receipt', 'document', or 'custom'
        
    Returns:
        Schema configuration dictionary
    """
    schemas = {
        "menu": MENU_EXTRACTION_SCHEMA,
        "receipt": RECEIPT_EXTRACTION_SCHEMA,
        "document": DOCUMENT_EXTRACTION_SCHEMA,
    }
    return schemas.get(schema_type, MENU_EXTRACTION_SCHEMA)
