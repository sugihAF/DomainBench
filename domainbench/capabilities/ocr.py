"""
OCR Capability - Document/Image extraction benchmark

Supports benchmarking vision models on structured data extraction tasks
(e.g., menu parsing, receipt extraction, document OCR).
Can benchmark a single model or compare two models.
"""

import re
import json
import base64
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from domainbench.capabilities.base import BaseCapability


# =============================================================================
# Fuzzy Matching Utilities
# =============================================================================

def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.
    
    Args:
        text: Input text string
        
    Returns:
        Normalized lowercase text with standardized whitespace
    """
    if not text:
        return ""
    # Lowercase, remove extra whitespace, strip punctuation
    normalized = text.lower().strip()
    normalized = re.sub(r'[^\w\s]', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized


def fuzzy_match_score(str1: str, str2: str) -> float:
    """
    Calculate similarity score between two strings.
    
    Args:
        str1: First string
        str2: Second string
        
    Returns:
        Similarity score between 0 and 1
    """
    return SequenceMatcher(None, normalize_text(str1), normalize_text(str2)).ratio()


def find_best_match(
    item_name: str,
    truth_items: List[Dict[str, Any]],
    name_field: str = "name",
    threshold: float = 0.8
) -> Tuple[Optional[Dict[str, Any]], float]:
    """
    Find the best matching item from truth data using fuzzy matching.
    
    Args:
        item_name: Name to search for
        truth_items: List of ground truth items
        name_field: Field name containing the item name
        threshold: Minimum similarity score to consider a match
        
    Returns:
        Tuple of (matched_item, score) or (None, 0) if no match found
    """
    best_match = None
    best_score = 0
    
    for truth_item in truth_items:
        truth_name = truth_item.get(name_field, "")
        score = fuzzy_match_score(item_name, truth_name)
        if score > best_score:
            best_score = score
            best_match = truth_item
    
    if best_score >= threshold:
        return best_match, best_score
    return None, 0


# =============================================================================
# Accuracy Calculation
# =============================================================================

def calculate_component_accuracy(
    parsed_items: List[Dict[str, Any]],
    truth_items: List[Dict[str, Any]],
    name_field: str = "name",
    threshold: float = 0.7,
    additional_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Calculate accuracy metrics for a single component type.
    
    Args:
        parsed_items: Items extracted by the model
        truth_items: Ground truth items
        name_field: Field containing the name for matching
        threshold: Fuzzy match threshold
        additional_fields: List of additional fields to check for accuracy
        
    Returns:
        Dict containing precision, recall, f1_score, and field-level accuracy
    """
    matched_count = 0
    field_matches: Dict[str, int] = {}
    matched_details: List[Dict[str, Any]] = []
    
    if additional_fields:
        for field in additional_fields:
            field_matches[field] = 0
    
    # Track which truth items have been matched to avoid double counting
    matched_truth_indices = set()
    
    for parsed_item in parsed_items:
        parsed_name = parsed_item.get(name_field, "")
        best_match = None
        best_score = 0
        best_idx = -1
        
        for idx, truth_item in enumerate(truth_items):
            if idx in matched_truth_indices:
                continue
            truth_name = truth_item.get(name_field, "")
            score = fuzzy_match_score(parsed_name, truth_name)
            if score > best_score:
                best_score = score
                best_match = truth_item
                best_idx = idx
        
        if best_score >= threshold and best_match is not None:
            matched_count += 1
            matched_truth_indices.add(best_idx)
            
            match_detail = {
                "parsed": parsed_name,
                "truth": best_match.get(name_field, ""),
                "score": best_score,
            }
            
            # Check additional fields
            if additional_fields:
                for field in additional_fields:
                    parsed_val = parsed_item.get(field)
                    truth_val = best_match.get(field)
                    
                    # Handle numeric comparison
                    if isinstance(parsed_val, (int, float)) and isinstance(truth_val, (int, float)):
                        if abs(float(parsed_val) - float(truth_val)) < 0.01:
                            field_matches[field] += 1
                            match_detail[f"{field}_match"] = True
                        else:
                            match_detail[f"{field}_match"] = False
                    # Handle None/null comparison
                    elif parsed_val is None and truth_val is None:
                        field_matches[field] += 1
                        match_detail[f"{field}_match"] = True
                    # Handle string comparison with fuzzy matching
                    elif isinstance(parsed_val, str) and isinstance(truth_val, str):
                        if fuzzy_match_score(parsed_val, truth_val) > 0.6:
                            field_matches[field] += 1
                            match_detail[f"{field}_match"] = True
                        else:
                            match_detail[f"{field}_match"] = False
                    else:
                        match_detail[f"{field}_match"] = False
            
            matched_details.append(match_detail)
    
    # Calculate metrics
    precision = matched_count / len(parsed_items) if parsed_items else 0
    recall = matched_count / len(truth_items) if truth_items else 0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    
    result = {
        "parsed_count": len(parsed_items),
        "truth_count": len(truth_items),
        "matched": matched_count,
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1_score * 100, 2),
        "matched_details": matched_details,
    }
    
    # Add field-level accuracy
    if additional_fields and matched_count > 0:
        for field in additional_fields:
            result[f"{field}_accuracy"] = round(field_matches[field] / matched_count * 100, 2)
    
    return result


def calculate_extraction_accuracy(
    parsed_result: Dict[str, Any],
    ground_truth: Dict[str, Any],
    schema_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Calculate comprehensive accuracy metrics for structured extraction.
    
    Args:
        parsed_result: The extracted data from the model
        ground_truth: The ground truth data
        schema_config: Optional configuration for schema-aware evaluation
            Example:
            {
                "components": {
                    "items": {"name_field": "name", "weight": 0.5, "additional_fields": ["price", "description"]},
                    "categories": {"name_field": "name", "weight": 0.2},
                    "modifiers": {"name_field": "name", "weight": 0.15},
                    "modifier_lists": {"name_field": "name", "weight": 0.15},
                },
                "threshold": 0.7
            }
            
    Returns:
        Dict containing metrics for each component and overall score
    """
    # Default schema config for menu extraction
    if schema_config is None:
        schema_config = {
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
    
    metrics: Dict[str, Any] = {}
    threshold = schema_config.get("threshold", 0.7)
    
    # Calculate metrics for each component
    for component_name, component_config in schema_config.get("components", {}).items():
        parsed_items = parsed_result.get(component_name, [])
        truth_items = ground_truth.get(component_name, [])
        
        name_field = component_config.get("name_field", "name")
        additional_fields = component_config.get("additional_fields", None)
        
        metrics[component_name] = calculate_component_accuracy(
            parsed_items=parsed_items,
            truth_items=truth_items,
            name_field=name_field,
            threshold=threshold,
            additional_fields=additional_fields,
        )
    
    # Calculate weighted overall score
    total_weight = 0
    weighted_f1_sum = 0
    
    for component_name, component_config in schema_config.get("components", {}).items():
        weight = component_config.get("weight", 1.0)
        if component_name in metrics:
            weighted_f1_sum += metrics[component_name]["f1_score"] * weight
            total_weight += weight
    
    overall_f1 = weighted_f1_sum / total_weight if total_weight > 0 else 0
    
    metrics["overall"] = {
        "f1_score": round(overall_f1, 2),
        "weights": {
            name: config.get("weight", 1.0)
            for name, config in schema_config.get("components", {}).items()
        },
    }
    
    return metrics


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
    
    # Component-level comparison
    component_comparison = {}
    for component in metrics_a:
        if component == "overall":
            continue
        if component in metrics_b:
            f1_a = metrics_a[component].get("f1_score", 0)
            f1_b = metrics_b[component].get("f1_score", 0)
            component_comparison[component] = {
                "score_A": f1_a,
                "score_B": f1_b,
                "winner": "A" if f1_a > f1_b + 2 else ("B" if f1_b > f1_a + 2 else "tie"),
            }
    
    return {
        "winner": winner,
        "score_A": score_a,
        "score_B": score_b,
        "difference": round(diff, 2),
        "component_comparison": component_comparison,
    }


# =============================================================================
# Image Processing Utilities
# =============================================================================

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


def get_image_mime_type(image_path: str) -> str:
    """
    Get the MIME type of an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        MIME type string (e.g., "image/png", "image/jpeg")
    """
    path = Path(image_path).suffix.lower()
    mime_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".pdf": "application/pdf",
    }
    return mime_types.get(path, "image/png")


# =============================================================================
# OCR Capability Class
# =============================================================================

class OCRCapability(BaseCapability):
    """
    OCR/Vision Extraction capability for structured data extraction from images.
    
    Supports:
    - Single model evaluation against ground truth
    - Two model comparison (head-to-head)
    - Fuzzy matching for accuracy calculation
    - Multiple extraction schemas (menu, receipt, document, custom)
    
    Test cases should have:
    - 'image_path' or 'image_paths': Path(s) to image file(s)
    - 'ground_truth': Expected extraction result
    - 'schema': (optional) Extraction schema configuration
    """
    
    name = "ocr"
    description = "Vision-based structured data extraction benchmark (OCR)"
    required_provider_features = ["vision"]
    
    # Default extraction prompt
    DEFAULT_EXTRACTION_PROMPT = """Extract all structured information from this image.
Return the result as valid JSON matching the expected schema.
Be thorough and accurate - extract ALL visible items."""
    
    def __init__(self, schema_config: Optional[Dict[str, Any]] = None):
        """
        Initialize OCR capability.
        
        Args:
            schema_config: Optional schema configuration for evaluation
        """
        self.schema_config = schema_config
    
    def build_messages(
        self,
        test_case: Dict[str, Any],
        system_prompt: str,
    ) -> List[Dict[str, Any]]:
        """
        Build messages with image content for vision API.
        
        Args:
            test_case: Dict with 'image_path' or 'image_paths' and optional 'prompt'
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
        
        # Get image paths
        image_paths = test_case.get("image_paths", [])
        if not image_paths:
            single_path = test_case.get("image_path")
            if single_path:
                image_paths = [single_path]
        
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
        
        # Add images
        for image_path in image_paths:
            try:
                base64_image = load_image_as_base64(image_path)
                mime_type = get_image_mime_type(image_path)
                
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}",
                    },
                })
            except FileNotFoundError as e:
                # Log error but continue with other images
                content.append({
                    "type": "text",
                    "text": f"[Image not found: {image_path}]",
                })
        
        messages.append({
            "role": "user",
            "content": content,
        })
        
        return messages
    
    def validate_test_case(self, test_case: Dict[str, Any]) -> bool:
        """
        Validate test case has required image and ground truth fields.
        
        Args:
            test_case: Test case data
            
        Returns:
            True if valid, False otherwise
        """
        # Must have image path(s)
        has_images = (
            "image_path" in test_case or 
            ("image_paths" in test_case and len(test_case["image_paths"]) > 0)
        )
        
        if not has_images:
            return False
        
        # Should have ground truth for evaluation
        # (not strictly required - can run without for pure inference)
        return True
    
    def get_required_fields(self) -> List[str]:
        """Required fields for OCR test cases"""
        return ["id", "image_path"]  # or image_paths
    
    def get_metrics(self) -> List[str]:
        """Metrics collected by OCR capability"""
        return [
            "precision",
            "recall",
            "f1_score",
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
            "component_comparison": comparison["component_comparison"],
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
            reasons.append(f"Model {winner} achieved higher overall F1 ({winning_score:.1f}% vs {losing_score:.1f}%)")
        
        # Add component-level insights
        for component, comp_data in comparison.get("component_comparison", {}).items():
            if comp_data["winner"] != "tie":
                reasons.append(
                    f"{component.capitalize()}: Model {comp_data['winner']} performed better "
                    f"({comp_data['score_A']:.1f}% vs {comp_data['score_B']:.1f}%)"
                )
        
        return reasons
    
    def format_result_for_display(self, result: Dict[str, Any]) -> str:
        """Format evaluation result for console display."""
        lines = []
        
        metrics = result.get("metrics", {})
        for component, data in metrics.items():
            if component == "overall":
                continue
            if isinstance(data, dict):
                lines.append(
                    f"  {component.upper():15} | "
                    f"P: {data.get('precision', 0):5.1f}% | "
                    f"R: {data.get('recall', 0):5.1f}% | "
                    f"F1: {data.get('f1_score', 0):5.1f}%"
                )
        
        overall = metrics.get("overall", {})
        lines.append(f"  {'OVERALL':15} | F1: {overall.get('f1_score', 0):5.1f}%")
        
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
