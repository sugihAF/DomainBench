"""
OCR Capability - Document/Image extraction benchmark
"""

from domainbench.capabilities.ocr.ocr import (
    OCRCapability,
    calculate_extraction_accuracy,
    compare_model_results,
    get_schema_config,
    MENU_EXTRACTION_SCHEMA,
    RECEIPT_EXTRACTION_SCHEMA,
    DOCUMENT_EXTRACTION_SCHEMA,
    SchemaScoreConfig,
    score_with_schema,
    normalize_ground_truth,
    infer_schema_from_ground_truth,
    convert_pdf_to_images,
    load_image_as_base64,
    load_file_as_images,
    get_image_mime_type,
    is_pdf_file,
)

__all__ = [
    "OCRCapability",
    "calculate_extraction_accuracy",
    "compare_model_results",
    "get_schema_config",
    "MENU_EXTRACTION_SCHEMA",
    "RECEIPT_EXTRACTION_SCHEMA",
    "DOCUMENT_EXTRACTION_SCHEMA",
    "SchemaScoreConfig",
    "score_with_schema",
    "normalize_ground_truth",
    "infer_schema_from_ground_truth",
    "convert_pdf_to_images",
    "load_image_as_base64",
    "load_file_as_images",
    "get_image_mime_type",
    "is_pdf_file",
]
