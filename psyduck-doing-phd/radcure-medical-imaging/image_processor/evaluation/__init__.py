"""Evaluation utilities for medical image segmentation."""

from image_processor.evaluation.evaluator import SegmentationEvaluator
from image_processor.evaluation.dsc_agg import (
    binary_confusion_counts,
    dsc_agg_from_totals,
)

__all__ = [
    "SegmentationEvaluator",
    "binary_confusion_counts",
    "dsc_agg_from_totals",
]
