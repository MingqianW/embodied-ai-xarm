"""Compatibility imports for model contracts now owned by evaluation.common."""

from evaluation.common.models import ModelSpec
from evaluation.common.models import load_model_spec
from evaluation.common.models import validate_abc_comparison_specs
from evaluation.common.models import validate_model_spec
from evaluation.common.models import validate_training_config_asset

__all__ = [
    "ModelSpec",
    "load_model_spec",
    "validate_abc_comparison_specs",
    "validate_model_spec",
    "validate_training_config_asset",
]

