"""Cheap validation gates run before any GPU training."""

from training.validation.preflight import PreflightReport, preflight

__all__ = ["PreflightReport", "preflight"]
