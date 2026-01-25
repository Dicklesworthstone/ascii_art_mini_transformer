"""
Python-native inference utilities for ASCII Art Mini Transformer.

This package is intended for:
- development/debugging
- golden test generation (Python <-> Rust cross-validation)
- interactive exploration

Torch is an optional dependency for *importing* this package; most symbols that
require torch will raise a clear error if torch is not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .constraints import ConstrainedDecoder
from .sampler import Sampler, TopKSampler, TopPSampler

if TYPE_CHECKING:  # pragma: no cover
    from .generate import (
        generate,
        generate_greedy,
        generate_sample,
        generate_golden_tests,
    )

try:  # Optional torch dependency for the generation helpers.
    from .generate import (
        generate,
        generate_greedy,
        generate_sample,
        generate_golden_tests,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    if exc.name != "torch":
        raise

    torch_exc = exc

    def _torch_required(*_args, **_kwargs):
        raise ModuleNotFoundError(
            "torch is required for python.inference.generate* helpers. "
            "Install it (see python/requirements.txt) or import lower-level "
            "helpers that don't require torch."
        ) from torch_exc

    generate = _torch_required  # type: ignore[assignment]
    generate_greedy = _torch_required  # type: ignore[assignment]
    generate_sample = _torch_required  # type: ignore[assignment]
    generate_golden_tests = _torch_required  # type: ignore[assignment]


__all__ = [
    "ConstrainedDecoder",
    "Sampler",
    "TopKSampler",
    "TopPSampler",
    "generate",
    "generate_greedy",
    "generate_sample",
    "generate_golden_tests",
]
