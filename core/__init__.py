"""Core contract and orchestration for the lead-optimization demo."""

from .contract import is_valid, normalize, propose, score

__all__ = ["is_valid", "normalize", "propose", "score"]
