"""Evaluation module for Solar Merchant RL.

Provides tools for evaluating trading policies against the solar merchant
environment and comparing their performance.
"""

from .evaluate import evaluate_policy

__all__ = ["evaluate_policy"]
