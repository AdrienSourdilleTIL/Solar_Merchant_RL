"""Baseline policies for Solar Merchant RL.

This module provides rule-based baseline strategies for benchmarking
against the RL agent. Each policy takes a raw observation vector and
returns an action array compatible with SolarMerchantEnv.
"""

from .baseline_policies import aggressive_policy, conservative_policy

__all__ = ["aggressive_policy", "conservative_policy"]
