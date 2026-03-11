"""
Echo-royalty calculation for innovators.
"""
from typing import Dict


def compute_echo_royalty(dar_score: float,
                         total_emission: float,
                         dar_registry: Dict[int, float]) -> float:
    """
    Compute echo-royalty for an innovator firm.

    Formula: Echo_i = (DAR_i / Σ DAR) * E_total * echo_share

    Where:
    - DAR_i: This firm's DAR score
    - Σ DAR: Total DAR across all firms
    - E_total: Total emission this period
    - echo_share: Share allocated to innovators (e.g., 0.30)

    Args:
        dar_score: This firm's DAR score
        total_emission: Total emission volume
        dar_registry: All firms' DAR scores

    Returns:
        Echo-royalty amount for this firm
    """
    if dar_score <= 0 or total_emission <= 0:
        return 0.0

    # Sum all DAR scores
    total_dar = sum(dar_registry.values())

    if total_dar <= 0:
        return 0.0

    # Echo share (typically 20% of emission goes to innovators)
    echo_share = 0.20

    # Proportional allocation
    echo_royalty = (dar_score / total_dar) * total_emission * echo_share

    return echo_royalty
