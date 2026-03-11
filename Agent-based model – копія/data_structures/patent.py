"""
Patent data structure.
"""
from dataclasses import dataclass


@dataclass
class Patent:
    """Patent record."""
    patent_id: int
    technology_level: float
    cost_reduction_factor: float
    royalty_rate: float
    is_open: bool = False
