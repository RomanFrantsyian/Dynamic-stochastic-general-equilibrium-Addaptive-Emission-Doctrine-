"""
KPI computation functions.
"""
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from model import AEDModel


def compute_gini(model: 'AEDModel') -> float:
    """
    Compute Gini coefficient for wealth inequality.

    Args:
        model: AEDModel instance

    Returns:
        Gini coefficient (0 = perfect equality, 1 = perfect inequality)
    """
    wealths = [h.wealth for h in model.households]
    if not wealths:
        return 0.0

    wealths = sorted(wealths)
    n = len(wealths)

    if sum(wealths) == 0:
        return 0.0

    cumsum = 0
    for i, w in enumerate(wealths):
        cumsum += (2 * (i + 1) - n - 1) * w

    gini = cumsum / (n * sum(wealths))
    return gini


def compute_dar_concentration(dar_registry: dict) -> float:
    """
    Compute concentration of DAR scores (Herfindahl index).

    Args:
        dar_registry: Dictionary of firm_id -> dar_score

    Returns:
        Herfindahl index (0 = distributed, 1 = concentrated)
    """
    if not dar_registry:
        return 0.0

    total_dar = sum(dar_registry.values())
    if total_dar == 0:
        return 0.0

    shares = [dar / total_dar for dar in dar_registry.values()]
    herfindahl = sum(s**2 for s in shares)

    return herfindahl
