"""
Emission computation algorithms.
"""


def compute_deflationary_vacuum(debt_annihilated: float,
                                price_target: float,
                                inflation_buffer: float) -> float:
    """
    Compute deflationary vacuum from debt annihilation.

    Formula: V = D / (P_target * k)

    Where:
    - D: Debt annihilated
    - P_target: Price stability target
    - k: Inflation buffer (safety margin)

    Args:
        debt_annihilated: Total debt written off this period
        price_target: Target price level
        inflation_buffer: Safety multiplier (e.g., 1.02 for 2% buffer)

    Returns:
        Deflationary vacuum amount
    """
    if debt_annihilated <= 0:
        return 0.0

    vacuum = debt_annihilated / (price_target * inflation_buffer)
    return vacuum


def compute_emission_volume(deflationary_vacuum: float,
                            emission_coverage_ratio: float,
                            money_supply: float = 0.0,
                            qe_rate: float = 0.02,
                            mode: str = 'AED') -> float:
    """
    Compute emission volume from deflationary vacuum.

    Formula: E = V * α

    Where:
    - V: Deflationary vacuum
    - α: Emission coverage ratio (0 to 1)

    Args:
        deflationary_vacuum: Calculated vacuum
        emission_coverage_ratio: Coverage ratio (typically 0.75)

    Returns:
        Emission volume
    """

    if mode == 'BASELINE':
        if money_supply <= 0:
            return 0.0
        return money_supply * qe_rate

    if deflationary_vacuum <= 0:
        return 0.0

    emission = deflationary_vacuum * emission_coverage_ratio
    return emission
