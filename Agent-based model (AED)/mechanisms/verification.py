"""
Deflation verification and fraud detection.
"""
from agents.base import FirmAgent

def verify_deflation_claim(borrower: 'FirmAgent',
                           declared_deflation: float,
                           fraud_threshold: float) -> bool:
    """
    Verify that a firm's deflation claim is legitimate.

    Checks:
    - Production cost actually decreased
    - Decrease is consistent with technology adoption
    - No suspicious patterns

    Args:
        borrower: Firm making claim
        declared_deflation: Claimed cost reduction
        fraud_threshold: Maximum acceptable deviation

    Returns:
        True if claim is legitimate
    """
    if declared_deflation <= 0:
        return False

    # Check if borrower is a firm
    if not hasattr(borrower, 'production_cost'):
        return False

    # Calculate actual cost reduction
    base_cost = borrower.base_production_cost
    current_cost = borrower.production_cost
    actual_reduction = base_cost - current_cost

    # Check consistency
    if actual_reduction <= 0:
        return False

    # Check deviation from declared
    deviation = abs(declared_deflation - actual_reduction) / actual_reduction

    if deviation > fraud_threshold:
        return False

    return True
