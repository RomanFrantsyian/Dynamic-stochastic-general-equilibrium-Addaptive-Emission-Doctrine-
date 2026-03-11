"""
Debt restructuring logic.
"""


def validate_restructuring_report(report: dict, fraud_threshold: float) -> bool:
    """
    Validate debt restructuring report from bank.

    Checks for:
    - Reasonable NPL count
    - Deflation verification flag
    - Fraud detection heuristics

    Args:
        report: Restructuring report from bank
        fraud_threshold: Maximum acceptable fraud ratio

    Returns:
        True if report is valid
    """
    debt_annihilated = report.get('debt_annihilated', 0.0)
    npl_count = report.get('npl_count', 0)
    deflation_verified = report.get('deflation_verified', False)

    # Check basic validity
    if debt_annihilated <= 0 or npl_count <= 0:
        return False

    # Must have deflation verification
    if not deflation_verified:
        return False

    # Check average NPL size (fraud detection)
    avg_npl = debt_annihilated / npl_count

    # Reject if average NPL is suspiciously large
    # (This is a simplified heuristic)
    max_reasonable_npl = 10_000_000  # 10M
    if avg_npl > max_reasonable_npl:
        return False

    return True
