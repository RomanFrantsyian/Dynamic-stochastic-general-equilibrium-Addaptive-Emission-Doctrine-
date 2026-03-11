"""
Test verification mechanisms - tests/test_mechanisms/test_verification.py
"""
from mechanisms.verification import verify_deflation_claim


class MockFirm:
    """Mock firm for testing verification."""
    def __init__(self, base_cost, current_cost):
        self.base_production_cost = base_cost
        self.production_cost = current_cost


def test_valid_deflation_claim():
    """Test legitimate deflation claim."""
    firm = MockFirm(base_cost=100.0, current_cost=90.0)

    result = verify_deflation_claim(
        borrower=firm,
        declared_deflation=10.0,
        fraud_threshold=0.05
    )

    assert result is True


def test_invalid_no_reduction():
    """Test claim with no actual reduction."""
    firm = MockFirm(base_cost=100.0, current_cost=100.0)

    result = verify_deflation_claim(
        borrower=firm,
        declared_deflation=10.0,
        fraud_threshold=0.05
    )

    assert result is False


def test_invalid_excessive_deviation():
    """Test claim with excessive deviation from actual."""
    firm = MockFirm(base_cost=100.0, current_cost=95.0)

    # Declared 10, actual is 5 -> deviation = |10-5|/5 = 1.0 > 0.05
    result = verify_deflation_claim(
        borrower=firm,
        declared_deflation=10.0,
        fraud_threshold=0.05
    )

    assert result is False


def test_invalid_zero_declared():
    """Test claim with zero declared deflation."""
    firm = MockFirm(base_cost=100.0, current_cost=90.0)

    result = verify_deflation_claim(
        borrower=firm,
        declared_deflation=0.0,
        fraud_threshold=0.05
    )

    assert result is False
