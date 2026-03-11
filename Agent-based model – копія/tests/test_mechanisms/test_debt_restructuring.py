"""
Test debt restructuring - tests/test_mechanisms/test_debt_restructuring.py
"""
from mechanisms.debt_restructuring import validate_restructuring_report


def test_valid_report():
    """Test valid restructuring report."""
    report = {
        'debt_annihilated': 500_000,
        'npl_count': 5,
        'deflation_verified': True
    }

    assert validate_restructuring_report(report, fraud_threshold=0.05) is True


def test_invalid_report_no_deflation():
    """Test report without deflation verification."""
    report = {
        'debt_annihilated': 500_000,
        'npl_count': 5,
        'deflation_verified': False
    }

    assert validate_restructuring_report(report, fraud_threshold=0.05) is False


def test_invalid_report_zero_debt():
    """Test report with zero debt."""
    report = {
        'debt_annihilated': 0,
        'npl_count': 0,
        'deflation_verified': True
    }

    assert validate_restructuring_report(report, fraud_threshold=0.05) is False


def test_invalid_report_suspicious_npl():
    """Test report with suspiciously large NPLs."""
    report = {
        'debt_annihilated': 50_000_000,
        'npl_count': 1,
        'deflation_verified': True
    }

    # Average NPL = 50M > 10M threshold
    assert validate_restructuring_report(report, fraud_threshold=0.05) is False
