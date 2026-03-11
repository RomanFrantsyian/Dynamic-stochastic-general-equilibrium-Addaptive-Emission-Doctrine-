"""
Test emission mechanisms - tests/test_mechanisms/test_emission.py
"""
from mechanisms.emission import compute_deflationary_vacuum, compute_emission_volume


def test_deflationary_vacuum_positive():
    """Test deflationary vacuum with positive debt annihilation."""
    vacuum = compute_deflationary_vacuum(
        debt_annihilated=1_000_000,
        price_target=1.0,
        inflation_buffer=1.02
    )

    assert vacuum > 0
    expected = 1_000_000 / (1.0 * 1.02)
    assert abs(vacuum - expected) < 0.01


def test_deflationary_vacuum_zero():
    """Test deflationary vacuum with zero debt."""
    vacuum = compute_deflationary_vacuum(
        debt_annihilated=0,
        price_target=1.0,
        inflation_buffer=1.02
    )

    assert vacuum == 0.0


def test_emission_volume():
    """Test emission volume calculation."""
    emission = compute_emission_volume(
        deflationary_vacuum=1_000_000,
        emission_coverage_ratio=0.75
    )

    assert emission == 750_000.0


def test_emission_volume_zero_vacuum():
    """Test emission with zero vacuum."""
    emission = compute_emission_volume(
        deflationary_vacuum=0,
        emission_coverage_ratio=0.75
    )

    assert emission == 0.0
