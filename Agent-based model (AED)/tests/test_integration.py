"""
Integration tests - tests/test_integration.py
"""
import pytest
from model import AEDModel


def test_model_initialization():
    """Test model initializes without errors."""
    model = AEDModel(scenario='baseline', seed=42)

    assert model.central_bank is not None
    assert model.government is not None
    assert len(model.commercial_banks) == 5
    assert len(model.innovator_firms) == 20
    assert len(model.implementer_firms) == 100
    assert len(model.households) == 500
    assert len(model.investors) == 20


def test_single_step():
    """Test model executes single step."""
    model = AEDModel(scenario='baseline', seed=42)

    model.step()

    assert model.steps == 1
    assert len(model.datacollector.get_model_vars_dataframe()) == 1


def test_run_for():
    """Test model.run_for() execution."""
    model = AEDModel(scenario='baseline', seed=42)

    model.run_for(100)

    assert model.steps == 100

    # Verify data collection
    model_data = model.datacollector.get_model_vars_dataframe()
    assert len(model_data) == 100
    assert 'MoneySupply' in model_data.columns


def test_scenario_differences():
    """Test different scenarios produce different results."""
    model_baseline = AEDModel(scenario='baseline', seed=42)
    model_aed = AEDModel(scenario='aed_full', seed=42)

    model_baseline.run_for(50)
    model_aed.run_for(50)

    # AED should have emission, baseline should not
    baseline_emission = model_baseline.central_bank.emission_volume
    aed_emission = model_aed.central_bank.emission_volume

    assert baseline_emission == 0.0
    assert aed_emission > 0.0


def test_gradual_transition_phases():
    """Test gradual scenario transitions through phases."""
    model = AEDModel(scenario='aed_gradual', seed=42)

    # Phase 0: BASELINE (steps 0-4)
    model.run_for(5)
    assert model.scenario_mode == 'BASELINE'
    assert model.central_bank.emission_volume == 0.0

    # Phase 1: AED_PILLAR1 (steps 5-9)
    model.run_for(5)
    assert model.scenario_mode == 'AED_PILLAR1'
    assert model.central_bank.emission_coverage_ratio == 0.75
    assert model.government.tax_rate == 0.35

    # Phase 2: AED_FULL @ 15% tax (steps 10-14)
    model.run_for(5)
    assert model.scenario_mode == 'AED_FULL'
    assert model.government.tax_rate == 0.15

    # Phase 3: AED_FULL @ 0% tax (steps 15+)
    model.run_for(5)
    assert model.scenario_mode == 'AED_FULL'
    assert model.government.tax_rate == 0.0
