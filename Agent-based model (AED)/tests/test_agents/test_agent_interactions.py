"""
Test agent interactions - tests/test_agents/test_agent_interactions.py
"""
import pytest
from model import AEDModel


def test_firm_production():
    """Test that firms can produce goods."""
    model = AEDModel(scenario='baseline', seed=42)

    # Give a firm employees
    firm = model.innovator_firms[0]
    firm.employees = 10

    # Produce
    units = firm.produce()
    assert units == int(10 * firm.technology_level * firm.productivity_factor)
    assert firm['goods'] >= units


def test_household_consumption():
    """Test household can buy and consume goods."""
    model = AEDModel(scenario='baseline', seed=42)

    household = model.households[0]
    household.create('money', 10000)
    household.compute_budget()

    # Budget is based on total money holdings (wealth * consumption_propensity)
    expected_budget = household.wealth * household.consumption_propensity
    assert household.consumption_budget == expected_budget


def test_central_bank_emission_baseline():
    """Test no emission in baseline scenario."""
    model = AEDModel(scenario='baseline', seed=42)

    model.central_bank.compute_and_execute_emission()

    assert model.central_bank.period_emission == 0.0


def test_goods_transfer_between_agents():
    """Test goods transfer between two agents."""
    model = AEDModel(scenario='baseline', seed=42)

    firm = model.innovator_firms[0]
    household = model.households[0]

    firm.create('money', 1000)
    initial_firm_money = firm['money']

    firm.give(household, 'money', 500)

    assert firm['money'] == initial_firm_money - 500
    assert household['money'] >= 500
