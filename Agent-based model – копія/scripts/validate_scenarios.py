"""
Scenario validation utilities.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import AEDModel


def validate_scenario(scenario: str, num_steps: int = 20, seed: int = 42):
    """
    Validate a scenario runs without errors.

    Args:
        scenario: Scenario name
        num_steps: Number of steps to run
        seed: Random seed
    """
    print(f"Validating {scenario} scenario...")

    try:
        model = AEDModel(scenario=scenario, seed=seed)
        model.run_for(num_steps)

        # Get results
        data = model.datacollector.get_model_vars_dataframe()

        print(f"  Steps completed: {model.steps}")
        print(f"  Final Money Supply: {model.central_bank.money_supply:,.0f}")
        print(f"  Total Emission: {model.central_bank.emission_volume:,.0f}")
        print(f"  Total Debt Annihilated: {model.central_bank.total_debt_annihilated:,.0f}")
        print(f"  Scenario Mode: {model.scenario_mode}")
        print(f"  Data Points: {len(data)}")
        print(f"  PASSED")
        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def validate_all_scenarios():
    """Validate all scenarios."""
    scenarios = ['baseline', 'aed_pillar1', 'aed_full', 'aed_gradual']
    results = {}

    for scenario in scenarios:
        results[scenario] = validate_scenario(scenario)
        print()

    # Summary
    print("=" * 50)
    print("Validation Summary:")
    for scenario, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {scenario}: {status}")

    all_passed = all(results.values())
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


if __name__ == '__main__':
    validate_all_scenarios()
