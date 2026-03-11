"""
Command-line interface for running AED simulations.
"""
import argparse
import mesa
import numpy as np
import sys
from model import AEDModel
import pandas as pd
from pathlib import Path


def run_single(scenario: str, num_periods: int = 100, seed: int = 42):
    """
    Run a single simulation.

    Args:
        scenario: Scenario name
        num_periods: Number of periods to simulate
        seed: Random seed for reproducibility
    """
    print(f"Running {scenario} scenario for {num_periods} periods (seed={seed})...")

    model = AEDModel(scenario=scenario, seed=seed)
    model.run_for(num_periods)

    # Export data
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)

    model_data = model.datacollector.get_model_vars_dataframe()
    agent_data = model.datacollector.get_agent_vars_dataframe()

    model_data.to_parquet(output_dir / f'{scenario}_model_{seed}.parquet')
    agent_data.to_parquet(output_dir / f'{scenario}_agents_{seed}.parquet')

    print(f"Simulation complete. Data saved to output/")
    print(f"Final money supply: {model.central_bank.money_supply:,.0f}")
    print(f"Total emission: {model.central_bank.emission_volume:,.0f}")

    return model_data, agent_data


def run_batch(scenario: str, num_runs: int = 100, num_periods: int = 100, base_seed: int = 42):
    """
    Run multiple simulations in parallel.

    Args:
        scenario: Scenario name
        num_runs: Number of simulation runs
        num_periods: Number of periods per run
        base_seed: Base random seed
    """
    print(f"Running {num_runs} batch simulations of {scenario} scenario...")

    # Generate random seeds (Mesa 3.4+ requirement)
    rng = np.random.default_rng(base_seed)
    rng_values = rng.integers(0, sys.maxsize, size=(num_runs,))

    params = {
        "scenario": [scenario],
    }

    results = mesa.batch_run(
        AEDModel,
        parameters=params,
        rng=rng_values.tolist(),
        max_steps=num_periods,
        number_processes=None,  # Use all CPUs
        data_collection_period=1,
        display_progress=True
    )

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Export
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    df.to_parquet(output_dir / f'{scenario}_batch_{num_runs}runs.parquet')

    print(f"Batch run complete. {num_runs} simulations saved to output/")
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run AED simulations - Mesa 3.5.0')
    parser.add_argument('--scenario', default='baseline',
                        choices=['baseline', 'aed_pillar1', 'aed_full', 'aed_gradual'],
                        help='Scenario to run')
    parser.add_argument('--periods', type=int, default=100,
                        help='Number of periods to simulate')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of runs (1 = single, >1 = batch)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    if args.runs == 1:
        run_single(args.scenario, args.periods, args.seed)
    else:
        run_batch(args.scenario, args.runs, args.periods, args.seed)
