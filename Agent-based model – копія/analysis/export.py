"""
Data export utilities.
"""
from typing import TYPE_CHECKING
import pandas as pd
from pathlib import Path

if TYPE_CHECKING:
    from model import AEDModel


def export_simulation_data(model: 'AEDModel', output_dir: str = 'output', prefix: str = 'sim'):
    """
    Export simulation data to parquet files.

    Args:
        model: AEDModel instance
        output_dir: Output directory
        prefix: Filename prefix
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Export model data
    model_data = model.datacollector.get_model_vars_dataframe()
    model_data.to_parquet(output_path / f'{prefix}_model.parquet')

    # Export agent data
    agent_data = model.datacollector.get_agent_vars_dataframe()
    agent_data.to_parquet(output_path / f'{prefix}_agents.parquet')

    print(f"Data exported to {output_dir}/")
    return model_data, agent_data


def load_simulation_data(filepath: str) -> pd.DataFrame:
    """
    Load simulation data from parquet.

    Args:
        filepath: Path to parquet file

    Returns:
        DataFrame with simulation data
    """
    return pd.read_parquet(filepath)
