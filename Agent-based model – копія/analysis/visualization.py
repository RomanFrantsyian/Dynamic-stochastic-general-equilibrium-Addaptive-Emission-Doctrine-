"""
Plotting utilities for AED simulation results.
"""
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional


def plot_scenario_comparison(data_baseline: pd.DataFrame,
                              data_aed: pd.DataFrame,
                              output_file: Optional[str] = None):
    """
    Plot comparison of baseline vs AED scenarios.

    Args:
        data_baseline: Model data from baseline scenario
        data_aed: Model data from AED scenario
        output_file: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Money Supply
    axes[0, 0].plot(data_baseline.index, data_baseline['MoneySupply'], label='Baseline')
    axes[0, 0].plot(data_aed.index, data_aed['MoneySupply'], label='AED')
    axes[0, 0].set_title('Money Supply')
    axes[0, 0].legend()

    # Firm Debt
    axes[0, 1].plot(data_baseline.index, data_baseline['TotalFirmDebt'], label='Baseline')
    axes[0, 1].plot(data_aed.index, data_aed['TotalFirmDebt'], label='AED')
    axes[0, 1].set_title('Total Firm Debt')
    axes[0, 1].legend()

    # Gini Coefficient
    axes[1, 0].plot(data_baseline.index, data_baseline['Gini'], label='Baseline')
    axes[1, 0].plot(data_aed.index, data_aed['Gini'], label='AED')
    axes[1, 0].set_title('Wealth Inequality (Gini)')
    axes[1, 0].legend()

    # Emission Volume
    axes[1, 1].plot(data_aed.index, data_aed['EmissionVolume'], label='AED Emission')
    axes[1, 1].set_title('Cumulative Emission (AED only)')
    axes[1, 1].legend()

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300)

    plt.show()


def plot_time_series(data: pd.DataFrame, columns: list,
                     title: str = 'Simulation Results',
                     output_file: Optional[str] = None):
    """
    Plot time series of selected metrics.

    Args:
        data: Model data DataFrame
        columns: Column names to plot
        title: Plot title
        output_file: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for col in columns:
        if col in data.columns:
            ax.plot(data.index, data[col], label=col)

    ax.set_title(title)
    ax.set_xlabel('Step')
    ax.legend()
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300)

    plt.show()
