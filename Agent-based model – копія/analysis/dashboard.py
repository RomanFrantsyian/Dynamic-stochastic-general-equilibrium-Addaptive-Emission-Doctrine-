"""
Comprehensive multi-scenario dashboard for AED model results.

Generates publication-quality figures comparing all 4 scenarios
across key macroeconomic, monetary, distributional, and innovation metrics.
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'lines.linewidth': 1.8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

SCENARIO_STYLES = {
    'baseline':    {'color': '#2c3e50', 'ls': '-',  'label': 'Baseline'},
    'aed_pillar1': {'color': '#e67e22', 'ls': '--', 'label': 'AED Pillar 1'},
    'aed_full':    {'color': '#27ae60', 'ls': '-',  'label': 'AED Full'},
    'aed_gradual': {'color': '#8e44ad', 'ls': '-.', 'label': 'AED Gradual'},
}


def _fmt_millions(x, _):
    """Axis formatter: display as millions."""
    return f'{x / 1e6:.0f}M'


def _fmt_billions(x, _):
    """Axis formatter: display as billions."""
    return f'{x / 1e9:.1f}B'


def _fmt_auto(ax, series_list):
    """Pick M or B formatter depending on data magnitude."""
    peak = max(s.max() for s in series_list if len(s))
    if peak >= 1e9:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_billions))
    elif peak >= 1e5:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_millions))


def _plot_metric(ax, datasets: Dict[str, pd.DataFrame], col: str,
                 title: str, ylabel: str = '', auto_fmt: bool = True):
    """Plot a single metric across all scenarios."""
    series_list = []
    for scenario, df in datasets.items():
        if col not in df.columns:
            continue
        s = SCENARIO_STYLES[scenario]
        ax.plot(df.index, df[col], color=s['color'], ls=s['ls'], label=s['label'])
        series_list.append(df[col])
    ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    if auto_fmt and series_list:
        _fmt_auto(ax, series_list)
    ax.legend(loc='best')


def run_scenarios(periods: int = 50, seed: int = 42) -> Dict[str, pd.DataFrame]:
    """Run all 4 scenarios and return DataFrames."""
    import sys, os
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from model import AEDModel

    datasets = {}
    scenarios = ['baseline', 'aed_pillar1', 'aed_full', 'aed_gradual']

    for sc in scenarios:
        print(f'  Running {sc} ({periods} steps)...')
        model = AEDModel(scenario=sc, seed=seed)
        model.run_for(periods)
        datasets[sc] = model.datacollector.get_model_vars_dataframe()

    return datasets


def generate_dashboard(datasets: Dict[str, pd.DataFrame],
                       output_dir: str = 'output',
                       periods: int = 50):
    """Generate the full comparison dashboard."""
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    # ═══════════════════════════════════════════════════════════════════════
    # FIGURE 1 — Macroeconomic Overview (2×3)
    # ═══════════════════════════════════════════════════════════════════════
    fig1, axes1 = plt.subplots(2, 3, figsize=(18, 10))
    fig1.suptitle('Figure 1 — Macroeconomic Overview', fontsize=14, fontweight='bold', y=0.98)

    _plot_metric(axes1[0, 0], datasets, 'GDP',
                 'GDP (Output × Price)', 'Monetary units')
    _plot_metric(axes1[0, 1], datasets, 'TotalOutput',
                 'Aggregate Output (units)', 'Units', auto_fmt=False)
    _plot_metric(axes1[0, 2], datasets, 'AveragePriceLevel',
                 'Average Price Level', 'Price', auto_fmt=False)
    _plot_metric(axes1[1, 0], datasets, 'TotalRevenue',
                 'Total Firm Revenue', 'Monetary units')
    _plot_metric(axes1[1, 1], datasets, 'AggregateConsumption',
                 'Aggregate Consumption Budget', 'Monetary units')
    _plot_metric(axes1[1, 2], datasets, 'Velocity',
                 'Velocity of Money', 'V', auto_fmt=False)

    for ax in axes1[1, :]:
        ax.set_xlabel('Step')

    fig1.tight_layout(rect=[0, 0, 1, 0.96])
    fig1.savefig(out / 'fig1_macro_overview.png', dpi=150, bbox_inches='tight')
    print(f'  Saved fig1_macro_overview.png')

    # ═══════════════════════════════════════════════════════════════════════
    # FIGURE 2 — Monetary & Debt Dynamics (2×3)
    # ═══════════════════════════════════════════════════════════════════════
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
    fig2.suptitle('Figure 2 — Monetary & Debt Dynamics', fontsize=14, fontweight='bold', y=0.98)

    _plot_metric(axes2[0, 0], datasets, 'MoneySupply',
                 'Money Supply (M)', 'Monetary units')
    _plot_metric(axes2[0, 1], datasets, 'EmissionVolume',
                 'Cumulative Emission', 'Monetary units')
    _plot_metric(axes2[0, 2], datasets, 'PeriodEmission',
                 'Per-Period Emission', 'Monetary units')
    _plot_metric(axes2[1, 0], datasets, 'TotalFirmDebt',
                 'Total Firm Debt', 'Monetary units')
    _plot_metric(axes2[1, 1], datasets, 'DebtAnnihilated',
                 'Cumulative Debt Annihilated', 'Monetary units')
    _plot_metric(axes2[1, 2], datasets, 'DebtToGDP',
                 'Debt-to-GDP Ratio', 'Ratio', auto_fmt=False)

    for ax in axes2[1, :]:
        ax.set_xlabel('Step')

    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    fig2.savefig(out / 'fig2_monetary_debt.png', dpi=150, bbox_inches='tight')
    print(f'  Saved fig2_monetary_debt.png')

    # ═══════════════════════════════════════════════════════════════════════
    # FIGURE 3 — Distribution & Inequality (2×2)
    # ═══════════════════════════════════════════════════════════════════════
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
    fig3.suptitle('Figure 3 — Distribution & Inequality', fontsize=14, fontweight='bold', y=0.98)

    _plot_metric(axes3[0, 0], datasets, 'Gini',
                 'Gini Coefficient', 'Gini', auto_fmt=False)
    _plot_metric(axes3[0, 1], datasets, 'MedianWealth',
                 'Median Household Wealth', 'Monetary units')
    _plot_metric(axes3[1, 0], datasets, 'MeanWealth',
                 'Mean Household Wealth', 'Monetary units')
    _plot_metric(axes3[1, 1], datasets, 'TotalHouseholdMoney',
                 'Total Household Money Holdings', 'Monetary units')

    for ax in axes3[1, :]:
        ax.set_xlabel('Step')

    fig3.tight_layout(rect=[0, 0, 1, 0.96])
    fig3.savefig(out / 'fig3_distribution.png', dpi=150, bbox_inches='tight')
    print(f'  Saved fig3_distribution.png')

    # ═══════════════════════════════════════════════════════════════════════
    # FIGURE 4 — Banking & Innovation (2×3)
    # ═══════════════════════════════════════════════════════════════════════
    fig4, axes4 = plt.subplots(2, 3, figsize=(18, 10))
    fig4.suptitle('Figure 4 — Banking Sector & Innovation', fontsize=14, fontweight='bold', y=0.98)

    _plot_metric(axes4[0, 0], datasets, 'TotalBankReserves',
                 'Total Bank Reserves', 'Monetary units')
    _plot_metric(axes4[0, 1], datasets, 'TotalBankCredit',
                 'Total Bank Credit', 'Monetary units')
    _plot_metric(axes4[0, 2], datasets, 'SystemNPLRatio',
                 'System NPL Ratio', 'Ratio', auto_fmt=False)
    _plot_metric(axes4[1, 0], datasets, 'TotalPatents',
                 'Cumulative Patents', 'Count', auto_fmt=False)
    _plot_metric(axes4[1, 1], datasets, 'AverageTechLevel',
                 'Average Technology Level', 'Level', auto_fmt=False)

    # Government panel
    _plot_metric(axes4[1, 2], datasets, 'GovRevenue',
                 'Cumulative Government Revenue', 'Monetary units')

    for ax in axes4[1, :]:
        ax.set_xlabel('Step')

    fig4.tight_layout(rect=[0, 0, 1, 0.96])
    fig4.savefig(out / 'fig4_banking_innovation.png', dpi=150, bbox_inches='tight')
    print(f'  Saved fig4_banking_innovation.png')

    # ═══════════════════════════════════════════════════════════════════════
    # FIGURE 5 — Summary Table (final-period values)
    # ═══════════════════════════════════════════════════════════════════════
    summary_cols = [
        'GDP', 'TotalOutput', 'MoneySupply', 'EmissionVolume',
        'TotalFirmDebt', 'DebtAnnihilated', 'DebtToGDP',
        'Gini', 'MedianWealth', 'MeanWealth',
        'SystemNPLRatio', 'TotalPatents', 'AverageTechLevel',
        'GovRevenue',
    ]

    summary_rows = {}
    for sc, df in datasets.items():
        row = {}
        for col in summary_cols:
            if col in df.columns:
                row[col] = df[col].iloc[-1]
        summary_rows[SCENARIO_STYLES[sc]['label']] = row

    summary_df = pd.DataFrame(summary_rows).T

    fig5, ax5 = plt.subplots(figsize=(18, 5))
    ax5.axis('off')
    fig5.suptitle(f'Figure 5 — Final-Period Summary (Step {periods})',
                  fontsize=14, fontweight='bold', y=0.95)

    # Format numbers for display
    cell_text = []
    for _, row in summary_df.iterrows():
        formatted = []
        for col in summary_df.columns:
            val = row[col]
            if col in ('Gini', 'DebtToGDP', 'SystemNPLRatio'):
                formatted.append(f'{val:.4f}')
            elif col == 'AverageTechLevel':
                formatted.append(f'{val:.3f}')
            elif col in ('TotalPatents', 'TotalOutput'):
                formatted.append(f'{val:,.0f}')
            elif val >= 1e6:
                formatted.append(f'{val / 1e6:,.1f}M')
            else:
                formatted.append(f'{val:,.0f}')
        cell_text.append(formatted)

    # Shortened column labels
    col_labels = [
        'GDP', 'Output', 'Money\nSupply', 'Cumul.\nEmission',
        'Firm\nDebt', 'Debt\nAnnih.', 'Debt/\nGDP',
        'Gini', 'Median\nWealth', 'Mean\nWealth',
        'NPL\nRatio', 'Patents', 'Tech\nLevel',
        'Gov\nRevenue',
    ]

    colors = [SCENARIO_STYLES[sc]['color'] for sc in datasets]
    table = ax5.table(
        cellText=cell_text,
        rowLabels=list(summary_df.index),
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.6)

    # Color row labels
    for i, c in enumerate(colors):
        table[i + 1, -1].set_text_props(color=c, fontweight='bold')

    # Header styling
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#ecf0f1')
        table[0, j].set_text_props(fontweight='bold')

    fig5.tight_layout(rect=[0, 0, 1, 0.92])
    fig5.savefig(out / 'fig5_summary_table.png', dpi=150, bbox_inches='tight')
    print(f'  Saved fig5_summary_table.png')

    plt.close('all')


def main(periods: int = 50, seed: int = 42):
    """Run all scenarios and generate dashboard."""
    import logging
    logging.disable(logging.INFO)

    print(f'Running 4 scenarios for {periods} steps (seed={seed})...')
    datasets = run_scenarios(periods=periods, seed=seed)
    print('Generating dashboard figures...')
    generate_dashboard(datasets, periods=periods)
    print('Done. All figures saved to output/')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate AED comparison dashboard')
    parser.add_argument('--periods', type=int, default=50, help='Simulation steps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    main(periods=args.periods, seed=args.seed)
