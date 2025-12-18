"""
Visualization Module
Creates charts and plots from portfolio optimization JSON results.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from typing import Optional


# Set style for better looking plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12


def load_results(json_path: str = 'portfolio_results.json') -> dict:
    """Load results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def plot_optimal_weights(results: dict, output_path: str) -> None:
    """
    Create a horizontal bar chart of optimal portfolio weights.

    Args:
        results: Dictionary with portfolio results
        output_path: Path to save the figure
    """
    weights = results['optimal_weights']

    # Filter out near-zero weights for cleaner visualization
    significant_weights = {k: v for k, v in weights.items() if v > 0.001}

    # Sort by weight
    sorted_weights = dict(sorted(significant_weights.items(), key=lambda x: x[1], reverse=True))

    fig, ax = plt.subplots(figsize=(10, 6))

    tickers = list(sorted_weights.keys())
    values = list(sorted_weights.values())

    # Color coding by asset type
    colors = []
    for ticker in tickers:
        if ticker.endswith('.BK'):
            colors.append('#E63946')  # Thai stocks - red
        elif ticker in ['LEMB', 'VWOB', 'EMLC']:
            colors.append('#457B9D')  # Fixed income - blue
        elif ticker == 'THB=X':
            colors.append('#2A9D8F')  # FX - teal
        else:
            colors.append('#F4A261')  # Global - orange

    bars = ax.barh(tickers, values, color=colors, edgecolor='white', linewidth=0.5)

    # Add percentage labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val*100:.1f}%', va='center', fontsize=10)

    ax.set_xlabel('Weight')
    ax.set_title('Optimal Portfolio Weights\n(Mean-Variance Optimization, No Short Selling)')
    ax.set_xlim(0, max(values) * 1.2)

    # Add legend for asset types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E63946', label='Thai Stocks'),
        Patch(facecolor='#457B9D', label='Fixed Income'),
        Patch(facecolor='#2A9D8F', label='FX'),
        Patch(facecolor='#F4A261', label='Global')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    # Add metrics annotation
    metrics = results['portfolio_metrics']
    metrics_text = (
        f"Expected Return: {metrics['expected_annual_return']*100:.1f}%\n"
        f"Volatility: {metrics['expected_annual_volatility']*100:.1f}%\n"
        f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}"
    )
    ax.text(0.98, 0.02, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_backtest_performance(results: dict, output_path: str) -> None:
    """
    Create a chart showing historical backtest performance.

    Args:
        results: Dictionary with portfolio results
        output_path: Path to save the figure
    """
    backtest = results['backward_looking_backtest']
    returns_data = backtest['returns_data']
    metrics = backtest['metrics']

    dates = pd.to_datetime(returns_data['dates'])
    values = np.array(returns_data['values'])

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])

    # Plot 1: Portfolio Value
    ax1 = axes[0]
    ax1.plot(dates, values, linewidth=1.5, color='#1D3557', label='Portfolio Value')
    ax1.fill_between(dates, values, alpha=0.3, color='#457B9D')

    # Add drawdown shading
    rolling_max = pd.Series(values).expanding().max()
    drawdowns = (values - rolling_max) / rolling_max

    ax1.set_ylabel('Portfolio Value (Starting = 100)')
    ax1.set_title('Historical Backtest Performance\n(Optimal Weights Applied to Historical Data)')
    ax1.legend(loc='upper left')

    # Format x-axis
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Add metrics annotation
    metrics_text = (
        f"Total Return: {metrics['total_return_pct']:.1f}%\n"
        f"Annualized Return: {metrics['annualized_return_pct']:.1f}%\n"
        f"Annualized Volatility: {metrics['annualized_volatility_pct']:.1f}%\n"
        f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
        f"Max Drawdown: {metrics['max_drawdown_pct']:.1f}%"
    )
    ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot 2: Drawdown
    ax2 = axes[1]
    ax2.fill_between(dates, drawdowns * 100, 0, color='#E63946', alpha=0.7)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.set_title('Drawdown from Peak')
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_stress_simulation(results: dict, output_path: str) -> None:
    """
    Create a fan chart showing stress scenario simulation paths.

    Args:
        results: Dictionary with portfolio results
        output_path: Path to save the figure
    """
    stress = results['forward_looking_stress_simulation']
    sim_data = stress['simulation_returns_data']
    stats = stress['portfolio_statistics']
    config = stress['scenario_config']

    days = np.array(sim_data['trading_days'])
    paths = sim_data['percentile_paths']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Fan chart of simulation paths
    ax1 = axes[0, 0]

    # Plot percentile bands
    ax1.fill_between(days, paths['p5'], paths['p95'], alpha=0.2, color='#E63946', label='5th-95th percentile')
    ax1.fill_between(days, paths['p25'], paths['p75'], alpha=0.4, color='#E63946', label='25th-75th percentile')
    ax1.plot(days, paths['p50'], linewidth=2, color='#1D3557', label='Median (50th)')

    ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Initial Value')
    ax1.set_xlabel('Trading Days')
    ax1.set_ylabel('Portfolio Value (Starting = 100)')
    ax1.set_title(f'Stress Scenario Simulation: {config["name"]}\n(1-Year Forward Looking)')
    ax1.legend(loc='upper left')

    # Plot 2: Distribution of final returns
    ax2 = axes[0, 1]

    # Create histogram data from percentiles (approximate)
    final_values = [paths['p5'][-1], paths['p25'][-1], paths['p50'][-1], paths['p75'][-1], paths['p95'][-1]]
    final_returns = [(v - 100) for v in final_values]

    # Use stats for better histogram
    mean_ret = stats['mean_return_pct']
    std_ret = stats['std_return_pct']
    x = np.linspace(mean_ret - 3*std_ret, mean_ret + 3*std_ret, 100)
    y = (1/(std_ret * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean_ret)/std_ret)**2)

    ax2.plot(x, y, linewidth=2, color='#1D3557')
    ax2.fill_between(x, y, alpha=0.3, color='#457B9D')
    ax2.axvline(x=0, color='#E63946', linestyle='--', linewidth=2, label='Break-even')
    ax2.axvline(x=stats['var_5_pct'], color='#E63946', linestyle=':', linewidth=2, label=f'VaR 5%: {stats["var_5_pct"]:.1f}%')
    ax2.axvline(x=mean_ret, color='#2A9D8F', linestyle='-', linewidth=2, label=f'Mean: {mean_ret:.1f}%')

    ax2.set_xlabel('Portfolio Return (%)')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution of 1-Year Returns Under Stress')
    ax2.legend(loc='upper right')

    # Plot 3: Risk metrics bar chart
    ax3 = axes[1, 0]

    risk_metrics = {
        'Mean Return': stats['mean_return_pct'],
        'Median Return': stats['median_return_pct'],
        'VaR (5%)': stats['var_5_pct'],
        'CVaR (5%)': stats['cvar_5_pct'],
        'Min Return': stats['min_return_pct'],
        'Max Return': stats['max_return_pct']
    }

    colors = ['#2A9D8F' if v >= 0 else '#E63946' for v in risk_metrics.values()]
    bars = ax3.bar(risk_metrics.keys(), risk_metrics.values(), color=colors, edgecolor='white')

    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.set_ylabel('Return (%)')
    ax3.set_title('Risk Metrics Summary')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add value labels on bars
    for bar, val in zip(bars, risk_metrics.values()):
        ypos = bar.get_height() + 1 if val >= 0 else bar.get_height() - 3
        ax3.text(bar.get_x() + bar.get_width()/2, ypos, f'{val:.1f}%',
                ha='center', va='bottom' if val >= 0 else 'top', fontsize=9)

    # Plot 4: Probability metrics
    ax4 = axes[1, 1]

    prob_metrics = {
        'Negative Return': stats['prob_negative_return'],
        'Loss > 10%': stats['prob_loss_10pct'],
        'Loss > 20%': stats['prob_loss_20pct']
    }

    bars = ax4.bar(prob_metrics.keys(), prob_metrics.values(), color='#E63946', edgecolor='white')
    ax4.set_ylabel('Probability (%)')
    ax4.set_title('Loss Probability Under Stress Scenario')
    ax4.set_ylim(0, max(100, max(prob_metrics.values()) * 1.2))

    # Add value labels
    for bar, val in zip(bars, prob_metrics.values()):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

    # Add scenario config annotation
    config_text = (
        f"Affected Assets: {len(config['affected_assets'])}\n"
        f"Expected Drop: {config['expected_drop_pct']}% ± {config['drop_std_pct']}%\n"
        f"Volatility Increase: {config['volatility_increase_pct']}%"
    )
    ax4.text(0.98, 0.98, config_text, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_asset_stress_impact(results: dict, output_path: str) -> None:
    """
    Create a chart showing the impact of stress on individual assets.

    Args:
        results: Dictionary with portfolio results
        output_path: Path to save the figure
    """
    stress = results['forward_looking_stress_simulation']
    asset_stats = stress['asset_statistics']
    stress_details = stress['stress_details']
    weights = stress.get('weights_used', results['optimal_weights'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Expected returns by asset under stress
    ax1 = axes[0]

    mean_returns = asset_stats['mean_return_by_asset']
    sorted_assets = dict(sorted(mean_returns.items(), key=lambda x: x[1]))

    tickers = list(sorted_assets.keys())
    returns = list(sorted_assets.values())

    # Color by affected vs unaffected
    affected = stress['scenario_config']['affected_assets']
    colors = ['#E63946' if t in affected else '#457B9D' for t in tickers]

    bars = ax1.barh(tickers, returns, color=colors, edgecolor='white')
    ax1.axvline(x=0, color='black', linewidth=0.5)
    ax1.set_xlabel('Expected Return (%)')
    ax1.set_title('Expected 1-Year Returns by Asset Under Stress')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E63946', label='Affected by Stress'),
        Patch(facecolor='#457B9D', label='Unaffected')
    ]
    ax1.legend(handles=legend_elements, loc='lower right')

    # Plot 2: Initial shock applied to affected assets
    ax2 = axes[1]

    if stress_details:
        shock_data = {k: v['initial_shock_pct'] for k, v in stress_details.items()}
        sorted_shocks = dict(sorted(shock_data.items(), key=lambda x: x[1]))

        shock_tickers = list(sorted_shocks.keys())
        shocks = list(sorted_shocks.values())

        bars = ax2.barh(shock_tickers, shocks, color='#E63946', edgecolor='white')
        ax2.axvline(x=0, color='black', linewidth=0.5)
        ax2.set_xlabel('Initial Shock (%)')
        ax2.set_title('Initial Price Shock Applied to Thai Assets\n(Randomized around expected drop)')

        # Add value labels
        for bar, val in zip(bars, shocks):
            xpos = bar.get_width() - 0.5 if val < 0 else bar.get_width() + 0.5
            ax2.text(xpos, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
                    va='center', ha='right' if val < 0 else 'left', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_summary_dashboard(results: dict, output_path: str) -> None:
    """
    Create a comprehensive summary dashboard.

    Args:
        results: Dictionary with portfolio results
        output_path: Path to save the figure
    """
    fig = plt.figure(figsize=(16, 12))

    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # ===== Section 1: Portfolio Weights (top left) =====
    ax1 = fig.add_subplot(gs[0, 0])

    weights = results['optimal_weights']
    significant_weights = {k: v for k, v in weights.items() if v > 0.001}
    sorted_weights = dict(sorted(significant_weights.items(), key=lambda x: x[1], reverse=True))

    colors_map = {
        'Thai': '#E63946',
        'Fixed Income': '#457B9D',
        'FX': '#2A9D8F',
        'Global': '#F4A261'
    }

    def get_asset_type(ticker):
        if ticker.endswith('.BK'):
            return 'Thai'
        elif ticker in ['LEMB', 'VWOB', 'EMLC']:
            return 'Fixed Income'
        elif ticker == 'THB=X':
            return 'FX'
        else:
            return 'Global'

    pie_colors = [colors_map[get_asset_type(t)] for t in sorted_weights.keys()]
    wedges, texts, autotexts = ax1.pie(
        sorted_weights.values(),
        labels=sorted_weights.keys(),
        autopct='%1.1f%%',
        colors=pie_colors,
        pctdistance=0.75
    )
    ax1.set_title('Optimal Portfolio Weights')

    # ===== Section 2: Key Metrics (top center) =====
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')

    metrics = results['portfolio_metrics']
    backtest = results['backward_looking_backtest']['metrics']
    stress = results['forward_looking_stress_simulation']['portfolio_statistics']

    metrics_text = (
        "PORTFOLIO METRICS\n"
        "─" * 30 + "\n\n"
        f"Expected Annual Return:  {metrics['expected_annual_return']*100:>8.1f}%\n"
        f"Expected Annual Vol:     {metrics['expected_annual_volatility']*100:>8.1f}%\n"
        f"Sharpe Ratio:            {metrics['sharpe_ratio']:>8.2f}\n\n"
        "BACKTEST RESULTS\n"
        "─" * 30 + "\n\n"
        f"Total Return:            {backtest['total_return_pct']:>8.1f}%\n"
        f"Annualized Return:       {backtest['annualized_return_pct']:>8.1f}%\n"
        f"Max Drawdown:            {backtest['max_drawdown_pct']:>8.1f}%\n\n"
        "STRESS SCENARIO\n"
        "─" * 30 + "\n\n"
        f"Mean Return:             {stress['mean_return_pct']:>8.1f}%\n"
        f"VaR (5%):                {stress['var_5_pct']:>8.1f}%\n"
        f"Prob. of Loss:           {stress['prob_negative_return']:>8.1f}%\n"
    )

    ax2.text(0.1, 0.95, metrics_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))

    # ===== Section 3: Data Summary (top right) =====
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')

    data_summary = results['data_summary']
    metadata = results['metadata']

    summary_text = (
        "DATA SUMMARY\n"
        "─" * 30 + "\n\n"
        f"Historical Period:       {metadata['historical_years']} years\n"
        f"Forward Simulation:      {metadata['forward_looking_days']} days\n"
        f"Monte Carlo Runs:        {metadata['n_simulations']}\n\n"
        f"Assets Requested:        {len(data_summary['tickers_requested'])}\n"
        f"Assets Available:        {len(data_summary['tickers_available'])}\n"
        f"Assets Dropped:          {len(data_summary['tickers_dropped'])}\n\n"
        f"Data Start:              {data_summary['data_start_date']}\n"
        f"Data End:                {data_summary['data_end_date']}\n"
        f"Trading Days:            {data_summary['num_trading_days']}\n"
    )

    ax3.text(0.1, 0.95, summary_text, transform=ax3.transAxes, fontsize=11,
             verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))

    # ===== Section 4: Historical Performance (middle, full width) =====
    ax4 = fig.add_subplot(gs[1, :])

    returns_data = results['backward_looking_backtest']['returns_data']
    dates = pd.to_datetime(returns_data['dates'])
    values = np.array(returns_data['values'])

    ax4.plot(dates, values, linewidth=1.5, color='#1D3557')
    ax4.fill_between(dates, values, alpha=0.3, color='#457B9D')
    ax4.set_ylabel('Portfolio Value')
    ax4.set_title('Historical Backtest Performance (Starting Value = 100)')
    ax4.xaxis.set_major_locator(mdates.YearLocator())
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # ===== Section 5: Stress Simulation (bottom left and center) =====
    ax5 = fig.add_subplot(gs[2, :2])

    sim_data = results['forward_looking_stress_simulation']['simulation_returns_data']
    days = np.array(sim_data['trading_days'])
    paths = sim_data['percentile_paths']

    ax5.fill_between(days, paths['p5'], paths['p95'], alpha=0.2, color='#E63946', label='5th-95th %ile')
    ax5.fill_between(days, paths['p25'], paths['p75'], alpha=0.4, color='#E63946', label='25th-75th %ile')
    ax5.plot(days, paths['p50'], linewidth=2, color='#1D3557', label='Median')
    ax5.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Trading Days')
    ax5.set_ylabel('Portfolio Value')
    ax5.set_title('1-Year Forward Stress Simulation (Trump Tariff Scenario)')
    ax5.legend(loc='upper left')

    # ===== Section 6: Risk Probabilities (bottom right) =====
    ax6 = fig.add_subplot(gs[2, 2])

    prob_metrics = {
        'Negative\nReturn': stress['prob_negative_return'],
        'Loss\n> 10%': stress['prob_loss_10pct'],
        'Loss\n> 20%': stress['prob_loss_20pct']
    }

    bars = ax6.bar(prob_metrics.keys(), prob_metrics.values(), color='#E63946', edgecolor='white')
    ax6.set_ylabel('Probability (%)')
    ax6.set_title('Loss Probabilities')
    ax6.set_ylim(0, max(100, max(prob_metrics.values()) * 1.3 + 5))

    for bar, val in zip(bars, prob_metrics.values()):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add title
    fig.suptitle('Portfolio Optimization & Stress Testing Dashboard', fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_all_visualizations(
    json_path: str = 'portfolio_results.json',
    output_dir: str = 'results'
) -> None:
    """
    Generate all visualizations from the results JSON.

    Args:
        json_path: Path to the JSON results file
        output_dir: Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load results
    print(f"Loading results from {json_path}...")
    results = load_results(json_path)

    print(f"\nGenerating visualizations in '{output_dir}/' directory...\n")

    # Generate all plots
    plot_optimal_weights(results, os.path.join(output_dir, '1_optimal_weights.png'))
    plot_backtest_performance(results, os.path.join(output_dir, '2_backtest_performance.png'))
    plot_stress_simulation(results, os.path.join(output_dir, '3_stress_simulation.png'))
    plot_asset_stress_impact(results, os.path.join(output_dir, '4_asset_stress_impact.png'))
    plot_summary_dashboard(results, os.path.join(output_dir, '5_summary_dashboard.png'))

    # Copy JSON to results folder
    import shutil
    json_output = os.path.join(output_dir, 'portfolio_results.json')
    shutil.copy(json_path, json_output)
    print(f"Saved: {json_output}")

    print(f"\n{'='*50}")
    print(f"All visualizations saved to '{output_dir}/' directory")
    print(f"{'='*50}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate portfolio visualizations')
    parser.add_argument(
        '--input', type=str, default='portfolio_results.json',
        help='Input JSON file path'
    )
    parser.add_argument(
        '--output-dir', type=str, default='results',
        help='Output directory for visualizations'
    )

    args = parser.parse_args()

    generate_all_visualizations(
        json_path=args.input,
        output_dir=args.output_dir
    )
