"""
Main Runner Script
Runs the complete portfolio optimization and stress testing pipeline.
Generates JSON output with results.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Optional

from data_loader import load_and_prepare_data
from portfolio_optimizer import optimize_portfolio, backtest_portfolio
from scenario_modeler import (
    create_trump_tariff_scenario,
    simulate_portfolio_stress,
    get_simulation_returns_data,
    StressScenario
)


# Default asset universe
DEFAULT_TICKERS = [
    # Thai Export
    'DELTA.BK', 'HANA.BK', 'STA.BK', 'IVL.BK', 'PTTGC.BK',
    # Thai Domestic
    'CPALL.BK', 'AOT.BK', 'BDMS.BK', 'SCB.BK', 'CPN.BK', 'MINT.BK',
    # Global
    'WDC', 'THD',
    # Fixed Income
    'LEMB', 'VWOB', 'EMLC',
    # FX
    'THB=X'
]

# Thai-related assets for stress testing
THAI_ASSETS = [
    'DELTA.BK', 'HANA.BK', 'STA.BK', 'IVL.BK', 'PTTGC.BK',
    'CPALL.BK', 'AOT.BK', 'BDMS.BK', 'SCB.BK', 'CPN.BK', 'MINT.BK',
    'THD', 'THB=X'
]


def run_portfolio_optimization(
    tickers: List[str] = None,
    years_back: int = 10,
    risk_free_rate: float = 0.02,
    stress_scenario_config: Optional[dict] = None,
    n_simulations: int = 1000,
    output_file: str = 'reports/portfolio_results.json'
) -> dict:
    """
    Run the complete portfolio optimization and stress testing pipeline.

    Args:
        tickers: List of ticker symbols (defaults to DEFAULT_TICKERS)
        years_back: Years of historical data
        risk_free_rate: Risk-free rate for optimization
        stress_scenario_config: Custom stress scenario configuration
        n_simulations: Number of Monte Carlo simulations
        output_file: Path for JSON output file

    Returns:
        Dictionary with all results
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS

    print("=" * 60)
    print("PORTFOLIO OPTIMIZATION WITH STRESS TESTING")
    print("=" * 60)

    # Step 1: Load and prepare data
    print("\n[1/4] Loading and cleaning data...")
    try:
        prices, returns, statistics, cleaning_report, benchmark_prices = load_and_prepare_data(
            tickers=tickers,
            years_back=years_back,
            include_benchmark=True,
            benchmark_ticker='^GSPC'
        )
        available_tickers = list(prices.columns)
        print(f"Successfully loaded {len(available_tickers)} assets")
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

    # Step 2: Portfolio optimization
    print("\n[2/4] Running portfolio optimization...")
    optimization_result = optimize_portfolio(
        mean_returns=statistics['mean_returns'],
        cov_matrix=statistics['cov_matrix'],
        risk_free_rate=risk_free_rate,
        allow_short=False  # No shorting constraint
    )

    print(f"Optimal portfolio:")
    print(f"  Expected Return: {optimization_result['expected_return']:.2%}")
    print(f"  Volatility: {optimization_result['volatility']:.2%}")
    print(f"  Sharpe Ratio: {optimization_result['sharpe_ratio']:.2f}")

    # Step 3: Backtest on historical data
    print("\n[3/4] Running historical backtest...")
    portfolio_values, backtest_metrics = backtest_portfolio(
        weights=optimization_result['weights'],
        returns=returns
    )

    print(f"Backtest Results:")
    print(f"  Total Return: {backtest_metrics['total_return_pct']:.2f}%")
    print(f"  Annualized Return: {backtest_metrics['annualized_return_pct']:.2f}%")
    print(f"  Max Drawdown: {backtest_metrics['max_drawdown_pct']:.2f}%")

    # Convert portfolio values to serializable format
    backtest_returns_data = {
        'dates': [str(d.date()) for d in portfolio_values.index],
        'values': portfolio_values.tolist(),
        'returns_pct': ((portfolio_values / portfolio_values.iloc[0] - 1) * 100).tolist()
    }

    # Calculate benchmark performance
    benchmark_data = None
    if benchmark_prices is not None:
        benchmark_aligned = benchmark_prices.reindex(portfolio_values.index).ffill().bfill()
        if isinstance(benchmark_aligned, pd.DataFrame):
            benchmark_values = benchmark_aligned.iloc[:, 0]
        else:
            benchmark_values = benchmark_aligned
        benchmark_normalized = 100 * benchmark_values / benchmark_values.iloc[0]
        benchmark_returns = benchmark_values.pct_change().dropna()

        # Calculate benchmark metrics
        benchmark_total_ret = (benchmark_normalized.iloc[-1] / 100 - 1) * 100
        benchmark_ann_ret = ((benchmark_normalized.iloc[-1] / 100) ** (252 / len(benchmark_normalized)) - 1) * 100
        benchmark_ann_vol = benchmark_returns.std() * np.sqrt(252) * 100

        benchmark_data = {
            'ticker': '^GSPC',
            'name': 'S&P 500',
            'dates': [str(d.date()) for d in benchmark_normalized.index],
            'values': benchmark_normalized.tolist(),
            'metrics': {
                'total_return_pct': float(benchmark_total_ret),
                'annualized_return_pct': float(benchmark_ann_ret),
                'annualized_volatility_pct': float(benchmark_ann_vol)
            }
        }
        print(f"Benchmark (S&P 500):")
        print(f"  Total Return: {benchmark_total_ret:.2f}%")
        print(f"  Annualized Return: {benchmark_ann_ret:.2f}%")

    # Calculate individual stock performance
    individual_stocks_backtest = {}
    for ticker in available_tickers:
        ticker_prices = prices[ticker]
        ticker_normalized = 100 * ticker_prices / ticker_prices.iloc[0]
        ticker_returns = ticker_prices.pct_change().dropna()

        individual_stocks_backtest[ticker] = {
            'dates': [str(d.date()) for d in ticker_normalized.index],
            'values': ticker_normalized.tolist(),
            'total_return_pct': float((ticker_normalized.iloc[-1] / 100 - 1) * 100),
            'annualized_return_pct': float(((ticker_normalized.iloc[-1] / 100) ** (252 / len(ticker_normalized)) - 1) * 100),
            'annualized_volatility_pct': float(ticker_returns.std() * np.sqrt(252) * 100)
        }

    # Calculate historical returns distribution
    portfolio_daily_returns = (portfolio_values.pct_change().dropna() * 100).tolist()
    returns_distribution = {
        'portfolio_daily_returns': portfolio_daily_returns,
        'mean_daily_return': float(np.mean(portfolio_daily_returns)),
        'std_daily_return': float(np.std(portfolio_daily_returns)),
        'skewness': float(pd.Series(portfolio_daily_returns).skew()),
        'kurtosis': float(pd.Series(portfolio_daily_returns).kurtosis()),
        'individual_stocks': {
            ticker: {
                'daily_returns': (prices[ticker].pct_change().dropna() * 100).tolist()[-252:],  # Last year
                'mean': float(returns[ticker].mean() * 100),
                'std': float(returns[ticker].std() * 100)
            }
            for ticker in available_tickers
        }
    }

    # Step 4: Stress scenario simulation
    print("\n[4/4] Running stress scenario simulation...")

    # Configure stress scenario
    if stress_scenario_config is None:
        stress_scenario_config = {
            'expected_drop_pct': 10.0,
            'drop_std_pct': 2.0,
            'volatility_increase_pct': 10.0,
            'duration_days': 252  # 1 year forward
        }

    # Filter Thai assets to those available
    thai_assets_available = [t for t in THAI_ASSETS if t in available_tickers]

    scenario = create_trump_tariff_scenario(
        thai_assets=thai_assets_available,
        expected_drop_pct=stress_scenario_config['expected_drop_pct'],
        drop_std_pct=stress_scenario_config['drop_std_pct'],
        volatility_increase_pct=stress_scenario_config['volatility_increase_pct'],
        duration_days=stress_scenario_config['duration_days']
    )

    stress_result = simulate_portfolio_stress(
        weights=optimization_result['weights'],
        prices=prices,
        returns=returns,
        statistics=statistics,
        scenario=scenario,
        n_simulations=n_simulations
    )

    print(f"Stress Scenario: {stress_result['scenario_name']}")
    print(f"  Affected Assets: {len(stress_result['affected_assets'])}")
    print(f"  Mean Return: {stress_result['portfolio_statistics']['mean_return_pct']:.2f}%")
    print(f"  VaR (5%): {stress_result['portfolio_statistics']['var_5_pct']:.2f}%")
    print(f"  Prob. of Loss: {stress_result['portfolio_statistics']['prob_negative_return']:.1f}%")

    # Get forward simulation data
    forward_simulation_data = get_simulation_returns_data(stress_result)

    # Compile final results
    results = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'historical_years': years_back,
            'forward_looking_days': stress_scenario_config['duration_days'],
            'risk_free_rate': risk_free_rate,
            'n_simulations': n_simulations
        },
        'data_summary': {
            'tickers_requested': tickers,
            'tickers_available': available_tickers,
            'tickers_dropped': cleaning_report['dropped_tickers'],
            'data_start_date': backtest_metrics['start_date'],
            'data_end_date': backtest_metrics['end_date'],
            'num_trading_days': backtest_metrics['num_trading_days']
        },
        'optimal_weights': optimization_result['weights'],
        'portfolio_metrics': {
            'expected_annual_return': optimization_result['expected_return'],
            'expected_annual_volatility': optimization_result['volatility'],
            'sharpe_ratio': optimization_result['sharpe_ratio']
        },
        'backward_looking_backtest': {
            'metrics': backtest_metrics,
            'returns_data': backtest_returns_data,
            'benchmark': benchmark_data,
            'individual_stocks': individual_stocks_backtest
        },
        'returns_distribution': returns_distribution,
        'forward_looking_stress_simulation': {
            'scenario_config': {
                'name': stress_result['scenario_name'],
                'affected_assets': stress_result['affected_assets'],
                'expected_drop_pct': stress_scenario_config['expected_drop_pct'],
                'drop_std_pct': stress_scenario_config['drop_std_pct'],
                'volatility_increase_pct': stress_scenario_config['volatility_increase_pct'],
                'duration_days': stress_scenario_config['duration_days']
            },
            'stress_details': stress_result['stress_details'],
            'portfolio_statistics': stress_result['portfolio_statistics'],
            'simulation_returns_data': forward_simulation_data,
            'asset_statistics': stress_result['asset_statistics'],
            'weights_used': stress_result.get('weights_used', optimization_result['weights'])
        }
    }

    # Ensure output directory exists
    import os
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save to JSON
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    return results


def run_custom_scenario(
    tickers: List[str] = None,
    affected_assets: List[str] = None,
    expected_drop_pct: float = 10.0,
    drop_std_pct: float = 2.0,
    volatility_increase_pct: float = 10.0,
    start_date: Optional[str] = None,
    duration_days: int = 252,
    years_back: int = 10,
    n_simulations: int = 1000,
    output_file: str = 'reports/custom_scenario_results.json'
) -> dict:
    """
    Run portfolio optimization with a custom stress scenario.

    Args:
        tickers: Universe of assets
        affected_assets: Assets affected by the stress scenario
        expected_drop_pct: Expected price drop percentage
        drop_std_pct: Standard deviation of the drop
        volatility_increase_pct: Volatility increase percentage
        start_date: Start date of the scenario
        duration_days: Duration in trading days
        years_back: Years of historical data
        n_simulations: Number of Monte Carlo simulations
        output_file: Output file path

    Returns:
        Results dictionary
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS

    if affected_assets is None:
        affected_assets = THAI_ASSETS

    stress_config = {
        'expected_drop_pct': expected_drop_pct,
        'drop_std_pct': drop_std_pct,
        'volatility_increase_pct': volatility_increase_pct,
        'duration_days': duration_days
    }

    return run_portfolio_optimization(
        tickers=tickers,
        years_back=years_back,
        stress_scenario_config=stress_config,
        n_simulations=n_simulations,
        output_file=output_file
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Portfolio Optimization with Stress Testing'
    )
    parser.add_argument(
        '--years-back', type=int, default=10,
        help='Years of historical data (default: 10)'
    )
    parser.add_argument(
        '--drop-pct', type=float, default=10.0,
        help='Expected drop percentage for stress scenario (default: 10.0)'
    )
    parser.add_argument(
        '--drop-std', type=float, default=2.0,
        help='Standard deviation of drop (default: 2.0)'
    )
    parser.add_argument(
        '--vol-increase', type=float, default=10.0,
        help='Volatility increase percentage (default: 10.0)'
    )
    parser.add_argument(
        '--forward-days', type=int, default=252,
        help='Forward looking simulation days (default: 252)'
    )
    parser.add_argument(
        '--simulations', type=int, default=1000,
        help='Number of Monte Carlo simulations (default: 1000)'
    )
    parser.add_argument(
        '--output', type=str, default='reports/portfolio_results.json',
        help='Output JSON file path (default: reports/portfolio_results.json)'
    )

    args = parser.parse_args()

    stress_config = {
        'expected_drop_pct': args.drop_pct,
        'drop_std_pct': args.drop_std,
        'volatility_increase_pct': args.vol_increase,
        'duration_days': args.forward_days
    }

    results = run_portfolio_optimization(
        years_back=args.years_back,
        stress_scenario_config=stress_config,
        n_simulations=args.simulations,
        output_file=args.output
    )

    print(f"\nResults saved to: {args.output}")
