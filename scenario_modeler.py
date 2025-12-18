"""
Scenario Modeler Module
Stress testing with customizable scenarios using Geometric Brownian Motion (GBM).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class StressScenario:
    """
    Configuration for a stress scenario.

    Attributes:
        name: Name of the scenario
        affected_assets: List of tickers affected by the stress
        expected_drop_pct: Expected price drop percentage (e.g., 10 for 10%)
        drop_std_pct: Standard deviation of the drop (e.g., 2 for ±2%)
        volatility_increase_pct: Increase in volatility (e.g., 10 for 10% increase)
        start_date: When the stress begins (optional, defaults to today)
        duration_days: Duration of the stress scenario in trading days
    """
    name: str
    affected_assets: List[str]
    expected_drop_pct: float = 10.0
    drop_std_pct: float = 2.0
    volatility_increase_pct: float = 10.0
    start_date: Optional[str] = None
    duration_days: int = 252  # 1 year of trading days


def create_trump_tariff_scenario(
    thai_assets: List[str],
    expected_drop_pct: float = 10.0,
    drop_std_pct: float = 2.0,
    volatility_increase_pct: float = 10.0,
    start_date: Optional[str] = None,
    duration_days: int = 252
) -> StressScenario:
    """
    Create a Trump tariff stress scenario for Thai assets.

    Args:
        thai_assets: List of Thai-related asset tickers
        expected_drop_pct: Expected drop percentage
        drop_std_pct: Standard deviation of the drop
        volatility_increase_pct: Volatility increase percentage
        start_date: Start date of the scenario
        duration_days: Duration in trading days

    Returns:
        StressScenario configuration
    """
    return StressScenario(
        name="Trump Tariff Impact",
        affected_assets=thai_assets,
        expected_drop_pct=expected_drop_pct,
        drop_std_pct=drop_std_pct,
        volatility_increase_pct=volatility_increase_pct,
        start_date=start_date,
        duration_days=duration_days
    )


def simulate_gbm(
    initial_prices: pd.Series,
    mean_returns: pd.Series,
    volatilities: pd.Series,
    n_days: int,
    n_simulations: int = 1000,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Simulate asset prices using Geometric Brownian Motion.

    GBM formula: S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)

    Args:
        initial_prices: Initial prices for each asset
        mean_returns: Annualized mean returns (drift)
        volatilities: Annualized volatilities
        n_days: Number of trading days to simulate
        n_simulations: Number of Monte Carlo simulations
        random_seed: Random seed for reproducibility

    Returns:
        Array of shape (n_simulations, n_days, n_assets) with simulated prices
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    n_assets = len(initial_prices)
    dt = 1 / 252  # Daily timestep

    # Convert to numpy arrays
    S0 = initial_prices.values
    mu = mean_returns.values
    sigma = volatilities.values

    # Initialize price array
    prices = np.zeros((n_simulations, n_days + 1, n_assets))
    prices[:, 0, :] = S0

    # Generate random shocks
    Z = np.random.standard_normal((n_simulations, n_days, n_assets))

    # Simulate GBM paths
    for t in range(1, n_days + 1):
        drift = (mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt) * Z[:, t - 1, :]
        prices[:, t, :] = prices[:, t - 1, :] * np.exp(drift + diffusion)

    return prices[:, 1:, :]  # Exclude initial price


def apply_stress_scenario(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    statistics: dict,
    scenario: StressScenario,
    n_simulations: int = 1000,
    random_seed: Optional[int] = 42
) -> Dict:
    """
    Apply a stress scenario to simulate future prices.

    Args:
        prices: Historical prices DataFrame
        returns: Historical returns DataFrame
        statistics: Return statistics dictionary
        scenario: StressScenario configuration
        n_simulations: Number of Monte Carlo simulations
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with simulation results
    """
    np.random.seed(random_seed)

    # Get last prices
    last_prices = prices.iloc[-1].copy()
    tickers = list(prices.columns)

    # Get historical statistics
    mean_returns = statistics['mean_returns'].copy()
    daily_vol = returns.std()
    annual_vol = daily_vol * np.sqrt(252)

    # Apply stress adjustments to affected assets
    stressed_initial_prices = last_prices.copy()
    stressed_volatilities = annual_vol.copy()

    affected_tickers = [t for t in scenario.affected_assets if t in tickers]

    stress_details = {}
    for ticker in affected_tickers:
        # Apply initial shock with randomness
        shock = np.random.normal(
            -scenario.expected_drop_pct / 100,
            scenario.drop_std_pct / 100
        )
        stressed_initial_prices[ticker] *= (1 + shock)

        # Increase volatility
        vol_multiplier = 1 + scenario.volatility_increase_pct / 100
        stressed_volatilities[ticker] *= vol_multiplier

        stress_details[ticker] = {
            'initial_shock_pct': shock * 100,
            'volatility_multiplier': vol_multiplier,
            'original_price': float(last_prices[ticker]),
            'stressed_price': float(stressed_initial_prices[ticker])
        }

    # Run GBM simulation
    simulated_prices = simulate_gbm(
        initial_prices=stressed_initial_prices,
        mean_returns=mean_returns,
        volatilities=stressed_volatilities,
        n_days=scenario.duration_days,
        n_simulations=n_simulations,
        random_seed=random_seed
    )

    # Calculate simulation statistics
    final_prices = simulated_prices[:, -1, :]  # Shape: (n_simulations, n_assets)

    # Calculate returns from stressed initial prices
    initial_prices_arr = stressed_initial_prices.values
    simulated_returns = (final_prices / initial_prices_arr - 1) * 100

    # Calculate portfolio-level statistics (assuming equal weights for now)
    n_assets = len(tickers)
    equal_weights = np.ones(n_assets) / n_assets

    portfolio_returns = np.dot(simulated_returns, equal_weights)

    return {
        'scenario_name': scenario.name,
        'affected_assets': affected_tickers,
        'duration_days': scenario.duration_days,
        'n_simulations': n_simulations,
        'stress_details': stress_details,
        'simulated_prices': simulated_prices,
        'tickers': tickers,
        'simulation_statistics': {
            'mean_return_by_asset': {
                ticker: float(simulated_returns[:, i].mean())
                for i, ticker in enumerate(tickers)
            },
            'std_return_by_asset': {
                ticker: float(simulated_returns[:, i].std())
                for i, ticker in enumerate(tickers)
            },
            'percentile_5_by_asset': {
                ticker: float(np.percentile(simulated_returns[:, i], 5))
                for i, ticker in enumerate(tickers)
            },
            'percentile_95_by_asset': {
                ticker: float(np.percentile(simulated_returns[:, i], 95))
                for i, ticker in enumerate(tickers)
            }
        }
    }


def simulate_portfolio_stress(
    weights: dict,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    statistics: dict,
    scenario: StressScenario,
    n_simulations: int = 1000,
    random_seed: Optional[int] = 42
) -> Dict:
    """
    Simulate portfolio performance under stress scenario.

    Args:
        weights: Portfolio weights dictionary
        prices: Historical prices DataFrame
        returns: Historical returns DataFrame
        statistics: Return statistics dictionary
        scenario: StressScenario configuration
        n_simulations: Number of Monte Carlo simulations
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with portfolio simulation results
    """
    np.random.seed(random_seed)

    # Apply stress scenario
    stress_result = apply_stress_scenario(
        prices=prices,
        returns=returns,
        statistics=statistics,
        scenario=scenario,
        n_simulations=n_simulations,
        random_seed=random_seed
    )

    simulated_prices = stress_result['simulated_prices']
    tickers = stress_result['tickers']

    # Align weights with available tickers
    aligned_weights = np.array([
        weights.get(ticker, 0.0) for ticker in tickers
    ])
    aligned_weights = aligned_weights / aligned_weights.sum()

    # Calculate daily portfolio values for each simulation
    n_sims, n_days, n_assets = simulated_prices.shape

    # Initial portfolio value (normalized to 100)
    initial_value = 100.0

    # Calculate portfolio values over time
    portfolio_values = np.zeros((n_sims, n_days))

    for sim in range(n_sims):
        # Normalize prices to returns
        sim_prices = simulated_prices[sim]
        initial_sim_prices = sim_prices[0]
        price_ratios = sim_prices / initial_sim_prices

        # Portfolio value as weighted sum of price ratios
        portfolio_values[sim] = initial_value * np.dot(price_ratios, aligned_weights)

    # Calculate portfolio returns
    final_values = portfolio_values[:, -1]
    portfolio_returns = (final_values / initial_value - 1) * 100

    # Calculate VaR and CVaR
    var_5 = np.percentile(portfolio_returns, 5)
    cvar_5 = portfolio_returns[portfolio_returns <= var_5].mean()

    # Time series percentiles
    percentile_paths = {
        'p5': np.percentile(portfolio_values, 5, axis=0).tolist(),
        'p25': np.percentile(portfolio_values, 25, axis=0).tolist(),
        'p50': np.percentile(portfolio_values, 50, axis=0).tolist(),
        'p75': np.percentile(portfolio_values, 75, axis=0).tolist(),
        'p95': np.percentile(portfolio_values, 95, axis=0).tolist(),
    }

    return {
        'scenario_name': scenario.name,
        'affected_assets': stress_result['affected_assets'],
        'stress_details': stress_result['stress_details'],
        'duration_days': scenario.duration_days,
        'n_simulations': n_simulations,
        'portfolio_statistics': {
            'mean_return_pct': float(portfolio_returns.mean()),
            'std_return_pct': float(portfolio_returns.std()),
            'median_return_pct': float(np.median(portfolio_returns)),
            'var_5_pct': float(var_5),
            'cvar_5_pct': float(cvar_5),
            'min_return_pct': float(portfolio_returns.min()),
            'max_return_pct': float(portfolio_returns.max()),
            'prob_negative_return': float((portfolio_returns < 0).mean() * 100),
            'prob_loss_10pct': float((portfolio_returns < -10).mean() * 100),
            'prob_loss_20pct': float((portfolio_returns < -20).mean() * 100),
        },
        'percentile_paths': percentile_paths,
        'asset_statistics': stress_result['simulation_statistics'],
        'weights_used': {ticker: float(w) for ticker, w in zip(tickers, aligned_weights)}
    }


def get_simulation_returns_data(
    simulation_result: Dict,
    sample_size: int = 100
) -> List[Dict]:
    """
    Extract sample simulation return paths for output.

    Args:
        simulation_result: Result from simulate_portfolio_stress
        sample_size: Number of sample paths to include

    Returns:
        List of dictionaries with sample return data
    """
    percentile_paths = simulation_result['percentile_paths']
    duration = simulation_result['duration_days']

    # Create trading day index
    days = list(range(1, duration + 1))

    return {
        'trading_days': days,
        'percentile_paths': percentile_paths,
        'summary': {
            'initial_value': 100.0,
            'final_p5': percentile_paths['p5'][-1],
            'final_p50': percentile_paths['p50'][-1],
            'final_p95': percentile_paths['p95'][-1],
        }
    }


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)

    # Create sample data
    tickers = ['DELTA.BK', 'HANA.BK', 'WDC', 'LEMB']
    n_days = 252 * 10
    dates = pd.date_range(end='2024-01-01', periods=n_days, freq='B')

    # Generate random prices
    returns = pd.DataFrame(
        np.random.randn(n_days, len(tickers)) * 0.02,
        index=dates,
        columns=tickers
    )
    prices = (1 + returns).cumprod() * 100

    # Statistics
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    statistics = {
        'mean_returns': mean_returns,
        'cov_matrix': cov_matrix,
        'daily_returns': returns
    }

    # Create Thai assets scenario
    thai_assets = ['DELTA.BK', 'HANA.BK']
    scenario = create_trump_tariff_scenario(
        thai_assets=thai_assets,
        expected_drop_pct=10.0,
        drop_std_pct=2.0,
        volatility_increase_pct=10.0,
        duration_days=252
    )

    # Test weights
    weights = {ticker: 0.25 for ticker in tickers}

    # Run simulation
    result = simulate_portfolio_stress(
        weights=weights,
        prices=prices,
        returns=returns,
        statistics=statistics,
        scenario=scenario,
        n_simulations=1000
    )

    print(f"Scenario: {result['scenario_name']}")
    print(f"Affected assets: {result['affected_assets']}")
    print(f"\nPortfolio Statistics:")
    for key, value in result['portfolio_statistics'].items():
        print(f"  {key}: {value:.2f}")
