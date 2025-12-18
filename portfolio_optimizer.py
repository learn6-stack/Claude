"""
Portfolio Optimizer Module
Mean-variance portfolio optimization using scipy.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional, Tuple


def portfolio_return(weights: np.ndarray, mean_returns: np.ndarray) -> float:
    """Calculate expected portfolio return."""
    return np.dot(weights, mean_returns)


def portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Calculate portfolio volatility (standard deviation)."""
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


def portfolio_sharpe(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.02
) -> float:
    """Calculate portfolio Sharpe ratio."""
    ret = portfolio_return(weights, mean_returns)
    vol = portfolio_volatility(weights, cov_matrix)
    return (ret - risk_free_rate) / vol


def neg_sharpe(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float
) -> float:
    """Negative Sharpe ratio for minimization."""
    return -portfolio_sharpe(weights, mean_returns, cov_matrix, risk_free_rate)


def optimize_portfolio(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.02,
    target_return: Optional[float] = None,
    allow_short: bool = False
) -> dict:
    """
    Optimize portfolio weights using mean-variance optimization.

    Args:
        mean_returns: Annualized mean returns for each asset
        cov_matrix: Annualized covariance matrix
        risk_free_rate: Risk-free rate for Sharpe ratio calculation
        target_return: Target return for minimum variance portfolio (optional)
        allow_short: Whether to allow short selling

    Returns:
        Dictionary with optimal weights and portfolio metrics
    """
    n_assets = len(mean_returns)
    tickers = list(mean_returns.index)

    # Convert to numpy arrays
    mu = mean_returns.values
    sigma = cov_matrix.values

    # Initial guess (equal weights)
    init_weights = np.ones(n_assets) / n_assets

    # Constraints
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]  # weights sum to 1

    if target_return is not None:
        constraints.append({
            'type': 'eq',
            'fun': lambda w: portfolio_return(w, mu) - target_return
        })

    # Bounds
    if allow_short:
        bounds = tuple((-1, 1) for _ in range(n_assets))
    else:
        bounds = tuple((0, 1) for _ in range(n_assets))

    # Optimize for maximum Sharpe ratio
    if target_return is None:
        result = minimize(
            neg_sharpe,
            init_weights,
            args=(mu, sigma, risk_free_rate),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )
    else:
        # Minimize variance for target return
        result = minimize(
            lambda w: portfolio_volatility(w, sigma),
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )

    if not result.success:
        print(f"Optimization warning: {result.message}")

    optimal_weights = result.x

    # Calculate portfolio metrics
    port_return = portfolio_return(optimal_weights, mu)
    port_vol = portfolio_volatility(optimal_weights, sigma)
    port_sharpe = portfolio_sharpe(optimal_weights, mu, sigma, risk_free_rate)

    # Create weights dictionary
    weights_dict = {ticker: float(w) for ticker, w in zip(tickers, optimal_weights)}

    return {
        'weights': weights_dict,
        'expected_return': float(port_return),
        'volatility': float(port_vol),
        'sharpe_ratio': float(port_sharpe),
        'optimization_success': result.success,
        'tickers': tickers
    }


def calculate_efficient_frontier(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    n_points: int = 50,
    risk_free_rate: float = 0.02,
    allow_short: bool = False
) -> pd.DataFrame:
    """
    Calculate the efficient frontier.

    Args:
        mean_returns: Annualized mean returns
        cov_matrix: Annualized covariance matrix
        n_points: Number of points on the frontier
        risk_free_rate: Risk-free rate
        allow_short: Whether to allow short selling

    Returns:
        DataFrame with efficient frontier points
    """
    min_ret = mean_returns.min()
    max_ret = mean_returns.max()
    target_returns = np.linspace(min_ret, max_ret, n_points)

    frontier_points = []

    for target in target_returns:
        try:
            result = optimize_portfolio(
                mean_returns,
                cov_matrix,
                risk_free_rate=risk_free_rate,
                target_return=target,
                allow_short=allow_short
            )
            frontier_points.append({
                'target_return': target,
                'expected_return': result['expected_return'],
                'volatility': result['volatility'],
                'sharpe_ratio': result['sharpe_ratio']
            })
        except Exception:
            continue

    return pd.DataFrame(frontier_points)


def backtest_portfolio(
    weights: dict,
    returns: pd.DataFrame,
    initial_value: float = 100.0
) -> Tuple[pd.Series, dict]:
    """
    Backtest portfolio with given weights.

    Args:
        weights: Dictionary of asset weights
        returns: DataFrame with daily returns
        initial_value: Initial portfolio value

    Returns:
        Tuple of (portfolio value series, backtest metrics)
    """
    # Align weights with available returns
    available_tickers = [t for t in weights.keys() if t in returns.columns]
    aligned_weights = np.array([weights[t] for t in available_tickers])
    aligned_returns = returns[available_tickers]

    # Normalize weights
    aligned_weights = aligned_weights / aligned_weights.sum()

    # Calculate portfolio returns
    portfolio_returns = (aligned_returns * aligned_weights).sum(axis=1)

    # Calculate cumulative returns
    cumulative_returns = (1 + portfolio_returns).cumprod()
    portfolio_values = initial_value * cumulative_returns

    # Calculate metrics
    total_return = (portfolio_values.iloc[-1] / initial_value - 1) * 100
    annualized_return = ((portfolio_values.iloc[-1] / initial_value) ** (252 / len(portfolio_values)) - 1) * 100
    annualized_vol = portfolio_returns.std() * np.sqrt(252) * 100
    sharpe = (annualized_return - 2) / annualized_vol  # Assuming 2% risk-free rate

    # Maximum drawdown
    rolling_max = portfolio_values.expanding().max()
    drawdowns = (portfolio_values - rolling_max) / rolling_max
    max_drawdown = drawdowns.min() * 100

    metrics = {
        'total_return_pct': float(total_return),
        'annualized_return_pct': float(annualized_return),
        'annualized_volatility_pct': float(annualized_vol),
        'sharpe_ratio': float(sharpe),
        'max_drawdown_pct': float(max_drawdown),
        'start_date': str(portfolio_values.index[0].date()),
        'end_date': str(portfolio_values.index[-1].date()),
        'num_trading_days': len(portfolio_values)
    }

    return portfolio_values, metrics


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    n_assets = 5
    tickers = [f'Asset_{i}' for i in range(n_assets)]

    # Generate random returns
    mean_returns = pd.Series(np.random.uniform(0.05, 0.15, n_assets), index=tickers)

    # Generate positive definite covariance matrix
    random_matrix = np.random.randn(n_assets, n_assets)
    cov_matrix = pd.DataFrame(
        np.dot(random_matrix, random_matrix.T) * 0.01,
        index=tickers,
        columns=tickers
    )

    result = optimize_portfolio(mean_returns, cov_matrix, allow_short=False)
    print("Optimal Portfolio:")
    print(f"Weights: {result['weights']}")
    print(f"Expected Return: {result['expected_return']:.2%}")
    print(f"Volatility: {result['volatility']:.2%}")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
