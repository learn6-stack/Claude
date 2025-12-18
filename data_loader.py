"""
Data Loader Module
Downloads historical data from Yahoo Finance and performs data cleaning.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Optional


def download_data(
    tickers: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    years_back: int = 10
) -> pd.DataFrame:
    """
    Download historical price data from Yahoo Finance.

    Args:
        tickers: List of ticker symbols
        start_date: Start date in 'YYYY-MM-DD' format (optional)
        end_date: End date in 'YYYY-MM-DD' format (optional)
        years_back: Number of years of historical data if dates not specified

    Returns:
        DataFrame with adjusted close prices for all tickers
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    if start_date is None:
        start_dt = datetime.now() - timedelta(days=years_back * 365)
        start_date = start_dt.strftime('%Y-%m-%d')

    print(f"Downloading data from {start_date} to {end_date}...")
    print(f"Tickers: {tickers}")

    # Download data
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)

    # Extract Close prices
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close']
    else:
        prices = data[['Close']]
        prices.columns = tickers

    print(f"Downloaded {len(prices)} trading days of data")
    return prices


def clean_data(
    prices: pd.DataFrame,
    na_threshold: float = 0.3,
    outlier_std: float = 4.0
) -> Tuple[pd.DataFrame, dict]:
    """
    Clean the price data by handling NAs and outliers.

    Args:
        prices: DataFrame with price data
        na_threshold: Maximum fraction of NAs allowed per column
        outlier_std: Number of standard deviations for outlier detection

    Returns:
        Tuple of (cleaned DataFrame, cleaning report dict)
    """
    report = {
        'original_shape': prices.shape,
        'dropped_tickers': [],
        'na_filled': {},
        'outliers_replaced': {}
    }

    # Drop tickers with too many NAs
    na_fractions = prices.isna().sum() / len(prices)
    tickers_to_drop = na_fractions[na_fractions > na_threshold].index.tolist()

    if tickers_to_drop:
        print(f"Dropping tickers with >{na_threshold*100}% NAs: {tickers_to_drop}")
        prices = prices.drop(columns=tickers_to_drop)
        report['dropped_tickers'] = tickers_to_drop

    # Forward fill then backward fill remaining NAs
    for col in prices.columns:
        na_count = prices[col].isna().sum()
        if na_count > 0:
            report['na_filled'][col] = na_count

    prices = prices.ffill().bfill()

    # Calculate returns for outlier detection
    returns = prices.pct_change().dropna()

    # Detect and handle outliers in returns
    cleaned_prices = prices.copy()

    for col in returns.columns:
        col_returns = returns[col]
        mean_ret = col_returns.mean()
        std_ret = col_returns.std()

        # Find outliers (beyond outlier_std standard deviations)
        outlier_mask = np.abs(col_returns - mean_ret) > outlier_std * std_ret
        outlier_count = outlier_mask.sum()

        if outlier_count > 0:
            report['outliers_replaced'][col] = int(outlier_count)

            # Replace outlier returns with clipped values
            clipped_returns = col_returns.clip(
                lower=mean_ret - outlier_std * std_ret,
                upper=mean_ret + outlier_std * std_ret
            )

            # Reconstruct prices from clipped returns
            first_price = cleaned_prices[col].iloc[0]
            reconstructed = first_price * (1 + clipped_returns).cumprod()
            cleaned_prices.loc[reconstructed.index, col] = reconstructed

    # Drop any remaining rows with NAs
    cleaned_prices = cleaned_prices.dropna()

    report['final_shape'] = cleaned_prices.shape
    print(f"Data cleaned: {report['original_shape']} -> {report['final_shape']}")

    return cleaned_prices, report


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily returns from prices.

    Args:
        prices: DataFrame with price data

    Returns:
        DataFrame with daily returns
    """
    returns = prices.pct_change().dropna()
    return returns


def get_return_statistics(returns: pd.DataFrame) -> dict:
    """
    Calculate return statistics for the portfolio optimization.

    Args:
        returns: DataFrame with daily returns

    Returns:
        Dictionary with mean returns and covariance matrix
    """
    # Annualize (252 trading days)
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    return {
        'mean_returns': mean_returns,
        'cov_matrix': cov_matrix,
        'daily_returns': returns
    }


def load_and_prepare_data(
    tickers: List[str],
    years_back: int = 10,
    na_threshold: float = 0.3,
    outlier_std: float = 4.0
) -> Tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """
    Main function to load and prepare all data for portfolio optimization.

    Args:
        tickers: List of ticker symbols
        years_back: Number of years of historical data
        na_threshold: Maximum fraction of NAs allowed per column
        outlier_std: Number of standard deviations for outlier detection

    Returns:
        Tuple of (prices, returns, statistics, cleaning_report)
    """
    # Download data
    prices = download_data(tickers, years_back=years_back)

    # Clean data
    cleaned_prices, cleaning_report = clean_data(
        prices,
        na_threshold=na_threshold,
        outlier_std=outlier_std
    )

    # Calculate returns
    returns = calculate_returns(cleaned_prices)

    # Get statistics
    statistics = get_return_statistics(returns)

    return cleaned_prices, returns, statistics, cleaning_report


if __name__ == "__main__":
    # Test with sample tickers
    tickers = [
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

    prices, returns, stats, report = load_and_prepare_data(tickers, years_back=10)
    print(f"\nAvailable tickers: {list(prices.columns)}")
    print(f"\nAnnualized mean returns:\n{stats['mean_returns']}")
