import math

import pandas as pd
import yfinance as yf

from agent.state import PortfolioState

# Number of trading days in a calendar year.  This constant converts daily
# statistics into annualised ones.  252 is the market standard — the NYSE
# and NASDAQ are open for approximately 252 sessions per year.
_TRADING_DAYS_PER_YEAR = 252


def _fetch_daily_returns(ticker: str) -> pd.Series:
    """Download 1 year of daily closing prices and return the daily return series.

    Daily return on day t = (close_t / close_{t-1}) - 1.
    This gives the fraction gained or lost each day, e.g. 0.012 = +1.2%.

    Returns a pandas Series indexed by date with NaN dropped.
    """
    # yf.Ticker() creates a lazy handle — no network request yet.
    ticker_obj = yf.Ticker(ticker)

    # .history() fetches OHLCV data.  period="1y" requests the last 52 weeks
    # of trading sessions.  Only the "Close" column is needed for returns.
    hist = ticker_obj.history(period="1y")

    # hist["Close"] is a Series of daily closing prices indexed by datetime.
    # .pct_change() computes (value_t - value_{t-1}) / value_{t-1} for each row,
    # which is mathematically equivalent to (close_t / close_{t-1}) - 1.
    # The first row becomes NaN because there is no prior day to compare against.
    # .dropna() removes that NaN so downstream arithmetic is not contaminated.
    return hist["Close"].pct_change().dropna()


def _weighted_portfolio_returns(
    ticker_returns: dict[str, pd.Series],
    weights: dict[str, float],
) -> pd.Series:
    """Combine per-ticker daily returns into a single weighted portfolio series.

    For each day: portfolio_return = sum(weight_i * return_i for all i).

    Args:
        ticker_returns: maps each ticker to its daily return Series.
        weights:        maps each ticker to its decimal weight (0.0 – 1.0).
    """
    # pd.DataFrame(ticker_returns) aligns all series by their date index,
    # filling gaps with NaN where a ticker had no data on a given day.
    # This ensures all series are the same length before multiplication.
    returns_df = pd.DataFrame(ticker_returns)

    # Multiply each column by its scalar weight, then sum across columns (axis=1)
    # to get one weighted return per row (day).
    # .dropna() removes any rows where at least one ticker had no data —
    # a partial day would produce a biased portfolio return.
    portfolio_series = (returns_df * weights).sum(axis=1)
    return portfolio_series.dropna()


def _compute_expected_return(portfolio_returns: pd.Series) -> float:
    """Annualise the mean daily return.

    Formula: E[r] = mean(daily_returns) * 252

    .mean() is the arithmetic average of all daily returns in the series.
    Multiplying by 252 scales it from a daily figure to an annual one under
    the assumption that each trading day is an independent, identically
    distributed draw.
    """
    return float(portfolio_returns.mean() * _TRADING_DAYS_PER_YEAR)


def _compute_max_drawdown(portfolio_returns: pd.Series) -> float:
    """Find the worst peak-to-trough loss over the observation window.

    A drawdown measures how far the portfolio has fallen from its most recent
    high.  The maximum drawdown is the single largest such fall in the period.

    Steps:
      1. Convert daily returns into a cumulative wealth index starting at 1.
      2. Track the running maximum of that index (the "peak").
      3. Compute the percentage decline from the peak for each day.
      4. Return the most negative value (the deepest trough).

    Returns a negative float, e.g. -0.183 means -18.3%.
    """
    # (1 + r_1) * (1 + r_2) * ... gives a dollar-growth index starting at 1.
    # .cumprod() computes the running product along the series.
    cumulative = (1 + portfolio_returns).cumprod()

    # .cummax() returns a series where each element is the maximum value seen
    # up to and including that date — the "high water mark" of the portfolio.
    rolling_peak = cumulative.cummax()

    # Drawdown at each day = (current_value - peak) / peak.
    # When the portfolio is at its peak the value is 0; it is negative otherwise.
    drawdown = (cumulative - rolling_peak) / rolling_peak

    # .min() picks the single most negative value across the entire series —
    # the worst loss from any peak to any subsequent trough.
    return float(drawdown.min())


def _compute_sharpe_ratio(
    portfolio_returns: pd.Series,
    risk_free_rate: float,
) -> float:
    """Compute the annualised Sharpe ratio.

    Formula: Sharpe = (annualised_return - risk_free_rate) / annualised_volatility

    The Sharpe ratio measures how much excess return (above the risk-free rate)
    the portfolio earns per unit of total risk (standard deviation).
    A ratio above 1.0 is generally considered acceptable; above 2.0 is strong.

    Args:
        portfolio_returns: daily return series for the weighted portfolio.
        risk_free_rate:    annual risk-free rate as a decimal (e.g. 0.046 = 4.6%).
    """
    # Annualise the mean daily return using the same scaling as expected_return.
    annualised_return = float(portfolio_returns.mean() * _TRADING_DAYS_PER_YEAR)

    # Daily volatility = standard deviation of daily returns.
    # Multiplying by sqrt(252) annualises it under the i.i.d. assumption —
    # variance scales linearly with time, so std dev scales with sqrt(time).
    annualised_volatility = float(portfolio_returns.std() * math.sqrt(_TRADING_DAYS_PER_YEAR))

    # Guard against division by zero if all daily returns happen to be identical
    # (extremely unlikely in practice, but defensive code is better than a crash).
    if annualised_volatility == 0:
        return 0.0

    # Subtract the risk-free rate from the annualised return to get excess return,
    # then divide by annualised volatility to normalise for risk taken.
    return (annualised_return - risk_free_rate) / annualised_volatility


def portfolio_metrics(state: PortfolioState) -> dict:
    """Compute expected return, max drawdown, and Sharpe ratio for the portfolio.

    Reads:
        state["allocations"]            — ticker → {"allocation": float, ...}
        state["macro_context"]["yield_10y"] — 10-year Treasury yield in percent

    Returns only {"portfolio_metrics": {...}} — LangGraph merges it into state.
    """
    allocations: dict[str, dict] = state["allocations"]

    # Convert allocation percentages (e.g. 35.0) to decimal weights (e.g. 0.35)
    # so they can be used directly in weighted-sum arithmetic.
    # Dividing by 100 transforms percentage points into the [0, 1] range.
    weights: dict[str, float] = {
        ticker: data["allocation"] / 100
        for ticker, data in allocations.items()
    }

    # yield_10y from macro_context is already in percentage form (e.g. 4.6).
    # Dividing by 100 gives the decimal annual rate (e.g. 0.046) that the
    # Sharpe formula expects — consistent with how annualised return is expressed.
    risk_free_rate: float = state["macro_context"]["yield_10y"] / 100

    # Fetch 1 year of daily returns for every ticker in the portfolio.
    # Each call makes one HTTP request to Yahoo Finance.
    ticker_returns: dict[str, pd.Series] = {
        ticker: _fetch_daily_returns(ticker)
        for ticker in allocations
    }

    # Combine the per-ticker series into a single weighted portfolio return series.
    port_returns = _weighted_portfolio_returns(ticker_returns, weights)

    # Compute and round each metric.  round() removes floating-point noise
    # that would otherwise produce values like 0.14200000000000002.
    expected_return = round(_compute_expected_return(port_returns), 4)
    max_drawdown    = round(_compute_max_drawdown(port_returns), 4)
    sharpe_ratio    = round(_compute_sharpe_ratio(port_returns, risk_free_rate), 4)

    # Return only the field this node writes.  LangGraph merges it into state.
    return {
        "portfolio_metrics": {
            "expected_return": expected_return,
            "max_drawdown":    max_drawdown,
            "sharpe_ratio":    sharpe_ratio,
        }
    }
