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


def _compute_volatility(portfolio_returns: pd.Series) -> float:
    """Compute annualised portfolio volatility (standard deviation of returns).

    Formula: volatility = std(daily_returns) * sqrt(252)

    Volatility is the most direct measure of total portfolio risk.  It quantifies
    how much the portfolio's daily return fluctuates around its mean.

    Under the i.i.d. (independent, identically distributed) assumption, daily
    variance is additive over time, so annual variance = daily_variance * 252.
    Taking the square root converts variance back to standard deviation, which
    is in the same units as the returns themselves (fractions / percentages).

    Args:
        portfolio_returns: daily return series for the weighted portfolio.

    Returns a positive float, e.g. 0.182 means 18.2% annualised volatility.
    """
    # .std() computes the sample standard deviation (ddof=1 by default in pandas)
    # of the daily return series.
    # math.sqrt(_TRADING_DAYS_PER_YEAR) is the annualisation factor — identical to
    # the one used inside _compute_sharpe_ratio, but exposed here as its own metric.
    return float(portfolio_returns.std() * math.sqrt(_TRADING_DAYS_PER_YEAR))


def _compute_beta(
    portfolio_returns: pd.Series,
    spy_returns: pd.Series,
) -> float:
    """Compute the portfolio's beta relative to the S&P 500 (SPY).

    Formula: beta = cov(portfolio_returns, spy_returns) / var(spy_returns)

    Beta measures the portfolio's sensitivity to broad market movements.
    A beta of 1.0 means the portfolio moves in lockstep with the S&P 500.
    A beta of 1.5 means it amplifies market moves by 50%.
    A beta of 0.5 means it moves at half the market's pace.
    Negative beta (rare) means the portfolio tends to rise when the market falls.

    Args:
        portfolio_returns: daily return series for the weighted portfolio.
        spy_returns:       daily return series for SPY (the S&P 500 benchmark).
    """
    # .align() reindexes both series to the same date index.
    # join='inner' keeps only dates that appear in BOTH series, ensuring we
    # compare portfolio and benchmark returns from exactly the same trading days.
    # This is important because SPY data may occasionally differ from ticker data
    # (e.g. a newly listed stock or data gaps from Yahoo Finance).
    port_aligned, spy_aligned = portfolio_returns.align(spy_returns, join="inner")

    if len(spy_aligned) < 2:
        # Not enough overlapping data points to compute a meaningful covariance.
        # Return 1.0 (market-neutral) as a safe default rather than crashing.
        return 1.0

    # .cov() computes the sample covariance between the two aligned series.
    # Covariance measures how much the portfolio and the benchmark move together
    # on the same days — positive means they tend to move in the same direction.
    covariance = float(port_aligned.cov(spy_aligned))

    # .var() computes the sample variance of SPY's daily returns.
    # Dividing covariance by SPY's variance normalises for the benchmark's own
    # level of volatility, giving a scale-free sensitivity coefficient.
    spy_variance = float(spy_aligned.var())

    if spy_variance == 0:
        # If SPY had zero variance (every day the same return), beta is undefined.
        # Return 1.0 as the neutral fallback.
        return 1.0

    return covariance / spy_variance


def _compute_var_95(portfolio_returns: pd.Series) -> float:
    """Compute the 1-day Value at Risk at the 95% confidence level.

    VaR 95% answers the question: "On a typical bad day (the worst 5% of days),
    what is the minimum loss we should expect?"

    Formula: VaR_95 = 5th percentile of the daily return distribution

    This is the historical (non-parametric) VaR — no assumption about the
    distribution shape.  It simply sorts all observed daily returns and reads
    off the value at the 5th percentile.

    Args:
        portfolio_returns: daily return series for the weighted portfolio.

    Returns a negative float, e.g. -0.023 means a worst-day loss of 2.3%
    at 95% confidence — on 95% of days, losses will be *smaller* than this.
    """
    # .quantile(0.05) returns the value below which 5% of observations fall.
    # Because returns are predominantly small positive numbers with a left tail,
    # the 5th percentile is typically a negative number (a loss day).
    return float(portfolio_returns.quantile(0.05))


def _compute_sortino_ratio(
    portfolio_returns: pd.Series,
    risk_free_rate: float,
) -> float:
    """Compute the annualised Sortino ratio.

    Formula: Sortino = (annualised_return - risk_free_rate) / downside_deviation

    The Sortino ratio is a refinement of the Sharpe ratio.  Sharpe penalises
    *all* volatility equally — including upward price swings that benefit the
    investor.  Sortino only penalises *downside* volatility (returns below zero),
    making it a better measure of risk-adjusted performance for asymmetric return
    distributions.

    A Sortino > 1.0 is generally considered acceptable.
    Sortino is typically higher than Sharpe for the same portfolio because the
    denominator (downside deviation) is smaller than total standard deviation.

    Args:
        portfolio_returns: daily return series for the weighted portfolio.
        risk_free_rate:    annual risk-free rate as a decimal (e.g. 0.046 = 4.6%).
    """
    # Annualise the mean daily return — same formula as in expected_return and Sharpe.
    annualised_return = float(portfolio_returns.mean() * _TRADING_DAYS_PER_YEAR)

    # Filter the return series to keep only days with a negative return.
    # "Below zero" is the minimum acceptable return (MAR = 0) — a common default.
    # Using MAR = 0 means any day with a loss is counted as downside.
    downside_returns = portfolio_returns[portfolio_returns < 0]

    if len(downside_returns) == 0:
        # No negative return days in the observation window — the portfolio never
        # lost money on any single day.  Sortino would technically be infinite.
        # Returning 0.0 signals "undefined / no downside data" rather than inf,
        # which would break JSON serialisation and display logic.
        return 0.0

    # Compute the standard deviation of *only* the negative return days,
    # then annualise with sqrt(252) — same annualisation as Sharpe volatility.
    # This is the "downside deviation" denominator specific to the Sortino ratio.
    downside_deviation = float(downside_returns.std() * math.sqrt(_TRADING_DAYS_PER_YEAR))

    if downside_deviation == 0:
        # All negative returns were identical — vanishingly rare, but guard anyway.
        return 0.0

    return (annualised_return - risk_free_rate) / downside_deviation


def _compute_concentration_risk(weights: dict[str, float]) -> float:
    """Compute the Herfindahl-Hirschman Index (HHI) as a concentration risk measure.

    Formula: HHI = sum(w_i^2) for all tickers i, where w_i is the decimal weight.

    HHI is a standard measure of market concentration originally used in antitrust
    economics, applied here to portfolio allocation:
      - HHI = 1/N for a perfectly equal-weight portfolio of N stocks (minimum risk).
      - HHI = 1.0 if 100% of the portfolio is in a single stock (maximum concentration).

    For reference:
      - 3 equal-weight positions: HHI ≈ 0.333
      - 5 equal-weight positions: HHI = 0.200
      - 10 equal-weight positions: HHI = 0.100

    A lower value means better diversification of allocation weights.

    Args:
        weights: dict mapping each ticker to its decimal weight (0.0 – 1.0).

    Returns a float in (0, 1].
    """
    # w ** 2 squares each decimal weight.
    # sum() adds them all together.
    # No annualisation or time scaling needed — HHI is a pure snapshot of weights.
    return float(sum(w ** 2 for w in weights.values()))


def portfolio_metrics(state: PortfolioState) -> dict:
    """Compute all eight quantitative metrics for the recommended portfolio.

    Reads:
        state["allocations"]                  — ticker → {"allocation": float, ...}
        state["macro_context"]["yield_10y"]   — 10-year Treasury yield in percent

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
    # Sharpe and Sortino formulas expect — consistent with how annualised
    # return is expressed.
    risk_free_rate: float = state["macro_context"]["yield_10y"] / 100

    # Fetch 1 year of daily returns for every ticker in the portfolio.
    # Each call makes one HTTP request to Yahoo Finance.
    ticker_returns: dict[str, pd.Series] = {
        ticker: _fetch_daily_returns(ticker)
        for ticker in allocations
    }

    # Fetch 1 year of SPY daily returns separately.
    # SPY (SPDR S&P 500 ETF) is the standard benchmark for US equity portfolios.
    # It is used as the market proxy in the beta computation.
    # We reuse _fetch_daily_returns because SPY is just another ticker from
    # yfinance's perspective — no special handling required.
    spy_returns: pd.Series = _fetch_daily_returns("SPY")

    # Combine the per-ticker series into a single weighted portfolio return series.
    # This is the core time series that all eight metrics are computed from
    # (beta additionally uses spy_returns; concentration_risk uses weights directly).
    port_returns = _weighted_portfolio_returns(ticker_returns, weights)

    # ── Compute all eight metrics ─────────────────────────────────────────────
    # round() removes floating-point noise that would otherwise produce values
    # like 0.14200000000000002 in the output.

    expected_return = round(_compute_expected_return(port_returns), 4)
    # Annualised mean daily return, e.g. 0.142 = 14.2% per year.

    max_drawdown = round(_compute_max_drawdown(port_returns), 4)
    # Worst peak-to-trough loss in the 1-year window, e.g. -0.183 = -18.3%.

    sharpe_ratio = round(_compute_sharpe_ratio(port_returns, risk_free_rate), 4)
    # Excess return per unit of total volatility; > 1.0 is generally good.

    volatility = round(_compute_volatility(port_returns), 4)
    # Annualised standard deviation of portfolio returns, e.g. 0.182 = 18.2%.

    beta = round(_compute_beta(port_returns, spy_returns), 4)
    # Sensitivity to SPY moves; 1.0 = moves with the market, > 1.0 = amplifies moves.

    var_95 = round(_compute_var_95(port_returns), 4)
    # Worst expected daily loss at 95% confidence, e.g. -0.023 = -2.3% on a bad day.

    sortino_ratio = round(_compute_sortino_ratio(port_returns, risk_free_rate), 4)
    # Like Sharpe but only penalises downside volatility; typically higher than Sharpe.

    concentration_risk = round(_compute_concentration_risk(weights), 4)
    # Herfindahl index of allocation weights; lower = better diversification.

    # Return only the field this node writes.  LangGraph merges it into state.
    return {
        "portfolio_metrics": {
            "expected_return":    expected_return,
            "max_drawdown":       max_drawdown,
            "sharpe_ratio":       sharpe_ratio,
            "volatility":         volatility,
            "beta":               beta,
            "var_95":             var_95,
            "sortino_ratio":      sortino_ratio,
            "concentration_risk": concentration_risk,
        }
    }