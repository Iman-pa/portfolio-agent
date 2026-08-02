# ---------------------------------------------------------------------------
# correlation_analyzer.py — LangGraph node
#
# Sits between macro_fetcher and research_loop in the graph.
# Fetches 90 days of daily closing prices for every ticker in the portfolio,
# builds a Pearson correlation matrix, and stores a summary in state so that
# allocation_decider can later reason about diversification risk.
# ---------------------------------------------------------------------------

import traceback
from datetime import datetime, timedelta
# datetime: used to get today's date as an end point for the price download.
# timedelta: used to subtract 90 days from today to get the start point.

import pandas as pd
# pandas is used for two things here:
#   1. The DataFrame of closing prices returned by yfinance.
#   2. .corr() — the built-in Pearson correlation matrix method on a DataFrame.

import yfinance as yf
# yfinance wraps Yahoo Finance's unofficial API.  yf.download() fetches OHLCV
# (Open, High, Low, Close, Volume) data for one or more tickers in one call.
from tenacity import retry, stop_after_attempt, wait_exponential

from agent.state import PortfolioState
# Import the shared TypedDict so type checkers and IDEs know the shape of state.

from security.errors import PortfolioAgentError
# Friendly exception type — the Streamlit UI catches this to show a clean
# error state instead of a raw traceback when a ticker is invalid or Yahoo
# Finance has no data for it.


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
def _download_prices(tickers: list[str], start_date, end_date) -> pd.DataFrame:
    """Thin, retryable wrapper around yf.download().

    Retries up to 3 times with exponential backoff (1s, 2s, 4s) — absorbs
    transient rate-limit/timeout failures from Yahoo's undocumented
    endpoints. Does nothing for a hard, sustained IP block.
    """
    return yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )


def correlation_analyzer(state: PortfolioState) -> dict:
    """Compute pairwise return correlations for every ticker in the portfolio.

    Reads:   state["tickers"]
    Writes:  state["correlation_context"]  (high_pairs, low_pairs, avg_correlation)
    Returns: a partial state dict with only the fields this node changes.
    """

    # -----------------------------------------------------------------------
    # STEP 1 — Read the list of tickers from shared state.
    # -----------------------------------------------------------------------
    tickers: list[str] = state["tickers"]
    # `tickers` was written by portfolio_loader at the very start of the graph.
    # It's a list of strings like ["AAPL", "NVDA", "TSLA"].

    # -----------------------------------------------------------------------
    # STEP 2 — Define the 90-day date window for the price download.
    # -----------------------------------------------------------------------
    end_date = datetime.today()
    # datetime.today() returns the current local date and time.
    # yfinance accepts datetime objects as start/end arguments directly.

    start_date = end_date - timedelta(days=90)
    # timedelta(days=90) creates a duration object representing 90 calendar days.
    # Subtracting it from today gives the date 90 days ago, which is our window start.
    # 90 days is enough history to produce a statistically meaningful correlation
    # (~60 trading days) while staying recent enough to reflect current relationships.

    # -----------------------------------------------------------------------
    # STEP 3 — Download adjusted daily closing prices for all tickers at once.
    # -----------------------------------------------------------------------
    try:
        raw = _download_prices(tickers, start_date, end_date)
    except Exception as exc:
        # Network hiccups or Yahoo Finance outages surface here as arbitrary
        # exceptions from the underlying HTTP client — wrap them in a message
        # that's safe to show to a site visitor. Logged first so the real
        # cause is visible in the server's own logs (Render/Streamlit Cloud
        # capture stderr) — without this, only the friendly message ever
        # surfaces anywhere.
        print(f"[correlation_analyzer] yf.download failed for {tickers}: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise PortfolioAgentError(
            "We couldn't fetch market data right now. This is usually a "
            "temporary issue with the data provider — please try again in "
            "a minute."
        ) from exc
    # `raw` is a pandas DataFrame.  When multiple tickers are requested, the
    # columns are a two-level MultiIndex: (field, ticker).
    # Example columns: ("Close","AAPL"), ("Close","NVDA"), ("Open","AAPL"), ...

    if raw.empty:
        raise PortfolioAgentError(
            "We couldn't find market data for any of the tickers you "
            "entered. Please check the symbols and try again."
        )

    # -----------------------------------------------------------------------
    # STEP 4 — Isolate the "Close" column for each ticker.
    # -----------------------------------------------------------------------
    prices: pd.DataFrame = raw["Close"]
    # Indexing a MultiIndex DataFrame with a first-level key ("Close") drops
    # that level and returns a plain DataFrame whose columns are the ticker symbols.
    # Shape: (n_trading_days, n_tickers).

    if isinstance(prices, pd.Series):
        # Safety net: if the portfolio has exactly one ticker, yfinance *may*
        # return a Series instead of a single-column DataFrame, depending on
        # the library version.  Converting it to a DataFrame keeps all
        # downstream code uniform and prevents .corr() from failing.
        prices = prices.to_frame(name=tickers[0])
        # to_frame() wraps the Series in a DataFrame; name= sets the column label
        # to the ticker symbol so corr_matrix.loc[t1, t2] still works by name.

    # Any ticker yfinance couldn't resolve comes back as an all-NaN column.
    # Surface exactly which symbol is bad rather than silently dropping it
    # (dropna() below would otherwise wipe out every row if even one column
    # is entirely NaN, producing a confusing "no data" error later).
    missing_tickers = [t for t in prices.columns if prices[t].isna().all()]
    if missing_tickers:
        raise PortfolioAgentError(
            "We couldn't find market data for: "
            f"{', '.join(missing_tickers)}. Please check the ticker "
            "symbol(s) and try again."
        )

    prices = prices.dropna()
    # Drop any row (trading day) where at least one ticker has a missing value.
    # NaN values occur when:
    #   - A ticker was not yet listed on that date.
    #   - Yahoo Finance had a data gap for that day.
    # Using only rows that are complete for *all* tickers ensures the correlation
    # values are computed on the same set of trading days for every pair.

    # -----------------------------------------------------------------------
    # STEP 5 — Compute the Pearson correlation matrix.
    # -----------------------------------------------------------------------
    corr_matrix: pd.DataFrame = prices.corr()
    # DataFrame.corr() computes the Pearson product-moment correlation coefficient
    # for every pair of columns using the formula:
    #   r(X,Y) = cov(X,Y) / (std(X) * std(Y))
    # The result is a symmetric (n_tickers × n_tickers) DataFrame:
    #   - Diagonal is always 1.0 (a series is perfectly correlated with itself).
    #   - Off-diagonal values range from -1.0 (perfect inverse) to +1.0 (perfect sync).
    # Index and columns are both the ticker symbols, so label-based lookup works:
    #   corr_matrix.loc["AAPL", "NVDA"]

    # -----------------------------------------------------------------------
    # STEP 6 — Walk the upper triangle and classify each unique pair.
    # -----------------------------------------------------------------------
    high_pairs: list[list[str]] = []
    # Will hold [ticker_a, ticker_b] pairs whose correlation > 0.7.
    # A threshold of 0.7 is a widely used "strong correlation" cut-off in finance.
    # Stocks in this bucket tend to move together, so holding both adds less
    # diversification benefit than it appears.

    low_pairs: list[list[str]] = []
    # Will hold [ticker_a, ticker_b] pairs whose correlation < 0.3.
    # Stocks in this bucket move relatively independently, which is desirable
    # for diversification — losses in one are not amplified by the other.

    all_values: list[float] = []
    # Accumulates every off-diagonal correlation value so we can average them.

    for i in range(len(tickers)):
        # Outer loop over every ticker's position index in the list.
        for j in range(i + 1, len(tickers)):
            # Inner loop starts at i+1 so we only visit the upper triangle:
            #   (i=0,j=1), (i=0,j=2), (i=1,j=2), ...
            # This avoids counting each pair twice and skips the diagonal (i==j).

            t1 = tickers[i]   # first ticker symbol in the pair, e.g. "AAPL"
            t2 = tickers[j]   # second ticker symbol in the pair, e.g. "NVDA"

            value = float(corr_matrix.loc[t1, t2])
            # .loc uses label-based indexing on both axes — it looks up the row
            # named t1 and the column named t2 in the correlation matrix.
            # float() converts from numpy.float64 to a native Python float so
            # the entire correlation_context dict is JSON-serialisable without
            # needing a custom encoder.

            all_values.append(value)
            # Store this value so we can compute the portfolio-wide average below.

            if value > 0.7:
                high_pairs.append([t1, t2])
                # Store as a two-element list (not a tuple) because lists are
                # JSON-serialisable; tuples are not.
            elif value < 0.3:
                low_pairs.append([t1, t2])
                # Same reasoning: list over tuple for JSON compatibility.
            # Values between 0.3 and 0.7 are considered "moderate" correlation
            # and are not highlighted in either bucket — they are still captured
            # in avg_correlation.

    # -----------------------------------------------------------------------
    # STEP 7 — Compute the portfolio-wide average pairwise correlation.
    # -----------------------------------------------------------------------
    if all_values:
        avg_correlation = round(sum(all_values) / len(all_values), 4)
        # Arithmetic mean of all unique pairwise correlations.
        # sum() + len() is used instead of statistics.mean() to avoid an extra
        # import for a trivial one-liner.
        # round(..., 4) keeps four decimal places — enough precision for an LLM
        # prompt without introducing floating-point noise like 0.72300000001.
    else:
        avg_correlation = 0.0
        # Edge case: the portfolio contains only one ticker, so no pairs exist
        # and all_values is empty.  We default to 0.0 instead of raising a
        # ZeroDivisionError so the graph run continues gracefully.

    # -----------------------------------------------------------------------
    # STEP 8 — Return only the field this node writes.
    # -----------------------------------------------------------------------
    return {
        "correlation_context": {
            "high_pairs": high_pairs,
            # e.g. [["AAPL", "NVDA"], ["NVDA", "MSFT"]]
            "low_pairs": low_pairs,
            # e.g. [["AAPL", "TSLA"]]
            "avg_correlation": avg_correlation,
            # e.g. 0.6312
        }
    }
    # LangGraph merges this partial dict into the shared PortfolioState.
    # Returning only "correlation_context" leaves every other field untouched.
