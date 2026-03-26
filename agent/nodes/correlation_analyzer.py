# ---------------------------------------------------------------------------
# correlation_analyzer.py — LangGraph node
#
# Sits between macro_fetcher and research_loop in the graph.
# Fetches 90 days of daily closing prices for every ticker in the portfolio,
# builds a Pearson correlation matrix, and stores a summary in state so that
# allocation_decider can later reason about diversification risk.
# ---------------------------------------------------------------------------

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

from agent.state import PortfolioState
# Import the shared TypedDict so type checkers and IDEs know the shape of state.


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
    raw = yf.download(
        tickers=tickers,
        # Pass the full list — yfinance fetches all tickers in a single HTTP
        # request, which is faster and rate-limit-friendlier than one call per ticker.
        start=start_date,
        # Inclusive start date for the price series.
        end=end_date,
        # End date is exclusive in yfinance — it returns data *up to but not
        # including* this date, so today's partial candle is excluded automatically.
        auto_adjust=True,
        # Replace raw OHLCV with adjusted values that account for stock splits
        # and dividends.  This is important for correlation: a nominal split
        # creates a sudden price drop that would skew the correlation if unadjusted.
        progress=False,
        # Suppress yfinance's tqdm download-progress bar so it doesn't pollute
        # the server logs or Streamlit output.
    )
    # `raw` is a pandas DataFrame.  When multiple tickers are requested, the
    # columns are a two-level MultiIndex: (field, ticker).
    # Example columns: ("Close","AAPL"), ("Close","NVDA"), ("Open","AAPL"), ...

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
