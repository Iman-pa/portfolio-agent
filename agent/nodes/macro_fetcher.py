import traceback

import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

from agent.state import PortfolioState
from security.errors import PortfolioAgentError

# ---------------------------------------------------------------------------
# Mapping: plain-English key → Yahoo Finance ticker symbol
# ---------------------------------------------------------------------------
# Keeping this as a module-level constant means we only have to update one
# place if we ever want to swap or add an indicator.
# ^VIX  — CBOE Volatility Index: measures expected 30-day market volatility.
#          High values (>30) signal fear; low values (<15) signal complacency.
# ^TNX  — CBOE 10-Year Treasury Note Yield Index: Yahoo reports this as the
#          yield × 10, so the raw value 46.0 means 4.60 %.  We divide by 10
#          inside _fetch_price() to return the true yield percentage.
# SPY   — SPDR S&P 500 ETF: a liquid proxy for overall US market momentum.
_INDICATORS: dict[str, str] = {
    "vix":       "^VIX",
    "yield_10y": "^TNX",
    "spy_price": "SPY",
}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
def _fetch_price(yahoo_ticker: str) -> float:
    """Return the most recent closing price for a Yahoo Finance ticker.

    Tries fast_info['last_price'] first because it is a lightweight property
    that skips downloading a full OHLCV history.  Falls back to
    .history(period='1d') when fast_info is unavailable (some index tickers
    behave inconsistently depending on market hours and the yfinance version).

    Retries up to 3 times with exponential backoff (1s, 2s, 4s) before
    giving up — yfinance calls Yahoo's undocumented endpoints, which are
    known to intermittently rate-limit or reject requests (especially from
    cloud/datacenter IP ranges); a short retry absorbs transient failures of
    that kind. It does nothing for a hard, sustained IP block, which needs a
    different fix entirely (see the caller's error message).
    """
    # yf.Ticker() creates a lazy handle — no network request happens yet.
    ticker_obj = yf.Ticker(yahoo_ticker)

    try:
        # fast_info is a cached property that fetches a minimal quote summary.
        # 'last_price' is the most recent trade price (or previous close if
        # the market is currently closed).
        price = ticker_obj.fast_info["last_price"]

        # fast_info can return None if the exchange has not yet opened or if
        # yfinance cannot resolve the symbol.  Falling back keeps the node
        # robust across different times of day and yfinance versions.
        if price is None:
            raise ValueError("fast_info returned None")

    except (KeyError, ValueError):
        # .history() downloads a proper OHLCV DataFrame.  period='1d' fetches
        # only today's (or the most recent trading day's) candles, keeping the
        # download as small as possible.
        hist = ticker_obj.history(period="1d")

        # hist["Close"] is a pandas Series indexed by datetime.
        # .iloc[-1] selects the last row — the most recent closing price.
        price = float(hist["Close"].iloc[-1])

    # Always return a plain Python float.  yfinance can return numpy.float64,
    # which is not JSON-serialisable and can cause issues downstream.
    return float(price)


def macro_fetcher(state: PortfolioState) -> dict:
    """Fetch market-wide indicators and store them in macro_context.

    Called once, immediately after portfolio_loader, before the research loop
    begins.  The macro data is available to every subsequent node via state.
    """
    macro_context: dict[str, float] = {}

    for key, yahoo_symbol in _INDICATORS.items():
        try:
            raw_price = _fetch_price(yahoo_symbol)
        except Exception as exc:
            # The friendly PortfolioAgentError below is what the visitor
            # sees; without this, the real cause (rate limit, IP block,
            # timeout, schema change, ...) is never visible anywhere,
            # including in the server's own logs. print() goes to stderr,
            # which Render (and Streamlit Cloud) capture in their log
            # viewers.
            print(f"[macro_fetcher] yfinance fetch failed for {yahoo_symbol!r}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            raise PortfolioAgentError(
                "We couldn't fetch current market conditions right now. "
                "This is usually a temporary issue with the data provider "
                "— please try again in a minute."
            ) from exc

        if key == "yield_10y":
            # ^TNX quotes the yield multiplied by 10 (e.g. 46.0 = 4.60 %).
            # Dividing by 10 converts it back to the conventional percentage
            # that finance professionals expect (e.g. 4.60).
            macro_context[key] = round(raw_price / 10, 3)
        else:
            # For VIX and SPY the raw price is already the correct value.
            # round() to 2 decimal places trims floating-point noise.
            macro_context[key] = round(raw_price, 2)

    # Return only the field this node writes.  LangGraph merges this partial
    # dict into the shared state — tickers, current_ticker_index, and all
    # other fields are left exactly as portfolio_loader set them.
    return {"macro_context": macro_context}
