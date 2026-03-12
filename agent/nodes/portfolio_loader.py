import json
from pathlib import Path

from agent.state import PortfolioState

# Resolve the path to data/portfolio.json relative to this file's location.
# __file__ is the absolute path of portfolio_loader.py at runtime.
# .parent gives agent/nodes/, .parent again gives agent/, .parent again gives
# the project root, then we descend into data/portfolio.json.
_PORTFOLIO_PATH = Path(__file__).parent.parent.parent / "data" / "portfolio.json"


def portfolio_loader(state: PortfolioState) -> dict:
    """Read portfolio.json and prepare the state for the research loop.

    Returns only the two fields this node owns: `tickers` (the list of stock
    symbols to research) and `current_ticker_index` (reset to 0 so the
    research loop always starts from the first ticker).
    """
    # Open the JSON file using the pre-computed absolute path.
    # encoding="utf-8" is explicit best-practice — avoids surprises on
    # Windows where the default encoding can vary by system locale.
    with open(_PORTFOLIO_PATH, encoding="utf-8") as f:
        # json.load() parses the file object directly into a Python dict.
        # We name it `portfolio` to match the top-level object in the file.
        portfolio = json.load(f)

    # portfolio["holdings"] is a list of dicts like {"ticker": "AAPL", "shares": 10}.
    # The list comprehension walks each holding and pulls out only the
    # "ticker" string, giving us ["AAPL", "NVDA", "TSLA"].
    tickers = [holding["ticker"] for holding in portfolio["holdings"]]

    # Return only the keys this node changes.  LangGraph merges this partial
    # dict into the full state — anything we don't mention is left untouched.
    # `current_ticker_index: 0` resets the loop counter so research_loop
    # always starts at tickers[0] regardless of any prior state.
    return {
        "tickers": tickers,
        "current_ticker_index": 0,
    }
