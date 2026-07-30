# ---------------------------------------------------------------------------
# validation.py — shared ticker-format validation.
#
# Used by both the Build Portfolio tab (tickers come from a fixed universe,
# so they're always well-formed) and the Upload Portfolio tab (tickers come
# from arbitrary user JSON, so format needs checking). Existence on the
# market isn't checked here — that's confirmed when the agent fetches data,
# surfacing as a friendly PortfolioAgentError (see
# agent/nodes/correlation_analyzer.py and portfolio_metrics.py).
# ---------------------------------------------------------------------------

import re

TICKER_RE = re.compile(r"^[A-Z]{1,6}([.\-][A-Z]{1,2})?$")

MAX_TICKERS = 8


def is_valid_ticker(ticker: str) -> bool:
    return bool(TICKER_RE.match(ticker))
