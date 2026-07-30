# ---------------------------------------------------------------------------
# validation.py — parses and validates user-entered ticker symbols.
#
# This only checks *format*, not whether the symbol actually exists — actual
# existence is confirmed when the agent fetches market data, and a bad
# symbol surfaces there as a friendly PortfolioAgentError (see
# agent/nodes/correlation_analyzer.py and portfolio_metrics.py). Format
# validation here exists to reject obvious junk early and to bound the size
# of a single run.
# ---------------------------------------------------------------------------

import re

_TICKER_RE = re.compile(r"^[A-Z]{1,6}([.\-][A-Z]{1,2})?$")

MAX_TICKERS = 8


def parse_tickers(raw_input: str) -> tuple[list[str], str | None]:
    """Parse a free-text ticker list into a clean, validated list.

    Accepts tickers separated by commas, whitespace, or newlines. Returns
    (tickers, error_message) — error_message is None on success, otherwise
    tickers is an empty list.
    """
    candidates = [
        t.strip().upper()
        for t in re.split(r"[,\s]+", raw_input.strip())
        if t.strip()
    ]

    if not candidates:
        return [], "Enter at least one ticker symbol."

    # Preserve order while dropping duplicates.
    seen = set()
    tickers = []
    for t in candidates:
        if t not in seen:
            seen.add(t)
            tickers.append(t)

    if len(tickers) > MAX_TICKERS:
        return [], f"Please enter at most {MAX_TICKERS} tickers for this demo."

    invalid = [t for t in tickers if not _TICKER_RE.match(t)]
    if invalid:
        return [], (
            f"These don't look like valid ticker symbols: {', '.join(invalid)}. "
            "Use standard symbols like AAPL, NVDA, or BRK.B."
        )

    return tickers, None
