# ---------------------------------------------------------------------------
# portfolio_upload.py — parses and validates an uploaded portfolio JSON file.
#
# Kept as a plain function (no Streamlit dependency) so it can be unit
# tested directly. Streamlit's AppTest framework has no way to simulate a
# file_uploader interaction, so this is the actual code path under test in
# tests/test_portfolio_upload.py — not a reimplementation of it.
# ---------------------------------------------------------------------------

import json

from security.validation import MAX_TICKERS, is_valid_ticker


def parse_uploaded_portfolio(raw: bytes) -> tuple[list[str], str | None]:
    """Parse and validate an uploaded portfolio JSON file.

    Expected shape: {"holdings": [{"ticker": "AAPL"}, {"ticker": "NVDA"}]}.
    `shares` (or any other per-holding field) is accepted but ignored, same
    as the original data/portfolio.json loader — only ticker symbols feed
    the graph.

    Returns (tickers, error_message). On success, error_message is None and
    tickers is a clean list of uppercase ticker symbols. On failure, tickers
    is [] and error_message is specific enough to show directly to the
    visitor who uploaded the file.
    """
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return [], "The uploaded file isn't valid UTF-8 text."

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [], "The uploaded file isn't valid JSON. Please check the format and try again."

    if not isinstance(data, dict) or "holdings" not in data:
        return [], 'The uploaded JSON is missing the required "holdings" key.'

    holdings = data["holdings"]
    if not isinstance(holdings, list) or len(holdings) == 0:
        return [], "The portfolio must contain at least one holding."

    tickers: list[str] = []
    seen: set[str] = set()
    duplicates: set[str] = set()
    missing_ticker_positions: list[int] = []
    invalid_format: list[str] = []

    for i, holding in enumerate(holdings):
        ticker_raw = holding.get("ticker") if isinstance(holding, dict) else None
        if not ticker_raw or not str(ticker_raw).strip():
            missing_ticker_positions.append(i + 1)
            continue

        ticker = str(ticker_raw).strip().upper()
        if ticker in seen:
            duplicates.add(ticker)
        seen.add(ticker)

        if not is_valid_ticker(ticker):
            invalid_format.append(ticker)

        tickers.append(ticker)

    if missing_ticker_positions:
        positions = ", ".join(str(p) for p in missing_ticker_positions)
        return [], (
            f"{len(missing_ticker_positions)} holding(s) are missing a "
            f"'ticker' field (position(s): {positions})."
        )

    if duplicates:
        return [], (
            f"Duplicate tickers found: {', '.join(sorted(duplicates))}. "
            "Each ticker should appear once."
        )

    if invalid_format:
        return [], (
            "These don't look like valid ticker symbols: "
            f"{', '.join(invalid_format)}."
        )

    if len(tickers) > MAX_TICKERS:
        return [], f"Please limit uploads to at most {MAX_TICKERS} holdings for this demo."

    return tickers, None
