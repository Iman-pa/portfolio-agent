# ---------------------------------------------------------------------------
# Tests for the Upload Portfolio tab's validation logic.
#
# Streamlit's AppTest framework (streamlit.testing.v1) has no API to
# simulate a file_uploader interaction — there is no `at.file_uploader`
# accessor and no way to inject an UploadedFile — so the upload flow can't
# be driven end-to-end through the UI in an automated test. These tests
# instead call security/portfolio_upload.parse_uploaded_portfolio directly:
# it's the exact same function frontend/app.py calls with the uploaded
# file's bytes, so this is still testing the real production validation
# logic, just below the widget rather than through it.
# ---------------------------------------------------------------------------

import json

from security.portfolio_upload import parse_uploaded_portfolio


def test_valid_portfolio_parses_successfully():
    raw = json.dumps({
        "holdings": [
            {"ticker": "AAPL", "shares": 10},
            {"ticker": "NVDA", "shares": 5},
            {"ticker": "TSLA", "shares": 8},
        ]
    }).encode("utf-8")

    tickers, error = parse_uploaded_portfolio(raw)

    assert error is None
    assert tickers == ["AAPL", "NVDA", "TSLA"]


def test_lowercase_tickers_are_normalised_to_uppercase():
    raw = json.dumps({"holdings": [{"ticker": "aapl"}]}).encode("utf-8")

    tickers, error = parse_uploaded_portfolio(raw)

    assert error is None
    assert tickers == ["AAPL"]


def test_malformed_json_shows_specific_error():
    raw = b"{not valid json at all"

    tickers, error = parse_uploaded_portfolio(raw)

    assert tickers == []
    assert error is not None
    assert "valid JSON" in error


def test_missing_holdings_key_shows_specific_error():
    raw = json.dumps({"tickers": ["AAPL"]}).encode("utf-8")

    tickers, error = parse_uploaded_portfolio(raw)

    assert tickers == []
    assert "holdings" in error


def test_empty_holdings_list_shows_specific_error():
    raw = json.dumps({"holdings": []}).encode("utf-8")

    tickers, error = parse_uploaded_portfolio(raw)

    assert tickers == []
    assert "at least one holding" in error


def test_holding_missing_ticker_field_shows_specific_error():
    raw = json.dumps({
        "holdings": [{"ticker": "AAPL"}, {"shares": 5}]
    }).encode("utf-8")

    tickers, error = parse_uploaded_portfolio(raw)

    assert tickers == []
    assert "missing a 'ticker' field" in error
    assert "position(s): 2" in error


def test_duplicate_tickers_show_specific_error():
    raw = json.dumps({
        "holdings": [{"ticker": "AAPL"}, {"ticker": "aapl"}]
    }).encode("utf-8")

    tickers, error = parse_uploaded_portfolio(raw)

    assert tickers == []
    assert "Duplicate tickers" in error
    assert "AAPL" in error


def test_invalid_ticker_format_shows_specific_error():
    raw = json.dumps({"holdings": [{"ticker": "$$$NOT-A-TICKER$$$"}]}).encode("utf-8")

    tickers, error = parse_uploaded_portfolio(raw)

    assert tickers == []
    assert "don't look like valid ticker symbols" in error


def test_too_many_holdings_shows_specific_error():
    # 9 distinct, format-valid tickers (MAX_TICKERS is 8) so this actually
    # exercises the count check rather than the format check.
    nine_valid_tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III"]
    raw = json.dumps({
        "holdings": [{"ticker": t} for t in nine_valid_tickers]
    }).encode("utf-8")

    tickers, error = parse_uploaded_portfolio(raw)

    assert tickers == []
    assert "at most 8 holdings" in error


def test_non_dict_holding_entry_is_treated_as_missing_ticker():
    raw = json.dumps({"holdings": ["AAPL"]}).encode("utf-8")

    tickers, error = parse_uploaded_portfolio(raw)

    assert tickers == []
    assert "missing a 'ticker' field" in error
