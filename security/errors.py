# ---------------------------------------------------------------------------
# errors.py — a friendly, user-facing exception type.
#
# Nodes raise PortfolioAgentError with a message that is safe to show
# directly to a site visitor (no stack traces, no internal details). The
# Streamlit UI catches this type specifically to render a clean error state,
# falling back to a generic message for anything unexpected.
# ---------------------------------------------------------------------------


class PortfolioAgentError(Exception):
    """Raised for expected, user-facing failure conditions (e.g. an invalid
    ticker symbol or a failed market data fetch)."""
