import os
import sys
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Path fix — make the project root importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
from dotenv import load_dotenv

# Local dev: values come from a .env file (never committed — see .gitignore).
load_dotenv()

# Streamlit Community Cloud: secrets set in the app's dashboard are exposed
# via st.secrets rather than the process environment. The rest of this app
# (and the agent/ nodes it calls) reads config with os.environ.get(...) for
# a single consistent code path across local dev and the deployed app, so
# mirror any st.secrets values into os.environ here. Wrapped in try/except
# because st.secrets raises if no secrets.toml exists at all, which is the
# normal case for local dev.
try:
    for _key, _value in st.secrets.items():
        os.environ.setdefault(_key, str(_value))
except Exception:
    pass

from security.auth import require_auth
from security.errors import PortfolioAgentError
from security.identity import get_identifier
from security.rate_limit import try_consume, log_usage
from security.validation import parse_tickers, MAX_TICKERS

# ---------------------------------------------------------------------------
# Page configuration — must be the first Streamlit call in the script
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Portfolio Allocation Agent",
    page_icon="📊",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Design system — matches the portfolio site's visual language via custom
# CSS injection. Streamlit doesn't expose a first-class theming API for this
# level of control, so this targets Streamlit's internal data-testid hooks,
# which are reasonably stable but not a guaranteed public API — a future
# Streamlit version could shift some of these selectors.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --pa-bg: #EFF3F1;
        --pa-primary: #3F6659;
        --pa-primary-hover: #345448;
        --pa-tint: #DCE8E3;
        --pa-orange: #C97C4C;
        --pa-gold: #C9A24A;
        --pa-blue: #4A7A9E;
        --pa-text: #22312A;
        --pa-muted: #5B6B64;
        --pa-radius: 14px;
    }

    html, body, [class^="css"], [class*=" css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }

    .stApp {
        background-color: var(--pa-bg) !important;
    }

    [data-testid="stHeader"] {
        background-color: transparent !important;
    }

    .main .block-container, [data-testid="stMainBlockContainer"] {
        max-width: 720px;
        padding-top: 3rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3 {
        color: var(--pa-primary) !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em;
    }

    p, li, label, .stMarkdown, .stCaption {
        color: var(--pa-text);
    }

    [data-testid="stForm"],
    [data-testid="stStatusWidget"],
    [data-testid="stExpander"] {
        background-color: #ffffff;
        border: 1px solid var(--pa-tint) !important;
        border-radius: var(--pa-radius) !important;
        padding: 0.25rem 0.5rem;
    }

    .stButton > button, .stFormSubmitButton > button {
        background-color: var(--pa-primary) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: var(--pa-radius) !important;
        font-weight: 500 !important;
        padding: 0.6rem 1.2rem !important;
        transition: background-color 0.15s ease, transform 0.05s ease;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background-color: var(--pa-primary-hover) !important;
        color: #ffffff !important;
    }
    .stButton > button:active, .stFormSubmitButton > button:active {
        transform: scale(0.98);
    }

    .stTextInput input,
    .stTextArea textarea,
    .stSelectbox div[data-baseweb="select"] > div {
        border-radius: var(--pa-radius) !important;
        border: 1px solid var(--pa-tint) !important;
        background-color: #ffffff !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: var(--pa-primary) !important;
        box-shadow: 0 0 0 1px var(--pa-primary) !important;
    }

    [data-testid="stAlert"] {
        border-radius: var(--pa-radius) !important;
    }

    pre, code {
        border-radius: 10px !important;
        background-color: var(--pa-tint) !important;
    }

    hr {
        border-color: var(--pa-tint) !important;
    }

    .pa-gate {
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .pa-gate-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--pa-primary);
        margin-bottom: 0.4rem;
    }
    .pa-gate-subtitle {
        color: var(--pa-muted);
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Access gate — blocks everything below until the correct password is given.
# ---------------------------------------------------------------------------
require_auth()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📊 Portfolio Allocation Agent")
st.caption("Powered by Gemini 2.0 Flash · Built with LangGraph")

st.markdown(
    """
    This agent researches a portfolio of stocks and uses an LLM to recommend
    an optimal percentage allocation across holdings — backed by live market
    data, correlation analysis, and eight quantitative risk metrics.

    This is a public demo with limited usage (5 runs per visitor / 30 runs
    total per day) to keep it available for everyone.
    """
)

st.divider()

# ---------------------------------------------------------------------------
# Holdings — editable ticker list
# ---------------------------------------------------------------------------
st.subheader("Holdings")

tickers_raw = st.text_area(
    "Ticker symbols",
    value="AAPL, NVDA, TSLA",
    help=f"Comma or space separated. Up to {MAX_TICKERS} tickers.",
)

# ---------------------------------------------------------------------------
# Strategy selector
# ---------------------------------------------------------------------------
st.subheader("Strategy")

_STRATEGY_OPTIONS: dict[str, str] = {
    "Conservative — Capital Preservation": "conservative",
    "Balanced — Growth with Stability":    "balanced",
    "Aggressive — Maximum Growth":         "aggressive",
    "Income — Dividend & Yield Focus":     "income",
}

selected_label = st.selectbox(
    "Investment Strategy",
    options=list(_STRATEGY_OPTIONS.keys()),
    label_visibility="collapsed",
)

_STRATEGY_CAPTIONS: dict[str, str] = {
    "conservative": "Prioritises capital preservation and low volatility.",
    "balanced":     "Mixes growth and stability across the portfolio.",
    "aggressive":   "Maximises returns; accepts high short-term volatility.",
    "income":       "Focuses on dividend-paying, stable income stocks.",
}

selected_strategy = _STRATEGY_OPTIONS[selected_label]
st.caption(_STRATEGY_CAPTIONS[selected_strategy])

st.divider()

# ---------------------------------------------------------------------------
# Step labels shown while the graph runs, keyed by LangGraph node name.
# ---------------------------------------------------------------------------
_STEP_LABELS: dict[str, str] = {
    "portfolio_loader":     "✓ Loaded portfolio holdings",
    "macro_fetcher":        "✓ Fetched market conditions (VIX, 10Y yield, SPY)",
    "correlation_analyzer": "✓ Analyzed correlations between holdings",
    "allocation_decider":   "✓ Gemini decided the allocation",
    "portfolio_metrics":    "✓ Computed portfolio risk metrics",
    "output_formatter":     "✓ Formatted the report",
}


def _step_message(node_name: str, partial: dict) -> str:
    if node_name == "research_loop":
        ticker = next(iter(partial.get("research_results", {})), None)
        return f"✓ Researched {ticker}" if ticker else "✓ Researched a holding"
    return _STEP_LABELS.get(node_name, f"✓ Ran {node_name}")


# ---------------------------------------------------------------------------
# Trigger button
# ---------------------------------------------------------------------------
if st.button("▶ Run Agent", type="primary", use_container_width=True):

    tickers, ticker_error = parse_tickers(tickers_raw)

    if ticker_error:
        st.error(ticker_error)
        st.stop()

    identifier = get_identifier()
    allowed, limit_message = try_consume(identifier)

    if not allowed:
        st.warning(limit_message)
        st.stop()

    # Import the compiled graph lazily, after the gate/rate-limit checks
    # above have already passed. This module chain instantiates the Gemini
    # client, so importing it eagerly at the top of the file would mean a
    # misconfigured GOOGLE_API_KEY crashes the whole page before Streamlit
    # can render anything — including the password gate.
    from agent.graph import graph

    run_failed = False

    with st.status(
        "Running portfolio analysis — this can take 30–90 seconds…",
        expanded=True,
    ) as status:
        # Manually accumulate the state exactly as LangGraph's own reducers
        # would: every field is a plain overwrite except research_results,
        # which merges (see the `operator.or_` reducer in agent/state.py).
        # This lets us stream per-node progress *and* still end up with the
        # same final state that graph.invoke() would have returned.
        accumulated: dict = {
            "strategy": selected_strategy,
            "tickers": tickers,
            "research_results": {},
        }

        error_message = None

        try:
            for chunk in graph.stream(
                {"strategy": selected_strategy, "tickers": tickers}
            ):
                node_name, partial = next(iter(chunk.items()))
                for key, value in partial.items():
                    if key == "research_results":
                        accumulated["research_results"].update(value)
                    else:
                        accumulated[key] = value
                status.write(_step_message(node_name, partial))

            status.update(label="Analysis complete", state="complete", expanded=False)

        except PortfolioAgentError as exc:
            run_failed = True
            error_message = str(exc)
            status.update(label="Analysis failed", state="error", expanded=False)

        except Exception:
            run_failed = True
            error_message = (
                "Something went wrong while running the analysis. "
                "Please try again in a minute."
            )
            # Full traceback goes to the server-side process log only —
            # never shown to the visitor.
            traceback.print_exc()
            status.update(label="Analysis failed", state="error", expanded=False)

    # Rendered outside the status widget (rather than inside it) so the
    # error is always visible, not tucked inside a collapsed expander that
    # a visitor would have to know to click open.
    if error_message:
        st.error(error_message)

    log_usage(
        identifier,
        selected_strategy,
        tickers,
        status="error" if run_failed else "success",
    )

    if not run_failed:
        st.markdown(accumulated["final_output"])
