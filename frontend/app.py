import os
import sys
import time
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

# LangSmith tracing is controlled entirely by these env vars, read directly
# by the langchain/langsmith SDKs — no code here turns it on or off by
# itself. It must stay off whenever LANGCHAIN_API_KEY is absent:
# LANGCHAIN_TRACING_V2=true with no key doesn't just skip tracing, it makes
# every graph run try to send traces and fail with a 401, filling the
# production logs with noise. Enforced here in code rather than trusting
# every deployment config (render.yaml, Streamlit Cloud secrets, .env) to
# always set this pair consistently.
if not os.environ.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

from security.auth import require_auth
from security.errors import PortfolioAgentError
from security.identity import get_identifier
from security.portfolio_upload import parse_uploaded_portfolio
from security.rate_limit import try_consume, log_usage
from security.validation import MAX_TICKERS

# ---------------------------------------------------------------------------
# Stock universe — Build Portfolio tab lets visitors pick from this fixed
# list rather than free-typing tickers, so every selection is guaranteed
# well-formed. (ticker, company name) tuples so the multiselect can show
# both.
# ---------------------------------------------------------------------------
STOCK_UNIVERSE: list[tuple[str, str]] = [
    ("AAPL", "Apple Inc."),
    ("MSFT", "Microsoft Corporation"),
    ("NVDA", "NVIDIA Corporation"),
    ("GOOGL", "Alphabet Inc."),
    ("AMZN", "Amazon.com, Inc."),
    ("META", "Meta Platforms, Inc."),
    ("TSLA", "Tesla, Inc."),
    ("JPM", "JPMorgan Chase & Co."),
    ("JNJ", "Johnson & Johnson"),
    ("V", "Visa Inc."),
    ("WMT", "Walmart Inc."),
    ("PG", "Procter & Gamble Co."),
    ("MA", "Mastercard Incorporated"),
    ("UNH", "UnitedHealth Group Incorporated"),
    ("HD", "The Home Depot, Inc."),
    ("BAC", "Bank of America Corporation"),
    ("XOM", "Exxon Mobil Corporation"),
    ("CVX", "Chevron Corporation"),
    ("ABBV", "AbbVie Inc."),
    ("MRK", "Merck & Co., Inc."),
    ("PFE", "Pfizer Inc."),
    ("KO", "The Coca-Cola Company"),
    ("PEP", "PepsiCo, Inc."),
    ("COST", "Costco Wholesale Corporation"),
    ("AVGO", "Broadcom Inc."),
    ("CSCO", "Cisco Systems, Inc."),
    ("ACN", "Accenture plc"),
    ("TMO", "Thermo Fisher Scientific Inc."),
    ("ABT", "Abbott Laboratories"),
    ("NKE", "NIKE, Inc."),
    ("DIS", "The Walt Disney Company"),
    ("ADBE", "Adobe Inc."),
    ("CRM", "Salesforce, Inc."),
    ("NFLX", "Netflix, Inc."),
    ("INTC", "Intel Corporation"),
    ("AMD", "Advanced Micro Devices, Inc."),
    ("QCOM", "QUALCOMM Incorporated"),
    ("TXN", "Texas Instruments Incorporated"),
    ("HON", "Honeywell International Inc."),
    ("UPS", "United Parcel Service, Inc."),
    ("CAT", "Caterpillar Inc."),
    ("DE", "Deere & Company"),
    ("GS", "The Goldman Sachs Group, Inc."),
    ("MS", "Morgan Stanley"),
    ("BLK", "BlackRock, Inc."),
    ("SPGI", "S&P Global Inc."),
    ("AXP", "American Express Company"),
    ("SYK", "Stryker Corporation"),
    ("ISRG", "Intuitive Surgical, Inc."),
    ("LMT", "Lockheed Martin Corporation"),
]

# Soft wall-clock budget for a full analysis run, checked between completed
# graph nodes. Not a hard preemptive timeout (a single node's own network
# call — e.g. a hung yfinance request — can't be interrupted mid-call
# without extra process/thread machinery), but it stops the UI from waiting
# forever and gives a clear, specific error instead.
RUN_TIMEOUT_SECONDS = 180

# ---------------------------------------------------------------------------
# Page configuration — must be the first Streamlit call in the script
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Portfolio Allocation Agent",
    page_icon="📊",
    layout="wide",
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
        max-width: 1000px;
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
    [data-testid="stExpander"],
    [data-testid="stFileUploaderDropzone"] {
        background-color: #ffffff;
        border: 1px solid var(--pa-tint) !important;
        border-radius: var(--pa-radius) !important;
        padding: 0.25rem 0.5rem;
    }

    [data-testid="stTabs"] button[data-baseweb="tab"] {
        border-radius: var(--pa-radius) var(--pa-radius) 0 0 !important;
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
    .stButton > button:disabled {
        background-color: var(--pa-tint) !important;
        color: var(--pa-muted) !important;
    }

    .stTextInput input,
    .stTextArea textarea,
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] > div,
    .stMultiSelect div[data-baseweb="select"] > div {
        border-radius: var(--pa-radius) !important;
        border: 1px solid var(--pa-tint) !important;
        background-color: #ffffff !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus {
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

st.markdown(
    """
    Upload a portfolio or build one from a list of well-known stocks, pick
    an investment strategy, and this agent researches each holding and uses
    an LLM to recommend an optimal allocation — backed by live market data,
    correlation analysis, and eight quantitative risk metrics.

    This is a public demo with limited usage (5 runs per visitor / 30 runs
    total per day) to keep it available for everyone. Nothing you enter is
    stored beyond this session, other than an operational log of run
    timestamps and ticker symbols used to monitor demo usage.
    """
)

st.divider()

# ---------------------------------------------------------------------------
# Strategy selector — placed above the tabs so it's visible regardless of
# which input mode (Upload / Build) the visitor is using.
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
# Portfolio input — Upload Portfolio / Build Portfolio tabs.
#
# `portfolio_version` is bumped by Clear Portfolio to force every widget
# below (uploader, multiselect, per-ticker weight inputs) to remount with
# fresh defaults — deleting a widget's session_state entry isn't reliable
# for file_uploader across Streamlit versions, but changing its key is.
# ---------------------------------------------------------------------------
if "portfolio_version" not in st.session_state:
    st.session_state["portfolio_version"] = 0

_ver = st.session_state["portfolio_version"]

st.subheader("Portfolio")

upload_tab, build_tab = st.tabs(["Upload Portfolio", "Build Portfolio"])

upload_tickers: list[str] | None = None

with upload_tab:
    st.markdown(
        "Upload a JSON file with a `holdings` list. Each holding needs a "
        "`ticker` field; other fields are ignored. Example:"
    )
    st.code(
        '{\n'
        '  "holdings": [\n'
        '    {"ticker": "AAPL"},\n'
        '    {"ticker": "NVDA"},\n'
        '    {"ticker": "TSLA"}\n'
        '  ]\n'
        '}',
        language="json",
    )

    uploaded_file = st.file_uploader(
        "Portfolio JSON",
        type=["json"],
        key=f"uploader_{_ver}",
        help=f"Up to {MAX_TICKERS} holdings.",
    )

    if uploaded_file is not None:
        parsed_tickers, upload_error = parse_uploaded_portfolio(uploaded_file.getvalue())
        if upload_error:
            st.error(upload_error)
        else:
            upload_tickers = parsed_tickers
            st.success(f"{len(upload_tickers)} holding(s) loaded.")
            st.dataframe(
                [{"Ticker": t} for t in upload_tickers],
                hide_index=True,
                use_container_width=True,
            )

build_selected: list[str] = []
build_weights: dict[str, float] = {}
build_total = 0.0

with build_tab:
    _label_to_ticker = {f"{t} — {n}": t for t, n in STOCK_UNIVERSE}
    selected_labels = st.multiselect(
        "Select stocks",
        options=list(_label_to_ticker.keys()),
        max_selections=MAX_TICKERS,
        key=f"stock_select_{_ver}",
        help=f"Up to {MAX_TICKERS} stocks.",
    )
    build_selected = [_label_to_ticker[label] for label in selected_labels]

    # number_input's `value=` argument only seeds a widget the *first* time
    # its key appears — it's silently ignored on later reruns even if a
    # different value= is passed. Computing `default_weight` from the
    # current selection size and passing it as value= therefore only gives
    # a correct equal split for whichever ticker happens to be selected
    # first; every ticker added afterward locks in based on the count at
    # *its* first render, so totals didn't actually default to 100%.
    # Instead, explicitly rebalance every selected ticker to an equal split
    # by writing straight into session_state whenever the selection set
    # changes — writing to session_state before a widget with that key is
    # instantiated *is* respected as its value, unlike value=.
    _selection_key = f"build_selection_{_ver}"
    if st.session_state.get(_selection_key) != build_selected:
        if build_selected:
            equal_weight = round(100 / len(build_selected), 2)
            for ticker in build_selected:
                st.session_state[f"weight_{_ver}_{ticker}"] = equal_weight
        st.session_state[_selection_key] = build_selected

    if build_selected:
        st.caption(
            "Set each holding's current weight below. The agent computes "
            "its own recommended allocation separately — these weights "
            "only define the starting portfolio. Changing your stock "
            "selection rebalances weights back to an equal split."
        )
        for ticker in build_selected:
            build_weights[ticker] = st.number_input(
                f"{ticker} weight (%)",
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                key=f"weight_{_ver}_{ticker}",
            )
        build_total = round(sum(build_weights.values()), 2)

        if abs(build_total - 100) <= 0.01:
            st.success(f"Total: {build_total}% ✓")
        elif build_total < 100:
            st.warning(f"Total: {build_total}% — {round(100 - build_total, 2)}% remaining")
        else:
            st.warning(f"Total: {build_total}% — {round(build_total - 100, 2)}% over 100%")
    else:
        st.caption("Select stocks above to assign weights.")

build_valid = bool(build_selected) and abs(build_total - 100) <= 0.01

# ---------------------------------------------------------------------------
# Resolve which input mode is active. Both tabs' widgets always run (that's
# how st.tabs works — every tab body executes on every rerun, only the
# display is hidden), so if both are simultaneously valid we need a
# deterministic, disclosed tie-break rather than silently picking one.
# ---------------------------------------------------------------------------
if upload_tickers:
    active_tickers = upload_tickers
elif build_valid:
    active_tickers = build_selected
else:
    active_tickers = None

portfolio_ready = active_tickers is not None

if upload_tickers and build_valid:
    st.info(
        "Both an uploaded file and a built portfolio are ready — the "
        "uploaded file will be used. Remove it (✕ on the file) to use "
        "your built portfolio instead."
    )

st.divider()

# ---------------------------------------------------------------------------
# Step labels shown while the graph runs, keyed by LangGraph node name.
# ---------------------------------------------------------------------------
_STEP_LABELS: dict[str, str] = {
    "portfolio_loader":     "✓ Loaded portfolio holdings",
    "macro_fetcher":        "✓ Fetched market data (VIX, 10Y yield, SPY)",
    "correlation_analyzer": "✓ Computed correlations between holdings",
    "allocation_decider":   "✓ Decided allocation",
    "portfolio_metrics":    "✓ Computed portfolio metrics",
    "output_formatter":     "✓ Formatted the report",
}


def _step_message(node_name: str, partial: dict) -> str:
    if node_name == "research_loop":
        ticker = next(iter(partial.get("research_results", {})), None)
        return f"✓ Researched {ticker}" if ticker else "✓ Researched a holding"
    return _STEP_LABELS.get(node_name, f"✓ Ran {node_name}")


def _clear_portfolio() -> None:
    """Reset every widget in the Portfolio section — uploader, multiselect,
    and all per-ticker weight inputs — plus any run-in-progress flag.

    Deleting a widget's session_state entry doesn't reliably reset a
    file_uploader across Streamlit versions, so instead every widget key in
    this section is suffixed with `portfolio_version`; bumping that version
    forces all of them to remount fresh on the next run. Old entries under
    the previous version are deleted outright rather than left orphaned.

    Wired up as the Clear Portfolio button's `on_click`, not as an `if
    st.button(...)` block calling `st.rerun()`. An on_click callback runs
    before the script's main body re-executes, so the widgets above (which
    read `portfolio_version` when building their keys) see the bumped
    version and render empty on the very same rerun the click triggers —
    no extra rerun needed.
    """
    old_ver = st.session_state.get("portfolio_version", 0)
    stale_prefixes = (
        f"uploader_{old_ver}",
        f"stock_select_{old_ver}",
        f"weight_{old_ver}_",
        f"build_selection_{old_ver}",
    )
    for key in list(st.session_state.keys()):
        if key.startswith(stale_prefixes):
            del st.session_state[key]
    st.session_state["portfolio_version"] = old_ver + 1
    st.session_state.pop("run_in_progress", None)


# ---------------------------------------------------------------------------
# Trigger buttons
# ---------------------------------------------------------------------------
run_in_progress = st.session_state.get("run_in_progress", False)

col_analyze, col_clear = st.columns([3, 1])

with col_analyze:
    analyze_clicked = st.button(
        "Analyze Portfolio",
        type="primary",
        use_container_width=True,
        disabled=not portfolio_ready or run_in_progress,
    )

with col_clear:
    st.button("Clear Portfolio", use_container_width=True, on_click=_clear_portfolio)

if analyze_clicked:
    st.session_state["run_in_progress"] = True

    try:
        tickers = active_tickers

        identifier = get_identifier()
        allowed, limit_message = try_consume(identifier)

        if not allowed:
            st.warning(limit_message)
            st.stop()

        # Import the compiled graph lazily, after the gate/rate-limit checks
        # above have already passed. This module chain instantiates the
        # Gemini client, so importing it eagerly at the top of the file
        # would mean a misconfigured GOOGLE_API_KEY crashes the whole page
        # before Streamlit can render anything — including the password
        # gate.
        from agent.graph import graph

        run_failed = False
        error_message = None
        timed_out = False

        accumulated: dict = {
            "strategy": selected_strategy,
            "tickers": tickers,
            "research_results": {},
        }

        with st.status(
            "Running portfolio analysis — this can take 30–90 seconds…",
            expanded=True,
        ) as status:
            start_time = time.monotonic()

            try:
                for chunk in graph.stream({"strategy": selected_strategy, "tickers": tickers}):
                    if time.monotonic() - start_time > RUN_TIMEOUT_SECONDS:
                        timed_out = True
                        break

                    node_name, partial = next(iter(chunk.items()))
                    for key, value in partial.items():
                        if key == "research_results":
                            accumulated["research_results"].update(value)
                        else:
                            accumulated[key] = value
                    status.write(_step_message(node_name, partial))

                if timed_out:
                    run_failed = True
                    error_message = (
                        "This analysis is taking longer than expected (over "
                        f"{RUN_TIMEOUT_SECONDS // 60} minutes) and was stopped. "
                        "This can happen with a slow market-data provider or a "
                        "large number of holdings — please try again, "
                        "possibly with fewer tickers."
                    )
                    status.update(label="Analysis timed out", state="error", expanded=False)
                else:
                    status.update(label="Analysis complete", state="complete", expanded=False)

            except PortfolioAgentError as exc:
                # Covers network/API failures from yfinance and Gemini —
                # the agent nodes already wrap those into this friendly,
                # user-safe message type (see security/errors.py) and log
                # the real cause themselves before raising. This is a
                # defense-in-depth fallback in case that ever isn't true —
                # `raise PortfolioAgentError(...) from exc` sets __cause__,
                # so the original exception is still recoverable here.
                run_failed = True
                error_message = str(exc)
                if exc.__cause__ is not None:
                    print(f"[app] PortfolioAgentError caused by: {type(exc.__cause__).__name__}: {exc.__cause__}")
                status.update(label="Analysis failed", state="error", expanded=False)

            except (KeyError, TypeError, ValueError) as exc:
                # The graph produced data in an unexpected shape (e.g. a
                # malformed LLM response that slipped past the JSON parse).
                run_failed = True
                error_message = (
                    "The analysis produced an unexpected result. "
                    "Please try running it again."
                )
                traceback.print_exc()
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
        # error is always visible, not tucked inside a collapsed expander
        # that a visitor would have to know to click open.
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

    finally:
        # Always clears, even if something above raised unexpectedly, so
        # the Analyze button can never get stuck permanently disabled.
        st.session_state["run_in_progress"] = False
