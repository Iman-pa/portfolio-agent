import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path fix — make the project root importable
# ---------------------------------------------------------------------------
# __file__ is frontend/app.py.  Two .parent calls walk up to the project root.
# Inserting it at position 0 of sys.path means Python checks the project root
# first when resolving imports, so `from agent.graph import graph` works when
# Streamlit is launched from any working directory.
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
from agent.graph import graph

# ---------------------------------------------------------------------------
# Page configuration — must be the first Streamlit call in the script
# ---------------------------------------------------------------------------
# `page_title` sets the browser tab label.
# `page_icon` is the favicon shown next to the tab label.
# `layout="centered"` keeps content in a readable column rather than
# stretching edge-to-edge on wide monitors.
st.set_page_config(
    page_title="Portfolio Allocation Agent",
    page_icon="📊",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
# st.title() renders an H1 heading styled by Streamlit's theme.
st.title("📊 Portfolio Allocation Agent")

# st.caption() renders small muted text — suitable for a one-line tagline.
st.caption("Powered by Gemini 2.0 Flash · Built with LangGraph")

# st.markdown() renders any markdown string.  We use it here for the
# description paragraph; the triple-quoted string keeps it readable in code.
st.markdown(
    """
    This agent reads your current portfolio from **data/portfolio.json**,
    researches each stock, and uses an LLM to recommend an optimal percentage
    allocation across your holdings.

    Click **Run Agent** to start a full analysis run.
    """
)

# A horizontal rule gives a clean visual break between the intro and the
# interactive section below it.
st.divider()

# ---------------------------------------------------------------------------
# Strategy selector
# ---------------------------------------------------------------------------
# Maps the human-readable label shown in the dropdown to the snake_case key
# expected by the YAML file and the state schema.  A dict preserves insertion
# order (Python 3.7+), so the options appear in this exact order in the UI.
_STRATEGY_OPTIONS: dict[str, str] = {
    "Conservative — Capital Preservation": "conservative",
    "Balanced — Growth with Stability":    "balanced",
    "Aggressive — Maximum Growth":         "aggressive",
    "Income — Dividend & Yield Focus":     "income",
}

# st.selectbox() renders a dropdown menu.
# The first argument is the label displayed above the widget.
# `options` receives the human-readable keys of the dict as the list of choices.
# The widget returns the selected label string on every script rerun.
selected_label = st.selectbox(
    "Investment Strategy",
    options=list(_STRATEGY_OPTIONS.keys()),
)

# st.caption() displays small muted helper text directly below the dropdown,
# giving the user a one-line reminder of what the selected strategy means
# without cluttering the main layout.
_STRATEGY_CAPTIONS: dict[str, str] = {
    "conservative": "Prioritises capital preservation and low volatility.",
    "balanced":     "Mixes growth and stability across the portfolio.",
    "aggressive":   "Maximises returns; accepts high short-term volatility.",
    "income":       "Focuses on dividend-paying, stable income stocks.",
}

# Look up the internal key for the selected label, then show its caption.
# _STRATEGY_OPTIONS[selected_label] converts e.g. "Balanced — Growth with
# Stability" → "balanced" so we can look up the matching caption string.
selected_strategy = _STRATEGY_OPTIONS[selected_label]
st.caption(_STRATEGY_CAPTIONS[selected_strategy])

# ---------------------------------------------------------------------------
# Trigger button
# ---------------------------------------------------------------------------
# st.button() renders a clickable button and returns True on the frame where
# the user clicked it, False on every other frame.  The entire block below
# only executes on the frame the button is pressed — Streamlit re-runs the
# whole script on every interaction, so this is the standard pattern.
if st.button("▶ Run Agent", type="primary", use_container_width=True):

    # st.spinner() shows an animated spinner with a status message while the
    # indented block executes.  It disappears automatically when the block
    # finishes (success or exception).
    with st.spinner("Agent is running — researching tickers and deciding allocations…"):
        try:
            # Pass the selected strategy into the graph as part of the initial
            # state dict.  LangGraph merges this with any defaults so
            # allocation_decider can read state["strategy"] directly.
            # Every other field (tickers, macro_context, etc.) is populated
            # by the nodes themselves during the run.
            result = graph.invoke({"strategy": selected_strategy})

            # result is the final PortfolioState dict after all nodes have run.
            # "final_output" is the markdown string written by output_formatter.
            # st.markdown() renders it with full markdown support so the bars,
            # bold text, and code blocks all display correctly.
            st.markdown(result["final_output"])

        except Exception as exc:
            # st.error() renders a red alert box — visible and clearly labelled
            # as an error without crashing the app with a full traceback.
            # str(exc) surfaces the error message so the user has something
            # actionable to look at (e.g. a missing API key, bad JSON, etc.).
            st.error(f"The agent encountered an error: {exc}")
