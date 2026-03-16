# Defines the shared state schema passed between all nodes in the LangGraph portfolio agent.

from typing import Annotated, Any, TypedDict
import operator


class PortfolioState(TypedDict):
    # -------------------------------------------------------------------------
    # INPUT — provided once at the start of the graph run
    # -------------------------------------------------------------------------

    # List of stock ticker symbols the user wants to analyse, e.g. ["AAPL", "NVDA", "TSLA"].
    # This is the primary input to the graph and never changes during a run.
    tickers: list[str]

    # -------------------------------------------------------------------------
    # LOOP CONTROL — tracks which stock is currently being researched
    # -------------------------------------------------------------------------

    # Index into `tickers` pointing at the stock that is being sent to the
    # Deep Research agent right now.  Starts at 0 and increments after each
    # research call so the graph can iterate through all tickers one by one.
    current_ticker_index: int

    # -------------------------------------------------------------------------
    # MACRO CONTEXT — fetched once after the portfolio is loaded
    # -------------------------------------------------------------------------

    # Market-wide indicators fetched by macro_fetcher, e.g.
    # {'vix': 18.3, 'yield_10y': 4.6, 'spy_price': 542.1}.
    # Plain English keys make it easy to reference in prompts and the UI.
    macro_context: dict[str, float]

    # -------------------------------------------------------------------------
    # RESEARCH ACCUMULATOR — grows as each stock is researched
    # -------------------------------------------------------------------------

    # Maps every ticker that has been researched so far to the raw text report
    # returned by the Deep Research agent, e.g. {"AAPL": "Apple Inc. is …"}.
    # `operator.or_` is used as the reducer so each node can add its own key
    # without overwriting what previous nodes already stored.
    research_results: Annotated[dict[str, str], operator.or_]

    # -------------------------------------------------------------------------
    # ALLOCATION OUTPUT — produced once all research is complete
    # -------------------------------------------------------------------------

    # Maps every ticker to the recommended allocation percentage decided by the
    # allocation node, e.g. {"AAPL": 40.0, "NVDA": 35.0, "TSLA": 25.0}.
    # Percentages must sum to 100 across all tickers.
    allocations: dict[str, float]

    # -------------------------------------------------------------------------
    # FINAL OUTPUT — produced by output_formatter, consumed by the UI
    # -------------------------------------------------------------------------

    # A fully formatted markdown string ready to be rendered by Streamlit.
    # Produced by output_formatter as the last step of the graph.
    final_output: str
