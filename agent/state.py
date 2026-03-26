# Defines the shared state schema passed between all nodes in the LangGraph portfolio agent.

from typing import Annotated, TypedDict
import operator


class PortfolioState(TypedDict):
    # -------------------------------------------------------------------------
    # INPUT — provided once at the start of the graph run
    # -------------------------------------------------------------------------

    # List of stock ticker symbols the user wants to analyse, e.g. ["AAPL", "NVDA", "TSLA"].
    # This is the primary input to the graph and never changes during a run.
    tickers: list[str]

    # The investment strategy selected by the user before running the graph.
    # Must be one of: "conservative", "balanced", "aggressive", "income".
    # Passed in via graph.invoke({"strategy": "balanced"}) from the frontend.
    # allocation_decider reads this to pick the matching strategy prompt section
    # from prompts/allocation_decider.yaml.
    strategy: str

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
    # CORRELATION CONTEXT — computed once after macro data is fetched
    # -------------------------------------------------------------------------

    # Pairwise return correlations for all holdings computed by
    # correlation_analyzer using 90 days of daily closing prices from yfinance.
    # The dict always has exactly three keys:
    #   "high_pairs"       — list of [ticker_a, ticker_b] pairs whose Pearson
    #                        correlation is above 0.7; these stocks tend to rise
    #                        and fall together, concentrating risk.
    #   "low_pairs"        — list of [ticker_a, ticker_b] pairs whose Pearson
    #                        correlation is below 0.3; these stocks move
    #                        independently, improving diversification.
    #   "avg_correlation"  — float, the arithmetic mean of all unique pairwise
    #                        correlations (upper triangle of the matrix, no
    #                        diagonal), rounded to 4 decimal places.
    # Example:
    #   {"high_pairs": [["AAPL","NVDA"]], "low_pairs": [], "avg_correlation": 0.72}
    correlation_context: dict

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

    # Maps every ticker to a richer allocation dict produced by allocation_decider.
    # Each inner dict has three keys:
    #   "allocation"  — float, percentage of the portfolio (all values sum to 100)
    #   "confidence"  — int 0-100, how confident the LLM is in its decision
    #   "reason"      — str, one sentence explaining the allocation choice
    # Example: {"AAPL": {"allocation": 35.0, "confidence": 82, "reason": "..."}}
    allocations: dict[str, dict]

    # -------------------------------------------------------------------------
    # PORTFOLIO METRICS — computed from historical prices after allocation
    # -------------------------------------------------------------------------

    # Quantitative metrics for the recommended portfolio, computed by
    # portfolio_metrics using 1 year of yfinance price history.
    # Keys:
    #   "expected_return"  — float, annualised weighted return (e.g. 0.142 = 14.2%)
    #   "max_drawdown"     — float, worst peak-to-trough loss (e.g. -0.183 = -18.3%)
    #   "sharpe_ratio"     — float, annualised excess return per unit of volatility
    portfolio_metrics: dict[str, float]

    # -------------------------------------------------------------------------
    # FINAL OUTPUT — produced by output_formatter, consumed by the UI
    # -------------------------------------------------------------------------

    # A fully formatted markdown string ready to be rendered by Streamlit.
    # Produced by output_formatter as the last step of the graph.
    final_output: str
