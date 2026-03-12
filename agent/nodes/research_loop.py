from agent.state import PortfolioState


def _call_deep_research(ticker: str) -> str:
    """Mock of the Deep Research agent call.

    Returns a plain string report for the given ticker.
    Replace the body of this function with the real API call when ready —
    the rest of the node does not need to change.
    """
    return f"Research report for {ticker}: This is a placeholder report."


def research_loop(state: PortfolioState) -> dict:
    """Research one ticker and advance the loop counter by one.

    LangGraph calls this node repeatedly (via the conditional edge in graph.py)
    until `current_ticker_index` reaches the end of `tickers`.  Each call
    handles exactly one ticker, which keeps the node simple and easy to test.
    """
    # Read the index that portfolio_loader initialised to 0 and that this
    # node itself incremented on every previous iteration.  It points at the
    # ticker we need to research *this* invocation.
    index = state["current_ticker_index"]

    # Use the index to look up the actual ticker symbol from the list that
    # portfolio_loader wrote into state, e.g. "AAPL" on the first iteration.
    ticker = state["tickers"][index]

    # Delegate to the mock (or future real) research function.
    # Keeping the call in its own function means we only change one place
    # when we swap the mock for the real Deep Research agent.
    report = _call_deep_research(ticker)

    # Return only the two fields this node modifies.  LangGraph merges this
    # partial dict into the shared state rather than replacing it.
    #
    # `research_results` uses the `operator.or_` reducer defined in state.py,
    # so returning {ticker: report} *adds* this key to the existing dict
    # instead of overwriting the results from previous iterations.
    #
    # `current_ticker_index` is incremented by 1 so that the next invocation
    # (if the conditional edge routes back here) picks up the following ticker.
    return {
        "research_results": {ticker: report},
        "current_ticker_index": index + 1,
    }
