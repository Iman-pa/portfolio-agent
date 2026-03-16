# ---------------------------------------------------------------------------
# graph.py — wires together the LangGraph portfolio agent
# ---------------------------------------------------------------------------

# LINE 1-2: LangGraph's core building block.  StateGraph is a directed graph
# whose nodes share a single typed state object.  Every node receives the
# full current state and returns a *partial* dict of keys it wants to update.
from langgraph.graph import StateGraph, END

# LINE 5-9: Import the shared state schema.  PortfolioState is a TypedDict
# that defines every key that can exist on the state object.  Passing it to
# StateGraph lets LangGraph validate node outputs at runtime and provide
# IDE auto-complete throughout the project.
from agent.state import PortfolioState

# LINE 12-15: Import each node function from its own module.  Each function
# has the signature  (state: PortfolioState) -> dict  — it reads whatever
# fields it needs from state and returns only the fields it changes.
from agent.nodes.portfolio_loader import portfolio_loader
from agent.nodes.macro_fetcher import macro_fetcher
from agent.nodes.research_loop import research_loop
from agent.nodes.allocation_decider import allocation_decider
from agent.nodes.portfolio_metrics import portfolio_metrics
from agent.nodes.output_formatter import output_formatter


# ---------------------------------------------------------------------------
# Routing function — the brain of the research loop
# ---------------------------------------------------------------------------

def should_continue_research(state: PortfolioState) -> str:
    # LINE 26: Called by LangGraph after every execution of `research_loop`.
    # It must return a string that matches one of the keys in the conditional
    # edge map defined below.  LangGraph uses that string to decide which node
    # to visit next — this is how loops are implemented without native cycles.

    # LINE 31: `current_ticker_index` was incremented by `research_loop` just
    # before this function runs, so comparing it against the length of
    # `tickers` tells us whether there are still unresearched stocks.
    if state["current_ticker_index"] < len(state["tickers"]):
        # LINE 34: More tickers remain — route back to `research_loop` so it
        # picks up the next ticker on the next graph step.
        return "continue"
    else:
        # LINE 38: All tickers have been researched — exit the loop and hand
        # off to the allocation node.
        return "done"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

# LINE 44: Instantiate the graph, binding it to PortfolioState.  From this
# point on, LangGraph knows the shape of the state that flows between nodes.
graph_builder = StateGraph(PortfolioState)

# LINE 48-51: Register each node.  The first argument is the *name* used in
# edge declarations; the second is the Python callable that implements it.
graph_builder.add_node("portfolio_loader",   portfolio_loader)
graph_builder.add_node("macro_fetcher",      macro_fetcher)
graph_builder.add_node("research_loop",      research_loop)
graph_builder.add_node("allocation_decider", allocation_decider)
graph_builder.add_node("portfolio_metrics",  portfolio_metrics)
graph_builder.add_node("output_formatter",   output_formatter)

# LINE 54: Declare the entry point.  LangGraph will call `portfolio_loader`
# first when `.invoke()` is called on the compiled graph.
graph_builder.set_entry_point("portfolio_loader")

# Unconditional edge — after `portfolio_loader` finishes, always move to
# `macro_fetcher` to collect market-wide context before research begins.
graph_builder.add_edge("portfolio_loader", "macro_fetcher")

# Unconditional edge — after `macro_fetcher` finishes, move straight to
# `research_loop`.  macro_context is now available in state for all nodes.
graph_builder.add_edge("macro_fetcher", "research_loop")

# LINE 62-70: Conditional edge — after `research_loop` finishes, call
# `should_continue_research` with the current state.  The returned string is
# looked up in the mapping dict:
#   "continue" → go back to `research_loop`  (keeps the loop running)
#   "done"     → go to `allocation_decider`  (exits the loop)
graph_builder.add_conditional_edges(
    "research_loop",            # source node
    should_continue_research,   # routing function
    {
        "continue": "research_loop",     # loop back for the next ticker
        "done":     "allocation_decider", # all tickers done, move on
    },
)

# Unconditional edge — once allocations are decided, compute portfolio metrics
# before formatting.  This gives output_formatter access to all three numbers.
graph_builder.add_edge("allocation_decider", "portfolio_metrics")

# Unconditional edge — after metrics are computed, run the formatter to
# produce the final markdown output that Streamlit will render.
graph_builder.add_edge("portfolio_metrics", "output_formatter")

# LINE 77: Unconditional edge to the special END sentinel.  Reaching END
# tells LangGraph the run is complete and `.invoke()` should return.
graph_builder.add_edge("output_formatter", END)

# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------

# LINE 83-85: `.compile()` validates the graph (checks for unreachable nodes,
# missing edges, etc.) and returns an executable `CompiledGraph` object.
# This is the object you call `.invoke(initial_state)` on.
graph = graph_builder.compile()
