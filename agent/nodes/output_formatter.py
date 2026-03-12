from agent.state import PortfolioState

# A simple bar made of Unicode block characters.  Each filled block (█)
# represents 2 percentage points, giving a maximum bar width of 50 chars
# for a 100 % allocation.  Stored as a module constant so the magic number
# is named and lives in one place.
_BAR_UNIT = 2  # percentage points represented by one █ character


def _make_bar(percentage: float) -> str:
    """Return a small ASCII progress bar for a single allocation percentage.

    Example: _make_bar(40.0) → "████████████████████░░░░░░░░░░  40.0%"
    """
    # How many filled blocks to draw.  int() truncates — intentional, since
    # we want a conservative visual rather than rounding up.
    filled = int(percentage / _BAR_UNIT)

    # The remainder of the bar is shown as light shade (░) so the total
    # bar width is always the same regardless of the percentage value.
    empty = (100 // _BAR_UNIT) - filled

    # Concatenate filled blocks, empty blocks, and the numeric label.
    # The label is right-aligned in a 6-character field (.6) so all
    # percentages line up vertically even when the integer part differs.
    return f"{'█' * filled}{'░' * empty}  {percentage:5.1f}%"


def output_formatter(state: PortfolioState) -> dict:
    """Format the final allocations into a markdown string for Streamlit.

    Reads `state["allocations"]` (and optionally `state["tickers"]` for
    ordering) and writes a fully rendered markdown report to `final_output`.
    Streamlit's `st.markdown()` can render this directly.
    """
    allocations: dict[str, float] = state["allocations"]

    # Sort tickers from highest to lowest allocation so the most significant
    # positions appear at the top of the report.  sorted() returns a new list
    # and does not mutate the original dict.
    sorted_tickers = sorted(allocations, key=lambda t: allocations[t], reverse=True)

    # ── Header ──────────────────────────────────────────────────────────────
    # Markdown H2 heading.  The horizontal rule (---) below it adds a visible
    # separator when rendered by Streamlit.
    lines = [
        "## Portfolio Allocation Report",
        "---",
        "",
    ]

    # ── Per-ticker rows ──────────────────────────────────────────────────────
    for ticker in sorted_tickers:
        pct = allocations[ticker]

        # Bold ticker symbol, followed by a code-fenced bar so Streamlit
        # renders it in a monospace font (important for bar alignment).
        lines.append(f"**{ticker}**")
        lines.append(f"```")
        lines.append(_make_bar(pct))
        lines.append(f"```")
        # Blank line between tickers for visual breathing room.
        lines.append("")

    # ── Summary footer ───────────────────────────────────────────────────────
    lines.append("---")

    # Sum the allocation values to show a total.  round() avoids floating-
    # point noise like 99.999999 when the values came from division.
    total = round(sum(allocations.values()), 2)

    # Inline code block (`...`) renders as monospace in Streamlit markdown.
    lines.append(f"**Total allocated:** `{total}%`")

    # ── Join and store ───────────────────────────────────────────────────────
    # "\n".join() collapses the list into a single newline-delimited string.
    # This is what Streamlit's st.markdown() will receive.
    final_output = "\n".join(lines)

    # Return only the field this node writes.  LangGraph merges it into state.
    return {"final_output": final_output}
