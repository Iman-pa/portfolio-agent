from agent.state import PortfolioState

# Each filled block represents 2 percentage points.
# A 100 % allocation therefore spans exactly 50 blocks.
_BAR_UNIT = 2


def _make_bar(percentage: float) -> str:
    """Return a fixed-width ASCII progress bar for an allocation percentage.

    Example: _make_bar(40.0) → "████████████████████░░░░░░░░░░  40.0%"
    The bar is always 50 characters wide so all tickers align vertically.
    """
    # int() truncates — we never want the bar to visually exceed the percentage.
    filled = int(percentage / _BAR_UNIT)

    # Pad the remainder with light-shade blocks to keep constant total width.
    empty = (100 // _BAR_UNIT) - filled

    # :5.1f formats the float as " 40.0" — one decimal place, right-aligned in
    # a 5-character field — so single- and double-digit values line up cleanly.
    return f"{'█' * filled}{'░' * empty}  {percentage:5.1f}%"


def _confidence_badge(confidence: int) -> str:
    """Return a short markdown label that colours the confidence score.

    Streamlit does not support arbitrary CSS in plain markdown, so we use
    emoji as a lightweight visual signal that requires no custom styling:
      🟢  high confidence  (>= 75)
      🟡  medium confidence (>= 50)
      🔴  low confidence   (< 50)
    The numeric value is shown alongside so the exact score is never hidden.
    """
    if confidence >= 75:
        # High confidence — green circle emoji.
        icon = "🟢"
    elif confidence >= 50:
        # Medium confidence — yellow circle emoji.
        icon = "🟡"
    else:
        # Low confidence — red circle emoji.
        icon = "🔴"

    # The result is a short inline string like "🟢 Confidence: 82/100".
    # It will be placed on the same line as the ticker heading.
    return f"{icon} Confidence: {confidence}/100"


def output_formatter(state: PortfolioState) -> dict:
    """Format the richer allocation dicts into a markdown string for Streamlit.

    Each ticker's allocation dict contains:
        "allocation"  — float percentage
        "confidence"  — int 0-100
        "reason"      — one-sentence string

    The final output renders:
        TICKER    🟢 Confidence: 82/100
        [bar]
        > reason sentence
    """
    # state["allocations"] is now dict[str, dict], e.g.:
    # {"AAPL": {"allocation": 35.0, "confidence": 82, "reason": "..."}, ...}
    allocations: dict[str, dict] = state["allocations"]

    # Sort tickers by allocation percentage, highest first, so the largest
    # positions appear at the top of the report.
    # The lambda extracts the "allocation" float from each inner dict for
    # comparison — sorted() never sees the outer ticker key.
    sorted_tickers = sorted(
        allocations,
        key=lambda t: allocations[t]["allocation"],
        reverse=True,
    )

    # ── Header ───────────────────────────────────────────────────────────────
    lines = [
        "## Portfolio Allocation Report",
        "---",
        "",
    ]

    # ── Per-ticker rows ───────────────────────────────────────────────────────
    for ticker in sorted_tickers:
        # Unpack the three keys from the inner dict.
        # We use explicit key access rather than .get() because all three keys
        # are guaranteed by the prompt — missing keys should raise immediately.
        pct: float = allocations[ticker]["allocation"]
        confidence: int = allocations[ticker]["confidence"]
        reason: str = allocations[ticker]["reason"]

        # Header line: bold ticker on the left, confidence badge on the right,
        # separated by four non-breaking spaces for visual spacing.
        # The &nbsp; trick does not work in st.markdown — plain spaces inside
        # a bold span collapse, so we use a dash separator instead.
        lines.append(f"**{ticker}** — {_confidence_badge(confidence)}")

        # Fenced code block forces monospace rendering so the bar characters
        # align correctly.  Without this, proportional fonts misalign █ and ░.
        lines.append("```")
        lines.append(_make_bar(pct))
        lines.append("```")

        # Blockquote ("> ") renders as an indented, visually distinct paragraph
        # in Streamlit markdown — ideal for the one-sentence reason.
        lines.append(f"> {reason}")

        # Blank line gives breathing room between tickers.
        lines.append("")

    # ── Allocation summary ────────────────────────────────────────────────────
    lines.append("---")

    # Sum only the "allocation" floats from each inner dict.
    # round() eliminates floating-point noise (e.g. 99.99999 → 100.0).
    total = round(sum(v["allocation"] for v in allocations.values()), 2)

    # Inline code renders as monospace so the percentage stands out.
    lines.append(f"**Total allocated:** `{total}%`")

    # ── Portfolio metrics section ─────────────────────────────────────────────
    # Blank line before the new section header for visual separation.
    lines.append("")
    lines.append("## Portfolio Metrics (1-Year Historical)")
    lines.append("---")
    lines.append("")

    # Read the metrics dict written by portfolio_metrics into state.
    # All three keys are guaranteed to exist — the node always writes all of them.
    metrics: dict[str, float] = state["portfolio_metrics"]

    # ── Expected return ───────────────────────────────────────────────────────
    # The stored value is a decimal fraction (e.g. 0.142).
    # Multiplying by 100 converts it to a percentage (14.2) for display.
    expected_pct = metrics["expected_return"] * 100

    # ▲ for positive return, ▼ for negative — gives an instant directional read.
    er_arrow = "▲" if expected_pct >= 0 else "▼"

    # :.2f formats to exactly two decimal places — standard in finance reports.
    lines.append(f"**Expected Annual Return:** {er_arrow} `{expected_pct:.2f}%`")

    lines.append("")

    # ── Max drawdown ──────────────────────────────────────────────────────────
    # The stored value is already negative (e.g. -0.183 = worst loss of 18.3%).
    # abs() removes the sign so we can prefix "−" ourselves, making the label
    # read as "−18.30%" — unambiguous regardless of display font.
    drawdown_pct = abs(metrics["max_drawdown"] * 100)
    lines.append(f"**Max Drawdown (1Y):** `−{drawdown_pct:.2f}%`")

    lines.append("")

    # ── Sharpe ratio ──────────────────────────────────────────────────────────
    sharpe = metrics["sharpe_ratio"]

    # Colour-code with emoji: green ≥ 1.0 (good), yellow ≥ 0.5 (acceptable),
    # red < 0.5 (poor).  Finance convention: Sharpe > 1 is generally desirable.
    if sharpe >= 1.0:
        sharpe_icon = "🟢"
    elif sharpe >= 0.5:
        sharpe_icon = "🟡"
    else:
        sharpe_icon = "🔴"

    # :.2f is the conventional precision for Sharpe ratios in professional reports.
    lines.append(f"**Sharpe Ratio:** {sharpe_icon} `{sharpe:.2f}`")

    lines.append("")

    # ── Volatility ────────────────────────────────────────────────────────────
    # The stored value is a decimal (e.g. 0.182).  Multiplying by 100 converts
    # it to a percentage (18.2%) which is the standard display format.
    vol_pct = metrics["volatility"] * 100

    # Thresholds: < 15% is low volatility (green), 15-25% is moderate (yellow),
    # > 25% is high (red).  These are loose industry conventions for diversified
    # equity portfolios; individual stocks would warrant higher thresholds.
    if vol_pct < 15:
        vol_icon = "🟢"
    elif vol_pct < 25:
        vol_icon = "🟡"
    else:
        vol_icon = "🔴"

    lines.append(f"**Annualised Volatility:** {vol_icon} `{vol_pct:.2f}%`")

    lines.append("")

    # ── Beta ──────────────────────────────────────────────────────────────────
    # Beta is already a dimensionless ratio — no unit conversion needed.
    beta = metrics["beta"]

    # < 1.0 means the portfolio is less sensitive to market swings than SPY (green).
    # 1.0–1.5 means moderate amplification of market moves (yellow).
    # > 1.5 means the portfolio swings significantly more than the market (red).
    if beta < 1.0:
        beta_icon = "🟢"
    elif beta < 1.5:
        beta_icon = "🟡"
    else:
        beta_icon = "🔴"

    lines.append(f"**Beta (vs SPY):** {beta_icon} `{beta:.2f}`")

    lines.append("")

    # ── VaR 95% ───────────────────────────────────────────────────────────────
    # The stored value is the 5th-percentile daily return — a negative number,
    # e.g. -0.023.  We take abs() and multiply by 100 to get a positive
    # percentage (2.3%) and then prefix "−" so it reads as an explicit loss.
    var_pct = abs(metrics["var_95"] * 100)

    # < 1.5% worst-day loss is low risk (green), 1.5–2.5% is moderate (yellow),
    # > 2.5% means the portfolio has a fat left tail / high daily loss risk (red).
    if var_pct < 1.5:
        var_icon = "🟢"
    elif var_pct < 2.5:
        var_icon = "🟡"
    else:
        var_icon = "🔴"

    lines.append(f"**VaR 95% (1-Day):** {var_icon} `−{var_pct:.2f}%`")

    lines.append("")

    # ── Sortino ratio ─────────────────────────────────────────────────────────
    sortino = metrics["sortino_ratio"]

    # Same colour thresholds as Sharpe — both are risk-adjusted return ratios.
    # Sortino is typically higher than Sharpe for the same portfolio because
    # it only penalises downside volatility, not total volatility.
    if sortino >= 1.0:
        sortino_icon = "🟢"
    elif sortino >= 0.5:
        sortino_icon = "🟡"
    else:
        sortino_icon = "🔴"

    # A value of 0.0 means there were no negative return days in the period —
    # technically not poor performance, but we display the number as-is and let
    # the colour coding convey that it signals missing data rather than bad risk.
    lines.append(f"**Sortino Ratio:** {sortino_icon} `{sortino:.2f}`")

    lines.append("")

    # ── Concentration risk (HHI) ──────────────────────────────────────────────
    hhi = metrics["concentration_risk"]

    # HHI ranges from 1/N (perfectly equal-weight) to 1.0 (single stock).
    # < 0.25 means the allocation weight is reasonably spread out (green).
    # 0.25–0.40 means moderate concentration — a few large positions (yellow).
    # > 0.40 means one or two names dominate the portfolio heavily (red).
    if hhi < 0.25:
        hhi_icon = "🟢"
    elif hhi < 0.40:
        hhi_icon = "🟡"
    else:
        hhi_icon = "🔴"

    # Display HHI as a raw decimal to 4 places — unlike return metrics it is
    # not a percentage, and rounding to 2 places would hide meaningful variation
    # (e.g. 0.2500 vs 0.2499 straddle the green/yellow threshold).
    lines.append(f"**Concentration Risk (HHI):** {hhi_icon} `{hhi:.4f}`")

    # ── Join and return ───────────────────────────────────────────────────────
    # "\n".join() collapses the list into one newline-delimited string.
    # st.markdown(result["final_output"]) in app.py renders this directly.
    final_output = "\n".join(lines)

    return {"final_output": final_output}
