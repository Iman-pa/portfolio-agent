# Portfolio Decision Agent

A multi-agent portfolio management system built with LangGraph and Gemini that takes a portfolio of stocks and produces a fully reasoned allocation recommendation backed by quantitative analysis.

## What it does

The agent fetches live market indicators (VIX, 10Y Treasury yield, SPY momentum), computes pairwise correlations between holdings, sends each stock to a Deep Research agent for analysis, and uses an LLM to decide the optimal allocation percentage for each position — accounting for the user's chosen investment strategy and current macro environment.

Every allocation comes with a confidence score and one-sentence reasoning. The system then computes eight quantitative portfolio metrics from one year of historical price data: expected return, volatility, beta, Value at Risk, Sharpe ratio, Sortino ratio, max drawdown, and concentration risk.

## Tech stack

LangGraph · Gemini 2.0 Flash · FastAPI · Streamlit · yfinance · LangSmith

## Investment strategies

Conservative · Balanced · Aggressive · Income

## Part of a larger multi-agent system

This agent operates as the portfolio decision layer in a broader trading system that includes a Basic Market Data Agent, Company Deep Research Agent, Technical Spike Scout, and Political Theme Miner.

## Deploying the live demo

The demo deployment is a single Streamlit service on Render — unlike a split frontend/backend setup, the UI and agent logic run in the same process, so there's only one service to deploy. It's gated behind `DEMO_PASSWORD` and a per-visitor / global daily analysis cap (see [Demo access control](#demo-access-control) below) to keep Gemini API usage bounded.

### Render

1. Push this repo to GitHub if the latest commit isn't there yet.
2. Go to [render.com](https://render.com) and sign in (GitHub login is easiest).
3. **New +** → **Blueprint** → connect this GitHub repo. Render will detect `render.yaml` at the repo root and propose one service, `portfolio-agent`.
4. Before the first deploy, Render will prompt for the env vars marked `sync: false` in `render.yaml`. Set:
   - `GOOGLE_API_KEY` — your real Gemini API key
   - `DEMO_PASSWORD` — the demo access password (e.g. `Portfolio138`)
   - `LANGCHAIN_API_KEY` — optional, only if you want LangSmith tracing; leave blank to skip it entirely
5. Deploy. First build takes a few minutes (installing `langgraph`, `langchain`, etc.).
6. Once live, note the URL (e.g. `https://portfolio-agent.onrender.com`).
7. Sanity check: `curl https://<your-app>.onrender.com/_stcore/health` should return `ok` — Streamlit's built-in health endpoint.

**Free-tier notes:**
- The service spins down after ~15 minutes of inactivity and takes up to ~50 seconds to wake on the next request. Because this is one process rather than a separate frontend pinging a backend, there's no dedicated "waking up" message — the first visitor after a period of inactivity just sees a slower-than-usual page load rather than anything that looks broken. Render restarts the already-built container on wake (no reinstall), so this is app-boot time, not a fresh build.
- The usage log (`logs/usage_log.jsonl`) and rate-limit counters (`logs/rate_limit_state.json`) are local files and reset whenever the service restarts or redeploys — acceptable for a demo, not a durable audit trail.
- Render's free/starter plans run a single instance with no autoscaling, which is what the local-file-based rate limiter in `security/rate_limit.py` assumes; it would need rework if this ever moved to a multi-instance plan.

### Demo access control

- The app is gated by a password screen (`DEMO_PASSWORD`, checked via Streamlit session state) before any of the portfolio UI renders — there is no unauthenticated path to an analysis run. If `DEMO_PASSWORD` isn't set, the app fails closed and blocks everyone rather than defaulting to open access.
- Rate limits (`security/rate_limit.py`): 5 analyses per visitor per rolling 24h, 30 total per day server-wide, both enforced before the graph runs. Hitting either shows a friendly message rather than an error.
- Every completed or failed analysis run is logged — timestamp, identifier, strategy, tickers, outcome — to `logs/usage_log.jsonl` for rough usage/cost review.
