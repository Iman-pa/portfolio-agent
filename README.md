README description (a few paragraphs):

Portfolio Decision Agent
A multi-agent portfolio management system built with LangGraph and Gemini that takes a portfolio of stocks and produces a fully reasoned allocation recommendation backed by quantitative analysis.
What it does
The agent fetches live market indicators (VIX, 10Y Treasury yield, SPY momentum), computes pairwise correlations between holdings, sends each stock to a Deep Research agent for analysis, and uses an LLM to decide the optimal allocation percentage for each position — accounting for the user's chosen investment strategy and current macro environment.
Every allocation comes with a confidence score and one-sentence reasoning. The system then computes eight quantitative portfolio metrics from one year of historical price data: expected return, volatility, beta, Value at Risk, Sharpe ratio, Sortino ratio, max drawdown, and concentration risk.
Tech stack
LangGraph · Gemini 2.0 Flash · FastAPI · Streamlit · yfinance · LangSmith
Investment strategies
Conservative · Balanced · Aggressive · Income
Part of a larger multi-agent system
This agent operates as the portfolio decision layer in a broader trading system that includes a Basic Market Data Agent, Company Deep Research Agent, Technical Spike Scout, and Political Theme Miner.

