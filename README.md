# TradingAgents: Multi-Agents LLM Financial Trading Framework

## About this fork

This repository is a **fork of [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents)**. It stays aligned with the multi-agent research framework while focusing on **Indian markets** and day-to-day operability:

- **Zerodha Kite Connect** — A full **login / token flow** (local OAuth callback + `request_token` → `access_token` exchange) and optional **Kite-backed OHLCV, quotes, and technical indicators** when you point `data_vendors` at `kite` (see `tradingagents/default_config.py` and `.env.example`).
- **Langfuse** — **Observability** for LLM generations and tool calls via LangChain callbacks, with per-run traces and session grouping (see [Observability (Langfuse)](#observability-langfuse) below).
- **Provider-friendly defaults** — LLM **rate limiting** and **retries** to reduce failures under API throttling (configurable via env / `DEFAULT_CONFIG`).

<div align="center">

🚀 [TradingAgents framework](#tradingagents-framework) | ⚡ [Installation](#installation) | 🎬 [Demo](https://www.youtube.com/watch?v=90gr5lwjIho)

</div>

## TradingAgents Framework

TradingAgents is a multi-agent trading framework that mirrors the dynamics of real-world trading firms. By deploying specialized LLM-powered agents: from fundamental analysts, sentiment experts, and technical analysts, to trader, risk management team, the platform collaboratively evaluates market conditions and informs trading decisions. Moreover, these agents engage in dynamic discussions to pinpoint the optimal strategy.

<p align="center">
  <img src="assets/schema.png" style="width: 100%; height: auto;">
</p>

> TradingAgents framework is designed for research purposes. Trading performance may vary based on many factors, including the chosen backbone language models, model temperature, trading periods, the quality of data, and other non-deterministic factors. [It is not intended as financial, investment, or trading advice.](https://tauric.ai/disclaimer/)

Our framework decomposes complex trading tasks into specialized roles. This ensures the system achieves a robust, scalable approach to market analysis and decision-making.

### Analyst Team
- Fundamentals Analyst: Evaluates company financials and performance metrics, identifying intrinsic values and potential red flags.
- Sentiment Analyst: Analyzes social media and public sentiment using sentiment scoring algorithms to gauge short-term market mood.
- News Analyst: Monitors global news and macroeconomic indicators, interpreting the impact of events on market conditions.
- Technical Analyst: Utilizes technical indicators (like MACD and RSI) to detect trading patterns and forecast price movements.

<p align="center">
  <img src="assets/analyst.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

### Researcher Team
- Comprises both bullish and bearish researchers who critically assess the insights provided by the Analyst Team. Through structured debates, they balance potential gains against inherent risks.

<p align="center">
  <img src="assets/researcher.png" width="70%" style="display: inline-block; margin: 0 2%;">
</p>

### Trader Agent
- Composes reports from the analysts and researchers to make informed trading decisions. It determines the timing and magnitude of trades based on comprehensive market insights.

<p align="center">
  <img src="assets/trader.png" width="70%" style="display: inline-block; margin: 0 2%;">
</p>

### Risk Management and Portfolio Manager
- Continuously evaluates portfolio risk by assessing market volatility, liquidity, and other risk factors. The risk management team evaluates and adjusts trading strategies, providing assessment reports to the Portfolio Manager for final decision.
- The Portfolio Manager approves/rejects the transaction proposal. If approved, the order will be sent to the simulated exchange and executed.

<p align="center">
  <img src="assets/risk.png" width="70%" style="display: inline-block; margin: 0 2%;">
</p>

## Installation

### Environment

Clone this repository (use your fork’s URL if applicable; upstream is [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents)):
```bash
git clone https://github.com/TauricResearch/TradingAgents.git
cd TradingAgents
```

Create a virtual environment in any of your favorite environment managers:
```bash
conda create -n tradingagents python=3.13
conda activate tradingagents
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### Required APIs

TradingAgents supports multiple LLM providers. Set the API key for your chosen provider:

```bash
export OPENAI_API_KEY=...          # OpenAI (GPT)
export GOOGLE_API_KEY=...          # Google (Gemini)
export ANTHROPIC_API_KEY=...       # Anthropic (Claude)
export XAI_API_KEY=...             # xAI (Grok)
export OPENROUTER_API_KEY=...      # OpenRouter
export ALPHA_VANTAGE_API_KEY=...   # Alpha Vantage
```

### Zerodha Kite Connect (optional, Indian markets)

To use **Kite** for prices, historical candles, and indicator-style series, configure `KITE_*` in `.env` (see `.env.example`) and set the relevant `data_vendors` entries to `"kite"` in `tradingagents/default_config.py` or your own config copy.

**Login and access token**

1. Create a Kite Connect app and set the redirect URL to **`http://127.0.0.1:8765/kite/callback`** (or match `KITE_OAUTH_*` if you override host/port/path).
2. Set `KITE_API_KEY` and `KITE_API_SECRET` (and optionally load them via `.env`).
3. Run the local helper:
   ```bash
   python scripts/kite_token_server.py
   ```
4. Open the Kite login URL with your API key (the script docstring shows the pattern: `https://kite.zerodha.com/connect/login?v=3&api_key=YOUR_API_KEY`).
5. After login, the callback handler returns JSON including **`access_token`** — set it as **`KITE_ACCESS_TOKEN`** for the tradingagents Kite client.

Details and troubleshooting (e.g. checksum / secret mismatches) are in the header comment of `scripts/kite_token_server.py`.

### Observability (Langfuse)

This fork treats **Langfuse** as the primary way to observe agent runs in production-like settings. TradingAgents emits Langfuse traces for LLM calls and tool execution via LangChain callbacks.

The `langchain` package is required for Langfuse’s `CallbackHandler` (per-LLM spans under the root trace). It is listed in `pyproject.toml`; if you use `pip install -r requirements.txt`, `langchain` and `langfuse` are included.

To enable tracing, set:

```bash
export LANGFUSE_ENABLED=true
export LANGFUSE_PUBLIC_KEY=...
export LANGFUSE_SECRET_KEY=...

# Optional (self-hosted or region-specific)
export LANGFUSE_BASE_URL="https://cloud.langfuse.com"

# Optional: a stable user identifier to group usage in Langfuse
export LANGFUSE_USER_ID="user-123"
```

When enabled, each CLI analysis run and each `TradingAgentsGraph.propagate(...)` call creates a root trace whose name is `TradingAgents analysis [<run_suffix>]` (same random `run_suffix` as below). Traces are grouped using `session_id = "<TICKER>:<TRADE_DATE>:<run_suffix>"` where `run_suffix` is random per run (so repeated runs on the same ticker/date do not collide). The same suffix is used to derive a correlated Langfuse trace id. A tag `run:<run_suffix>` is also set for quick filtering.

Validation checklist:
- Open the Langfuse Trace Table and filter by `session_id` or by tag `run:`.
- You should see exactly one root trace per run (name includes the per-run suffix, e.g. `TradingAgents analysis [a1b2c3d4]`).
- The root trace should include tags like `ticker:<TICKER>`, `trade_date:<TRADE_DATE>`, and `llm_provider:<provider>`.
- Inside the trace, Langfuse should show LLM generations and tool calls captured via the LangChain callback handler.

For local models, configure Ollama with `llm_provider: "ollama"` in your config.

Alternatively, copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```
