# TradingAgents — Implementation Roadmap

> Gap analysis against the **Single-Stock Multi-Agent Trading System** paper architecture.
> Derived from audit of the current codebase (April 2026).

---

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Already implemented |
| 🔶 | Partially implemented — needs extension |
| ❌ | Not implemented — greenfield work |

---

## Module 1 · Infrastructure & Orchestration

### 1.1 Asynchronous Message Broker ❌
**What it is:** Replace the current synchronous LangGraph `invoke`/`stream` with an external event bus (e.g. RabbitMQ, Redis Streams) so agents can process tasks in parallel without blocking the main execution loop.

**Current state:** All agents run in-process via LangGraph `StateGraph`. Edges are sequential; one agent finishes before the next starts. No broker, no queue.

**What to build:**
- [ ] Introduce a message broker (RabbitMQ or Redis Streams)
- [ ] Wrap each agent as an async worker that consumes from a topic/queue
- [ ] Allow the four analyst agents to run in **parallel** (they are currently chained sequentially)
- [ ] Implement a barrier/aggregation step before the researchers consume all four reports

---

### 1.2 LiteLLM Router 🔶
**What it is:** A runtime multi-provider router that selects the cheapest/fastest available provider, retries across providers on failure, and shields agent code from per-SDK differences.

**Current state:** `llm_clients/factory.py` selects **one provider at config time**. `openai_client.py` has a custom retry loop (429/5xx/JSON errors). `invoke_fallback.py` provides a deep→quick model fallback for research/risk nodes only.

**What to build:**
- [ ] Integrate [LiteLLM](https://github.com/BerriAI/litellm) as the unified call layer
- [ ] Configure a priority list of providers (e.g. `gpt-o3 → claude-3.7 → gemini-2.5`)
- [ ] Delegate all retry, backoff, and provider-error handling to LiteLLM
- [ ] Remove the hand-rolled retry loops in `UnifiedChatOpenAI` once LiteLLM covers them
- [ ] Preserve the existing `deep_think_llm` / `quick_think_llm` distinction

---

### 1.3 JSON Schema Enforcer ❌
**What it is:** Force all LLM outputs that feed downstream code to conform to a strict JSON schema, preventing parsing failures during automated signal extraction and feature engineering.

**Current state:** `validators.py` only validates model *names*. All agent outputs are free-form text. The trader enforces `FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**` by prompt convention only — a regex, not a schema.

**What to build:**
- [ ] Define Pydantic models for each structured output: `TradeProposal`, `AnalystReport`, `RiskAssessment`, `DebateRound`
- [ ] Use LangChain `.with_structured_output(schema)` on every node that feeds downstream code
- [ ] Add a validation wrapper that catches schema violations and re-prompts (up to N retries)
- [ ] Store validated outputs as typed objects in `AgentState` instead of raw strings

---

### 1.4 Rolling Context Window Manager 🔶
**What it is:** A module that maintains a fixed-length sequence of the last 7–10 days of prices and agent decisions in the prompt, giving the LLM temporal context without ballooning token usage.

**Current state:** `llm_max_input_tokens` trims analyst **message lists** (newest-first, ~4 chars/token heuristic). Researchers, trader, and risk debators build giant f-string prompts with no trimming. `FinancialSituationMemory` is BM25 retrieval of past text situations — not a structured rolling buffer of numeric observations.

**What to build:**
- [ ] Extend trimming to **all** agent nodes (researchers, trader, risk debators), not just the four analysts
- [ ] Build a `RollingObservationBuffer` that stores `(date, OHLCV, signal)` tuples for the last N days
- [ ] Serialize the buffer as a compact table and inject it into every agent's system prompt
- [ ] Make the window size configurable (default: 7 days) in `DEFAULT_CONFIG`
- [ ] Add proper token counting (tiktoken / provider-specific tokenizer) to replace the 4-chars heuristic

---

## Module 2 · Agentic Intelligence Layer

### 2.1 Technical Analyst 🔶
**What it is:** An agent that interprets **normalized** technical indicators (MACD, RSI, KDJ) to detect trading patterns.

**Current state:** `market_analyst.py` exists and calls `get_indicators` for MACD, RSI, Bollinger, ATR, SMA, EMA, VWMA, MFI. Indicators are **not normalized**. **KDJ is missing**.

**What to build:**
- [ ] Add **KDJ** (`k`, `d`, `j` from stockstats) to the indicator allowlist in `y_finance.py`, `kite_indicator.py`, and `alpha_vantage_indicator.py`
- [ ] Add a **normalization layer** in `stockstats_utils.py` that min-max scales or z-scores each indicator over its rolling window before returning values
- [ ] Update the market analyst system prompt to reflect normalized value ranges (0–1 or z-score)

---

### 2.2 Quantitative Analyst ❌
**What it is:** A dedicated agent focused on professional accounting sub-tasks — inventory turnover period, DuPont decomposition, ROE/ROA trend analysis, and other quant finance ratios.

**Current state:** No dedicated quant agent. The fundamentals analyst fetches ROE/ROA as raw `yfinance` fields but does no ratio decomposition or trend analysis.

**What to build:**
- [ ] Create `tradingagents/agents/analysts/quant_analyst.py`
- [ ] Define tools: `get_roe_roa_trend`, `get_inventory_turnover`, `get_dupont_decomposition`, `get_liquidity_ratios`
- [ ] Implement these tools in `tradingagents/dataflows/` using income statement + balance sheet data
- [ ] Wire the quant analyst into `graph/setup.py` as a fifth selectable analyst
- [ ] Add `"quant"` to the `selected_analysts` list in `DEFAULT_CONFIG`

---

### 2.3 News & Sentiment Analyst 🔶
**What it is:** An agent that performs fine-grained semantic nuance analysis of filtered, anonymized news headlines.

**Current state:** `social_media_analyst.py` + `news_analyst.py` exist. Both pass raw headlines directly to the LLM. Semantic nuance comes purely from the model's reading — no sentiment scoring, no filtering pipeline, no anonymization.

**What to build:**
- [ ] Add a **pre-filter step** that scores headlines by relevance to the ticker before they enter the agent prompt
- [ ] Integrate the anonymization scrubber (see Module 3.2) into the news pipeline
- [ ] Optionally add a lightweight sentiment score (e.g. FinBERT embedding) as a structured field alongside each headline

---

### 2.4 Macro Analyst 🔶
**What it is:** A dedicated agent that reads structured macro time series — interest rates, inflation, GDP — rather than inferring macro context from news articles.

**Current state:** `news_analyst.py` mentions macroeconomics and uses `get_global_news`, but there is no dedicated macro agent and no structured time-series tools for rates/inflation/GDP.

**What to build:**
- [ ] Create `tradingagents/agents/analysts/macro_analyst.py`
- [ ] Add data tools: `get_interest_rates`, `get_inflation_data`, `get_gdp_data` backed by FRED API (or yfinance macro tickers like `^TNX`, `^FVX`)
- [ ] Wire into `graph/setup.py` as a selectable analyst
- [ ] Add `"macro"` to `selected_analysts` options in `DEFAULT_CONFIG`

---

### 2.5 Adversarial Debate Logic 🔶
**What it is:** A Debate Facilitator that manages n-rounds of structured back-and-forth between Bull and Bear researchers before reaching consensus.

**Current state:** Bull ↔ Bear alternation exists, gated by `max_debate_rounds` (default = 1 round = one Bull + one Bear turn). `Research Manager` acts as a post-debate summarizer/judge, not a live moderator.

**What to build:**
- [ ] Promote `Research Manager` to an **active facilitator** role that poses a targeted rebuttal question to each side after each round (not just summarizes at the end)
- [ ] Add a **structured debate transcript** object (list of `{speaker, argument, evidence_refs}`) stored in `AgentState`
- [ ] Make it easy to run 3–5 debate rounds via config without hitting context limits (tie into the rolling context window)

---

### 2.6 CRO Veto Gate 🔶
**What it is:** A Chief Risk Officer class that vetoes or downsizes a trade proposal based on **programmatic** checks of portfolio beta, liquidity ratios, and sector exposure — not just LLM text judgment.

**Current state:** Three risk debators (Aggressive/Conservative/Neutral) + a Risk Judge exist. The Risk Judge's final `BUY/SELL/HOLD` is a pure LLM text decision with no hard programmatic thresholds or veto logic.

**What to build:**
- [ ] Add a `CROVetoGate` class in `tradingagents/agents/risk_mgmt/cro_veto.py`
- [ ] Implement hard-coded checks (configurable thresholds):
  - Portfolio Beta > 1.5 → reduce position size / veto
  - Bid-ask spread / volume liquidity ratio above threshold → veto MARKET order
  - Sector concentration > X% → flag
- [ ] Run the veto gate **after** the Risk Judge, before writing `final_trade_decision`
- [ ] If vetoed, write reason to `AgentState` and emit `HOLD` with explanation

---

## Module 3 · Anonymization & Data Engineering

### 3.1 Ticker Mapping Engine ❌
**What it is:** A dictionary-based module that replaces real tickers with synthetic IDs (e.g. `AAPL → STOCK_0026`) in all agent prompts to mitigate LLM memorization bias.

**Current state:** Tickers flow through verbatim in all prompts and tool outputs. The only symbol mapping is Kite's instrument token resolution for broker API calls.

**What to build:**
- [ ] Create `tradingagents/anonymization/ticker_map.py`
- [ ] Generate a deterministic, reversible mapping (real ticker → `STOCK_XXXX`)
- [ ] Apply the mapping in `Propagator.create_initial_state` before the ticker enters `AgentState`
- [ ] Strip real ticker from all tool output strings (OHLCV column headers, `yfinance` metadata fields)
- [ ] Reverse-map only in the final result/logging layer

---

### 3.2 Proper Noun Scrubber ❌
**What it is:** A text processing step that replaces company names, executive names, and product names (e.g. "Elon Musk" → "CEO of AUTO_COMPANY_0003") in news headlines before they reach an LLM.

**Current state:** Headlines go to the LLM verbatim. No entity scrubbing, no Knowledge Graph integration.

**What to build:**
- [ ] Create `tradingagents/anonymization/noun_scrubber.py`
- [ ] Integrate [Google Knowledge Graph Search API](https://developers.google.com/knowledge-graph) (or spaCy NER as a free alternative) to detect named entities in headline text
- [ ] Build a replacement map: person names → generic role descriptions, product names → generic category labels
- [ ] Apply scrubber to all `get_news` and `get_global_news` tool outputs before they are returned to agent prompts
- [ ] Add `GOOGLE_KG_API_KEY` to `.env.example`

---

### 3.3 60-Indicator Library ❌
**What it is:** Pre-calculate 60 standard technical indicators from raw OHLCV data to provide fine-grained inputs for the Technical Agent.

**Current state:** The indicator allowlist is **~13 indicators** (SMA-50, SMA-200, EMA-10, MACD/MACDS/MACDH, RSI, Bollinger upper/mid/lower, ATR, VWMA, MFI). `stockstats` can compute more but nothing exposes or pre-calculates a wider set.

**What to build:**
- [ ] Create `tradingagents/dataflows/indicator_library.py` with a catalog of 60 indicators across categories:
  - **Trend** (SMA-5/10/20/50/100/200, EMA-5/10/20/50, WMA, DEMA, TEMA, HMA, KAMA)
  - **Momentum** (RSI-14, MACD, Signal, Histogram, ROC, MOM, CCI, Williams %R, KDJ, Stochastic, StochRSI)
  - **Volatility** (ATR, Bollinger Bands, Keltner Channel, Donchian Channel, Historical Vol)
  - **Volume** (OBV, VWAP, VWMA, MFI, CMF, AD, Force Index, VROC)
  - **Trend strength** (ADX, +DI, -DI, Aroon Up/Down, PSAR)
  - **Oscillators** (DPO, TRIX, UO, PPO, Detrended Price)
- [ ] Expose via a single `compute_indicators(df, indicator_names)` function
- [ ] Update `best_ind_params` allowlists in all vendor files to reference this catalog
- [ ] Update the market analyst prompt to reflect the full catalog

---

## Module 4 · High-Fidelity Simulation Engine

### 4.1 Limit Order Book (LOB) Simulator ❌
**What it is:** An order-matching engine that processes MARKET, LIMIT, and STOP orders with price-time priority, simulating realistic fills.

**Current state:** `PaperLedger` is a simple cash/shares book that executes BUY/SELL at the day's raw close price. No order types, no partial fills, no LOB mechanics.

**What to build:**
- [ ] Create `tradingagents/simulation/order_book.py` with a `LimitOrderBook` class
- [ ] Implement order types: `MarketOrder`, `LimitOrder`, `StopOrder`
- [ ] Match against the OHLCV-reconstructed order flow using price-time priority
- [ ] Model partial fills (volume-limited) and price impact (linear or square-root model)
- [ ] Replace `PaperLedger.apply_signal` with `OrderBook.submit(order)` in `backtest/runner.py`
- [ ] Keep `PaperLedger` as a simple fallback mode

---

### 4.2 Temporal Partitioning Controller ❌
**What it is:** A strict knowledge cutoff that ensures no agent ever sees data beyond `t−1` (yesterday) relative to the simulation trade date.

**Current state:** `trade_date` is passed in prompts as "current date" but tools don't clamp data to `≤ trade_date`. `stockstats_utils.py` downloads data up to the **real system clock today** — not the simulation date. LLM-supplied tool call date ranges are not validated.

**What to build:**
- [ ] Add a `simulation_date` parameter to all data tool functions (`get_stock_data`, `get_indicators`, `get_news`, etc.)
- [ ] In every vendor implementation, clamp `end_date = min(requested_end, simulation_date - 1 day)`
- [ ] Fix `stockstats_utils.py` to use `simulation_date` as `end_date` instead of `pd.Timestamp.today()`
- [ ] Add a validation wrapper in `interface.py`'s `route_to_vendor` that enforces the cutoff before any vendor call
- [ ] Add tests that assert no returned data row has a date ≥ `trade_date`

---

### 4.3 Transaction Cost Model ❌
**What it is:** A cost calculator that subtracts a configurable number of basis points per trade to ensure all reported returns are net-of-cost.

**Current state:** `PaperLedger` executes at raw close price with zero commissions, spread, or slippage. All reported returns are gross.

**What to build:**
- [ ] Add `cost_bps: float = 10` parameter to `PaperLedger` (and `DEFAULT_CONFIG`)
- [ ] Deduct `cost_bps / 10_000 * trade_value` from cash on every BUY and SELL execution
- [ ] Add a `slippage_model` (e.g. linear: `slippage_bps * sqrt(order_size / avg_volume)`) as an optional second layer
- [ ] Report gross vs. net return separately in `summary.json`
- [ ] Add `KITE_BROKERAGE_BPS` env var for Indian market defaults (STT + brokerage + exchange charges ≈ 10–20 bps)

---

## Module 5 · Evaluation & Analytics Suite

### 5.1 Financial Performance Engine 🔶
**What it is:** Standard ROI, Annualized Return, Sharpe Ratio, and Maximum Drawdown calculators.

**Current state:** `runner.py` computes `total_return` and `max_drawdown`. Annualized return and Sharpe Ratio are **not implemented**.

**What to build:**
- [ ] Add `annualized_return(equity_series, start_date, end_date)` to `backtest/runner.py`
- [ ] Add `sharpe_ratio(daily_returns, risk_free_rate=0.0)` (annualized, using daily returns series)
- [ ] Add `sortino_ratio(daily_returns)` and `calmar_ratio(annualized_return, max_drawdown)` as bonus metrics
- [ ] Include all metrics in `summary.json` and the printed backtest report

---

### 5.2 Explainable Stability Tracker ❌
**What it is:** A textual comparison tool that measures **Rationale Persistence** — whether the causal story (evidence → analyst signal → researcher stance → trader decision → final order) remains consistent through the pipeline.

**Current state:** Per-date JSON state logs (`full_states_log_<date>.json`) store all intermediate reports, but no tool compares them for consistency. `Reflector` does retrospective narrative memory, not a forward consistency check.

**What to build:**
- [ ] Create `tradingagents/analytics/rationale_tracker.py`
- [ ] Define a `RationaleChain` data class: `{analyst_signals[], researcher_stance, trader_plan, risk_decision, final_signal}`
- [ ] After each run, extract the chain from `AgentState` and use an LLM (or embedding similarity) to score directional consistency between layers
- [ ] Flag "rationale breaks" (e.g. bull analyst → bear researcher → BUY trader) and log them
- [ ] Persist per-date rationale chains to `eval_results/<ticker>/rationale_chains/` for later review

---

### 5.3 CBS Calculator ❌
**What it is:** The **Coordination Breakeven Spread** — a formula that determines whether the alpha generated by multi-agent coordination justifies the latency cost (extra slippage from API call time).

**Current state:** Not implemented. No reference to CBS anywhere in the codebase.

**Formula:**
```
CBS = (avg_latency_seconds × daily_price_volatility_per_second) + cost_bps
```
If `expected_signal_alpha < CBS` → single-agent mode is preferred.

**What to build:**
- [ ] Create `tradingagents/analytics/cbs_calculator.py`
- [ ] Instrument each agent graph run to record per-node LLM call latency (already partially available via Langfuse)
- [ ] Compute intraday price volatility (σ per second) from OHLCV data
- [ ] Implement the CBS formula and compare against back-tested alpha
- [ ] Add CBS to `summary.json` output

---

### 5.4 Visual Diagnostics Dashboard ❌
**What it is:** An interactive frontend to visualize equity curves, candlestick charts with agent decision overlays, and hover-able agent reasoning logs.

**Current state:** No dashboard. Outputs are CSV/JSON files and optional Langfuse traces. `backtrader` is in `pyproject.toml` but never imported.

**What to build:**
- [ ] Create `dashboard/` directory with a Plotly Dash or Streamlit app
- [ ] **Equity curve panel:** line chart of portfolio value over time, with buy/sell markers
- [ ] **Candlestick panel:** OHLCV candles with overlaid technical indicator plots and entry/exit markers
- [ ] **Agent reasoning panel:** hover over any trade marker to see the full analyst → researcher → trader → risk rationale chain for that date
- [ ] **Metric summary panel:** ROI, Annualized Return, Sharpe, MDD, CBS side-by-side
- [ ] Load data from `eval_results/<ticker>/equity.csv` + `full_states_log_*.json`
- [ ] Add `plotly` and `dash` (or `streamlit`) to `pyproject.toml`

---

## Implementation Priority Order

> Suggested sequencing based on dependencies and impact.

| Phase | Items | Rationale |
|-------|-------|-----------|
| **Phase 1 — Foundation** | 4.2 Temporal Cutoff, 4.3 Cost Model, 5.1 Performance Metrics | Fix backtest integrity first — all analytics are invalid without correct simulation |
| **Phase 2 — Data** | 3.3 60-Indicator Library, 2.1 Technical Analyst (KDJ + norm), 2.4 Macro Analyst | Richer, cleaner data for agents before expanding agent count |
| **Phase 3 — Agent Quality** | 1.3 JSON Schema, 1.4 Context Window (all nodes), 2.2 Quant Analyst, 2.6 CRO Veto Gate | Structured outputs and proper token management before adding more agents |
| **Phase 4 — Anonymization** | 3.1 Ticker Map, 3.2 Noun Scrubber | Apply to all prompts once agent/data layer is stable |
| **Phase 5 — Simulation** | 4.1 LOB Simulator | High-fidelity execution layer; depends on cost model and temporal cutoff |
| **Phase 6 — Analytics** | 5.2 Rationale Tracker, 5.3 CBS Calculator, 5.4 Dashboard | Evaluation suite; depends on all upstream modules being stable |
| **Phase 7 — Infra** | 1.1 Async Broker, 1.2 LiteLLM Router | Optimization pass — current synchronous LangGraph works, just slower |

---

## File Map (New Files to Create)

```
tradingagents/
├── anonymization/
│   ├── __init__.py
│   ├── ticker_map.py            # 3.1
│   └── noun_scrubber.py         # 3.2
├── agents/
│   └── analysts/
│       ├── quant_analyst.py     # 2.2
│       └── macro_analyst.py     # 2.4
├── agents/
│   └── risk_mgmt/
│       └── cro_veto.py          # 2.6
├── dataflows/
│   └── indicator_library.py     # 3.3
├── simulation/
│   ├── __init__.py
│   └── order_book.py            # 4.1
└── analytics/
    ├── __init__.py
    ├── rationale_tracker.py     # 5.2
    └── cbs_calculator.py        # 5.3

dashboard/
├── app.py                       # 5.4
├── panels/
│   ├── equity_curve.py
│   ├── candlestick.py
│   └── reasoning_log.py
└── requirements.txt
```

---

*Last updated: April 10, 2026*
