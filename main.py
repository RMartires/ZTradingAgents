import logging
import os
from pathlib import Path

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from cli.stats_handler import StatsCallbackHandler
from tradingagents.observability.langfuse_config import (
    get_langfuse_client,
    get_langfuse_handler,
    langfuse_trace_display_name,
    new_langfuse_run_correlation,
    shutdown_langfuse,
)


def _load_dotenv(path: str | None = None) -> None:
    """Load KEY=VALUE pairs from a .env file without python-dotenv (avoids editor/Pyright issues on some venvs)."""
    env_path = Path(path) if path else Path(__file__).resolve().parent / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            os.environ.setdefault(key, value)


# Load environment variables from .env (needs OPENROUTER_API_KEY for OpenRouter)
_load_dotenv()

_level_name = os.getenv("LOG_LEVEL", "WARNING").upper()
_level = getattr(logging, _level_name, logging.WARNING)
if not logging.getLogger().handlers:
    logging.basicConfig(level=_level, format="%(levelname)s %(name)s: %(message)s")

# Create a custom config
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = os.getenv("LLM_PROVIDER", "openrouter")
config["quick_think_llm"] = os.getenv("QUICK_THINK_LLM", "openrouter/free")
config["deep_think_llm"] = os.getenv("DEEP_THINK_LLM", "openrouter/free")
config["max_debate_rounds"] = 1

# Optional: HTTP-level retries in the OpenAI SDK (5xx / 429 on the wire). Default SDK behavior is 2.
if os.getenv("LLM_MAX_RETRIES", "").strip():
    config["llm_max_retries"] = int(os.getenv("LLM_MAX_RETRIES", "2"))
# Optional: per-request timeout in seconds for the LLM HTTP client
if os.getenv("LLM_TIMEOUT", "").strip():
    config["llm_timeout"] = float(os.getenv("LLM_TIMEOUT", "600"))
if os.getenv("LLM_RATE_LIMIT_RPM", "").strip():
    config["llm_rate_limit_rpm"] = float(os.getenv("LLM_RATE_LIMIT_RPM", "0"))
if os.getenv("LLM_MAX_TOKENS", "").strip():
    config["llm_max_tokens"] = int(os.getenv("LLM_MAX_TOKENS", "8192"))
if os.getenv("LLM_MAX_INPUT_TOKENS", "").strip():
    config["llm_max_input_tokens"] = int(os.getenv("LLM_MAX_INPUT_TOKENS", "0"))
# Optional: sampling temperature only for schemas.outputs structured JSON invokes
# (see tradingagents.llm_clients.invoke_fallback.bound_llm_for_structured_output). Not loaded into config.

# Configure data vendors (default uses yfinance, no extra API keys needed)
config["data_vendors"] = {
    "core_stock_apis": "kite",
    "technical_indicators": "kite",
    "fundamental_data": "yfinance",
    "news_data": "alpha_vantage",
}

# Create stats callback handler for tracking LLM/tool calls
stats_handler = StatsCallbackHandler()

langfuse_client = get_langfuse_client()
langfuse_handler = None
root_span = None
trace_cm = None
propagate_cm = None

TICKER = "RELIANCE"
TRADE_DATE = "2024-04-05"

if langfuse_client is not None:
    from langfuse import propagate_attributes

    corr = new_langfuse_run_correlation(ticker=TICKER, trade_date=TRADE_DATE)
    trace_display_name = langfuse_trace_display_name(corr.run_suffix)
    trace_input = {
        "company_name": TICKER,
        "trade_date": TRADE_DATE,
        "llm_provider": config.get("llm_provider"),
        "quick_think_llm": config.get("quick_think_llm"),
        "deep_think_llm": config.get("deep_think_llm"),
    }
    trace_kwargs = dict(
        as_type="span",
        name=trace_display_name,
        input=trace_input,
    )
    tc = corr.trace_context
    if tc is not None:
        trace_kwargs["trace_context"] = tc

    trace_cm = langfuse_client.start_as_current_observation(**trace_kwargs)
    root_span = trace_cm.__enter__()
    tags = [
        f"ticker:{TICKER}",
        f"trade_date:{TRADE_DATE}",
        f"run:{corr.run_suffix}",
        f"llm_provider:{config.get('llm_provider')}",
        f"quick_model:{config.get('quick_think_llm')}",
        f"deep_model:{config.get('deep_think_llm')}",
    ]
    propagate_cm = propagate_attributes(
        trace_name=trace_display_name,
        session_id=corr.session_id,
        user_id=os.getenv("LANGFUSE_USER_ID"),
        tags=tags,
    )
    propagate_cm.__enter__()

    langfuse_handler = get_langfuse_handler()

callbacks = [stats_handler]
if langfuse_handler is not None:
    callbacks.append(langfuse_handler)

# Initialize with custom config
ta = TradingAgentsGraph(debug=True, config=config, callbacks=callbacks)

# forward propagate (hard-coded ticker + as-of date for now)
_, decision = ta.propagate(TICKER, TRADE_DATE)
print(decision)

# Memorize mistakes and reflect
# ta.reflect_and_remember(1000) # parameter is the position returns

if langfuse_client is not None:
    try:
        if root_span is not None:
            out = {"processed_signal": str(decision)[:200]}
            root_span.set_trace_io(input=trace_input, output=out)
            root_span.update(output=out)
    finally:
        if propagate_cm is not None:
            propagate_cm.__exit__(None, None, None)
        if trace_cm is not None:
            trace_cm.__exit__(None, None, None)
        shutdown_langfuse()
