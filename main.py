import os
from pathlib import Path

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG


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

# Create a custom config
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = os.getenv("LLM_PROVIDER", "openrouter")
config["quick_think_llm"] = os.getenv("QUICK_THINK_LLM", "openrouter/free")
config["deep_think_llm"] = os.getenv("DEEP_THINK_LLM", "openrouter/free")
config["max_debate_rounds"] = 1

# Configure data vendors (default uses yfinance, no extra API keys needed)
config["data_vendors"] = {
    "core_stock_apis": "yfinance",
    "technical_indicators": "yfinance",
    "fundamental_data": "yfinance",
    "news_data": "yfinance",
}

# Initialize with custom config
ta = TradingAgentsGraph(debug=True, config=config)

# forward propagate (hard-coded ticker + as-of date for now)
TICKER = "NVDA"
TRADE_DATE = "2024-05-10"
_, decision = ta.propagate(TICKER, TRADE_DATE)
print(decision)

# Memorize mistakes and reflect
# ta.reflect_and_remember(1000) # parameter is the position returns
