# LLM Trading Eval Harness — Blueprint v0.2

*Project: **agent-alpha-bench**.*

## 1. One-line pitch

A reproducible benchmark that puts LLMs in the seat of a medium-term equity investor under identical data, rules and costs, and measures not just who made money but *whether their judgement was any good*: calibration of stated confidence, sensitivity to what they were shown, and performance against dumb controls.

TradeRank.ai is the reference point for the competition rules and the leaderboard UX. The differentiator is that this is an **evaluation harness**, not a race.

## 2. Decisions taken so far

| Topic | Decision |
|---|---|
| Framing | Eval harness (ablations, calibration, non-LLM controls) |
| Asset universe | US equities only: the 50 largest US-domiciled companies by market cap at 2024-12-31 (approximate snapshot in `alphabench/universe.py`, to be verified); daily screener shows the top 5 plus held names |
| History | Backtest replay from 2024-01-01 with point-in-time data; live daily cron afterwards |
| Sides | Long **and** short, TradeRank-style (no borrow cost modelled) |
| LLM access | To be evaluated; local/offline (Ollama-style) iteration is the priority criterion |
| First deliverable | `notebooks/01_data_and_engine.ipynb` — data, payload, schema, engine, controls replay, look-ahead checks; no LLM calls (**done**) |
| Home | GitHub: `Barrelet/agent-alpha-bench` (conda env `alphabench`) |

## 3. Competition rules (v1, adapted from TradeRank)

Keep these close to TradeRank so results are comparable and the rules are defensible; deviate only where equities-only makes something moot.

- **Capital**: $10,000 paper, long/short, no leverage (gross exposure ≤ equity, fees included).
- **Cycle**: one decision per US market day at the close (16:00 ET). Orders fill at the *next* open (see §6 on look-ahead).
- **Universe**: 50 largest US-domiciled companies by market cap at 2024-12-31 (membership look-ahead for the 2024 part of the replay is documented; point-in-time constituents are the planned upgrade).
- **Screener**: deterministic. Rank by 20-day average dollar volume × (1 + |10-day return|); show the top 5 plus any held names.
- **Model inputs per cycle**: OHLCV candles (26 weekly, 30 daily; drop the 4-hour bars for v1), RSI-14 on each timeframe, universe summary table (price, 1d/10d/30d returns, 30d vol, RSI, distance from 30d high, volume, next earnings date), portfolio snapshot with carried-forward theses, SPY as non-tradeable benchmark.
- **Actions**: `open_long`, `open_short`, `add` (winners only), `close` (partial/full), `hold`.
- **Per-action fields**: `symbol`, `action`, `percent_of_equity` (10–100), `confidence` (0–1), `thesis`, `invalidation`, `invalidation_price`.
- **Constraints**: max 10 positions, max 1 new position per cycle, confidence ≥ 0.80 to open, no averaging down, no re-entry same cycle.
- **Costs**: 0.1% per side. No slippage in v1 (state it).
- **Invalidation monitor**: TradeRank checks every 15 min; v1 checks once per day against the bar's low/high and fills at the invalidation price (conservative, simple).
- **Output**: strict JSON, schema-validated; one repair attempt, else the cycle is treated as `hold`.

## 4. What makes it an eval harness

These are the features that turn a leaderboard into a portfolio piece. Each is a column or a page in the app.

1. **Non-LLM controls** on the same rules: buy-and-hold SPY, equal-weight universe, 10-day momentum rule, random-with-same-constraints. Any LLM that cannot beat these has nothing to say. Later: TimesFM zero-shot as a forecasting control (reuse from the oil lab).
2. **Confidence calibration**: bucket stated `confidence` vs realised P&L sign. Reliability diagram + Brier score per model. This is the single most interesting chart in the project.
3. **Ablations** (same model, different prompt variants): no RSI; no summary table; no carried-forward theses; "trader" mandate vs "investor" mandate; temperature 0 vs default. Measures what the model actually uses.
4. **Reproducibility**: pinned model IDs, temperature 0, every prompt and response cached to disk with a content hash. Rerunning a season should give byte-identical results for deterministic models.
5. **Cost-adjusted view**: return per $ of inference spend. A $0.50 open-weight model that ties a frontier model is a finding.
6. **Open dataset**: every cycle, prompt, response, fill and equity mark published as JSON/Parquet.

## 5. Architecture

```
agent-alpha-bench/
├── data/                 # cached raw prices, prompts, responses, results (gitignored except samples)
├── alphabench/
│   ├── universe.py       # ticker list, sector map, earnings calendar
│   ├── market.py         # yfinance fetch, point-in-time slicing, RSI, weekly resample
│   ├── screener.py       # deterministic ranking
│   ├── prompt.py         # builds the cycle payload (dict) and renders it to text
│   ├── schema.py         # pydantic models for decisions; validation + repair contract
│   ├── engine.py         # Portfolio, fills, fees, invalidation, mark-to-market
│   ├── agents/
│   │   ├── base.py       # Agent.decide(payload) -> Decision
│   │   ├── rules.py      # controls: hold_spy, equal_weight, momentum, random
│   │   └── llm.py        # provider-agnostic LLM agent with caching
│   ├── metrics.py        # return, Sharpe, max DD, win rate, calibration, Brier
│   ├── null.py           # 1,000 random traders under the same rules → percentile of every run (skill vs luck)
│   ├── compare.py        # model/prompt configurations for notebook 03 and scripts/run_compare.py
│   └── replay.py         # walk the calendar, run all agents, force-close at window end, persist results
├── notebooks/
│   └── 01_data_and_engine.ipynb   # ← first deliverable
├── app/
│   └── streamlit_app.py  # leaderboard, model pages, calibration, ablations
├── scripts/
│   └── daily_cycle.py    # cron entry point
└── tests/
```

Data flow: `market` → `screener` → `prompt` (payload) → `agent` → `schema` (validate) → `engine` (execute, mark) → `metrics` → app. The payload dict is the contract: rule agents read it as numbers, the LLM agent renders it to text.

## 6. Look-ahead and bias discipline (carry over from the oil lab)

- Every cycle sees only bars with `date ≤ decision_date`. Build the payload from a sliced frame, never from the full history.
- Decide at close *t*, fill at open *t+1*. No same-bar fills.
- Earnings dates come from a snapshot taken at replay build time; note that yfinance's calendar is not point-in-time.
- The fixed universe has survivorship bias; document it and keep the list of names that were large caps throughout the replay window.
- RSI and the screener use only past data (rolling windows, no centering).
- Replay window: start with 2025-01-01 → today so the SPY control has a bull-and-chop mix.

## 7. LLM provider evaluation (to do before spending money)

Criteria in priority order: local/offline iteration, reproducibility (pinned versions, temperature 0), model breadth, cost per cycle.

Candidates to assess in the blueprint's next revision:

- **Ollama** (local): free, offline, fully reproducible; limited to open-weight models that fit on the Mac; slow for 30-daily-candle prompts × 50 tickers. Good for developing prompt/schema and running ablations cheaply.
- **OpenRouter**: one key, widest breadth incl. open-weight and frontier; version pinning varies by provider.
- **Direct provider APIs** (OpenAI, Anthropic, Google, Mistral): best pinning and structured-output support; more keys to manage.
- **Hybrid**: Ollama for iteration and ablations, one paid route for the headline leaderboard.

The `agents/llm.py` adapter should expose `complete(messages, model, temperature, json_schema) -> str` so the backend is swappable and cached.

Prompt size is the main cost driver: ~5 screened + held names × (30 daily + 26 weekly candles) ≈ 8–12k tokens. Trimming candles is the first lever.

## 8. Metrics (leaderboard columns)

Total return, annualised Sharpe (daily, rf = 0), max drawdown, win rate, number of trades, average holding days, exposure %, return vs SPY buy-and-hold, Brier score, inference cost, invalid-output rate.

## 9. Roadmap

1. **Notebook 01** — data fetch and cache, point-in-time slicing, indicators, screener, payload builder, schema, engine, four control agents, replay 2024→today, leaderboard, equity curves, calibration machinery, look-ahead checks. *(done — 8 Sep 2026)*
2. **Schema + LLM agent** — pydantic schema, prompt renderer, repair loop, Ollama backend, response cache. Run one local model on 20 cycles.
3. **Provider evaluation** — cost per cycle measured empirically; pick the paid route for the headline run.
4. **Full replay** — 6–8 models + 4 controls over the window. Calibration and ablation runs.
5. **Streamlit app** — leaderboard, model page (equity curve, positions, reasoning log), calibration page, ablation page, methodology page.
6. **Live** — daily cron (GitHub Actions or a small VPS), append to dataset, redeploy.
7. **Write-up** — README with methodology and findings; the findings are the LinkedIn post.

## 10. Open questions for later

- Verify the 50-name list against a point-in-time market-cap source.
- Short borrow cost: ignore (current), or charge a flat annualised fee?
- Do we keep TradeRank's "investor mandate" wording verbatim for comparability, or write our own?
- Season structure (monthly resets like TradeRank) or one continuous run?
