"""Collect the numbers notebook 04 tells its story from into data/summary/ (small CSVs
that are committed, unlike the rest of data/), so the story notebook runs in seconds
on any machine and does not change when the experiments are re-run.

    python scripts/export_summary.py

Reads the two whole-universe runs (data/results/compare_universe, the leaderboard rules;
data/results/compare_universe_cap25, the 25% position limit), their controls, the cached
random-trader distributions, and the price cache for the two benchmarks.
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alphabench.universe import ALL_SYMBOLS, TICKERS, BENCHMARK
from alphabench.market import load_prices
from alphabench.metrics import equal_weight_index, equity_metrics

RES = ROOT / "data" / "results"
OUT = ROOT / "data" / "summary"
OUT.mkdir(parents=True, exist_ok=True)

RUNS = {  # run id -> (results folder, config suffix, label for the story)
    "leaderboard_rules": ("compare_universe", "_all50", "leaderboard rules (no position limit)"),
    "position_limit":    ("compare_universe_cap25", "_all50_cap25", "25% position limit"),
}
PROMPT_LABEL = {"v1": "Prompt 1 — leaderboard mandate", "v2": "Prompt 2 — plus risk rules", "v3": "Prompt 3 — full decision procedure"}
# model slug -> (story label, prefix put in front of the prompt label so the agent name stays unique).
# The local model keeps the bare prompt label; every other model is prefixed, e.g. "GPT-5.1, Prompt 1 — leaderboard mandate".
MODELS = {"qwen3-8b": ("Qwen3 8B (laptop)", ""), "gpt-5-1": ("GPT-5.1 (OpenAI API)", "GPT-5.1, ")}
CONTROL_LABEL = {"control_momentum_10d_long": "Momentum rule (long only)", "control_momentum_10d_ls": "Momentum rule (long and short)"}

md = load_prices(ALL_SYMBOLS, "2023-09-01", None, ROOT / "data" / "prices.parquet")

equity_rows, trade_rows, agent_rows = [], [], []
meta = {"runs": {}}
for run, (folder, sfx, label) in RUNS.items():
    base = RES / folder
    if not base.exists():
        print(f"skip {run}: {base} not found"); continue
    agents = {f"{slug}_{v}{sfx}": (v, prefix + PROMPT_LABEL[v], model_label)
              for slug, (model_label, prefix) in MODELS.items() for v in ("v1", "v2", "v3")}
    model_of, no_action = {}, {}
    eq_ctrl = pd.read_parquet(base / "controls" / "equity.parquet")
    tr_ctrl = pd.read_parquet(base / "controls" / "trades.parquet")
    series, trades = {}, {}
    for name, (v, lab, model_label) in agents.items():
        d = base / name
        if not d.exists():
            print(f"skip {name}"); continue
        series[lab] = pd.read_parquet(d / "equity.parquet").iloc[:, 0]
        trades[lab] = pd.read_parquet(d / "trades.parquet").assign(prompt=v)
        model_of[lab] = model_label
        if (d / "decisions.jsonl").exists():   # days on which the model returned an empty decision list
            recs = [json.loads(line) for line in (d / "decisions.jsonl").read_text().splitlines() if line.strip()]
            no_action[lab] = sum(1 for r in recs if not r.get("decision", {}).get("decisions"))
    for name, lab in CONTROL_LABEL.items():
        series[lab] = eq_ctrl[name]
        trades[lab] = tr_ctrl[tr_ctrl["agent"] == name].copy() if "agent" in tr_ctrl else pd.DataFrame()
    first, last = next(iter(series.values())).index[[0, -1]]
    series["Just hold all 50 stocks equally"] = equal_weight_index(md.close[TICKERS], start=first).loc[:last]
    series["S&P 500 (SPY)"] = md.close[BENCHMARK].loc[first:last] / md.close[BENCHMARK].loc[first] * 10_000
    meta["runs"][run] = {"label": label, "start": str(first.date()), "end": str(last.date()), "cycles": int(len(series[lab]) - 1)}
    for lab, s in series.items():
        s = s.dropna()
        for dt, val in s.items():
            equity_rows.append({"run": run, "agent": lab, "date": str(pd.Timestamp(dt).date()), "equity": round(float(val), 2)})
        m = equity_metrics(s)
        t = trades.get(lab, pd.DataFrame())
        biggest = np.nan
        if len(t):
            eq_at = s.reindex(pd.to_datetime(t["entry_date"])).ffill().to_numpy()
            biggest = float(np.nanmax(t["qty"].to_numpy() * t["entry_price"].to_numpy() / eq_at))
        agent_rows.append({"run": run, "agent": lab, "model": model_of.get(lab, ""),
                           "kind": "prompt" if lab in model_of else "control" if "rule" in lab else "benchmark",
                           "no_action_days": no_action.get(lab, np.nan),
                           "stopped_out": int((t["reason"] == "invalidation").sum()) if len(t) else np.nan,
                           "closed_by_choice": int((t["reason"] == "close").sum()) if len(t) else np.nan,
                           "avg_holding_days": round(float(t["holding_days"].mean()), 1) if len(t) else np.nan,
                           "total_return": round(m["total_return"], 6), "sharpe": round(m["sharpe"], 3), "max_drawdown": round(m["max_drawdown"], 4),
                           "n_trades": int(len(t)), "win_rate": round(float((t["pnl"] > 0).mean()), 3) if len(t) else np.nan,
                           "biggest_bet_share": round(biggest, 3) if biggest == biggest else np.nan,
                           "best_trade_pnl": round(float(t["pnl"].max()), 2) if len(t) else np.nan, "total_pnl": round(float(t["pnl"].sum()), 2) if len(t) else np.nan,
                           "stated_confidence": round(float(t["confidence"].mean()), 3) if len(t) and t["confidence"].notna().any() else np.nan})
    for lab, t in trades.items():
        for r in t.itertuples():
            trade_rows.append({"run": run, "agent": lab, "model": model_of.get(lab, ""), "symbol": r.symbol, "side": "long" if r.side > 0 else "short",
                               "entry_date": str(pd.Timestamp(r.entry_date).date()), "exit_date": str(pd.Timestamp(r.exit_date).date()),
                               "holding_days": int(r.holding_days), "pnl": round(float(r.pnl), 2), "pnl_pct": round(float(r.pnl_pct), 4),
                               "reason": r.reason, "confidence": round(float(r.confidence), 3) if r.confidence == r.confidence else np.nan,
                               "position_usd": round(float(r.qty * r.entry_price), 2)})

pd.DataFrame(equity_rows).to_csv(OUT / "equity.csv", index=False)
pd.DataFrame(trade_rows).to_csv(OUT / "trades.csv", index=False)
pd.DataFrame(agent_rows).to_csv(OUT / "agents.csv", index=False)

# random traders: every cached distribution for the window, tagged
null_rows = []
for p in sorted((RES / "compare" / "null").glob("random_2026-06-01_2026-08-25_n500_*.parquet")):
    df = pd.read_parquet(p)
    tag = p.stem.split("_n500_")[1]
    df = df.assign(variant=tag, run="position_limit" if tag.endswith("_cap25") else "leaderboard_rules",
                   candidates="all 50" if "univ" in tag else "screened five", pyramiding="pyr" in tag,
                   long_only=tag.startswith("long"))
    null_rows.append(df[["run", "variant", "candidates", "long_only", "pyramiding", "seed", "total_return", "sharpe", "max_drawdown", "n_trades", "win_rate"]].round(4))
pd.concat(null_rows, ignore_index=True).to_csv(OUT / "random_traders.csv", index=False)
(OUT / "meta.json").write_text(json.dumps(meta, indent=2))
print("wrote", OUT, "|", len(equity_rows), "equity points,", len(trade_rows), "trades,", len(agent_rows), "agents,", sum(len(x) for x in null_rows), "random traders")
