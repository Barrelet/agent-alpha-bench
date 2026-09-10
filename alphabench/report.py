"""Plain-language charts and tables for the story notebook (04).

Everything here reads `data/summary/` (written by `scripts/export_summary.py`), never the
raw results, so it runs in seconds anywhere and the story does not move when experiments
are re-run. One function per picture; each returns the matplotlib figure so a notebook
cell is one line.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROMPTS = ["Prompt 1 — leaderboard mandate", "Prompt 2 — plus risk rules", "Prompt 3 — full decision procedure"]
INDEX = "Just hold all 50 stocks equally"
SPY = "S&P 500 (SPY)"
RULE = "Momentum rule (long only)"
RUN_LABEL = {"leaderboard_rules": "Leaderboard rules (no position limit)", "position_limit": "With a 25% position limit"}

# categorical slots 1-3 for the three prompts; everything that is not a model is grey
COLOR = {PROMPTS[0]: "#2a78d6", PROMPTS[1]: "#eb6834", PROMPTS[2]: "#1baf7a", INDEX: "#52514e", SPY: "#8a8987", RULE: "#b3b2ae"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({"font.size": 10.5, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titleweight": "normal", "axes.titlesize": 12, "figure.dpi": 110})


@dataclass
class Summary:
    equity: pd.DataFrame
    trades: pd.DataFrame
    agents: pd.DataFrame
    random: pd.DataFrame
    meta: dict

    def window(self, run: str) -> str:
        m = self.meta["runs"][run]
        return f"{pd.Timestamp(m['start']).strftime('%-d %B')} to {pd.Timestamp(m['end']).strftime('%-d %B %Y')}, {m['cycles']} trading days"

    def value(self, run: str, agent: str, col: str):
        a = self.agents
        return a.loc[(a["run"] == run) & (a["agent"] == agent), col].iloc[0]

    def random_returns(self, run: str, pyramiding: bool = False, long_only: bool = False) -> pd.Series:
        r = self.random
        sel = (r["run"] == run) & (r["candidates"] == "all 50") & (r["pyramiding"] == pyramiding) & (r["long_only"] == long_only)
        return r.loc[sel, "total_return"]

    def beat_share(self, run: str, agent: str, **kw) -> float:
        """Share of random traders (in percent) that this agent beat."""
        v = self.value(run, agent, "total_return")
        return float(100 * (self.random_returns(run, **kw) <= v).mean())


def load_summary(root: Path | str) -> Summary:
    d = Path(root) / "data" / "summary"
    return Summary(pd.read_csv(d / "equity.csv", parse_dates=["date"]), pd.read_csv(d / "trades.csv", parse_dates=["entry_date", "exit_date"]),
                   pd.read_csv(d / "agents.csv"), pd.read_csv(d / "random_traders.csv"), json.loads((d / "meta.json").read_text()))


def _tidy(ax, ylabel=""):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)


def _pct(x): return f"{x:+.1%}"


# ---- 1. what happened --------------------------------------------------------------
def _spread(ys: list[float], gap: float) -> list[float]:
    """Push label y-positions apart so none overlap (keeps their order)."""
    order = np.argsort(ys); out = list(ys)
    for k in range(1, len(order)):
        lo, hi = order[k - 1], order[k]
        if out[hi] - out[lo] < gap:
            out[hi] = out[lo] + gap
    return out


def equity_chart(S: Summary, run: str, title: str | None = None):
    """Account value over the window, indexed to 100, three prompts in colour and the
    passive index / S&P 500 / simple rule in grey, with the final return written at the line end."""
    e = S.equity[S.equity["run"] == run].pivot(index="date", columns="agent", values="equity")
    e = e / e.iloc[0] * 100
    fig, ax = plt.subplots(figsize=(11, 5))
    order = [n for n in [INDEX, SPY, RULE] + PROMPTS if n in e]
    ends = [float(e[n].dropna().iloc[-1]) for n in order]
    span = float(e.max().max() - e.min().min())
    ys = _spread(ends, gap=span * 0.055)
    for name, y in zip(order, ys):
        s = e[name].dropna()
        style = dict(lw=2.2) if name in PROMPTS else dict(lw=1.6, ls="--" if name != RULE else ":")
        ax.plot(s.index, s.values, color=COLOR[name], **style)
        ax.text(s.index[-1], y, f"  {name.split(' — ')[0]}  {_pct(S.value(run, name, 'total_return'))}", color=COLOR[name], va="center", fontsize=9.5,
                fontweight="bold" if name in PROMPTS else "normal")
    ax.axhline(100, color=GRID, lw=1)
    ax.set_title(title or f"{RUN_LABEL[run]} — value of a $10,000 account, {S.window(run)}", loc="left")
    ax.set_xlim(e.index[0], e.index[-1] + pd.Timedelta(days=12))
    _tidy(ax, "account value (start = 100)")
    fig.tight_layout()
    return fig


def scoreboard(S: Summary, run: str) -> pd.DataFrame:
    """Five plain columns per player."""
    a = S.agents[S.agents["run"] == run].set_index("agent")
    rows = []
    for name in PROMPTS + [RULE, INDEX, SPY]:
        r = a.loc[name]
        rows.append({"Who": name, "Result": _pct(r["total_return"]), "Trades": int(r["n_trades"]) if r["n_trades"] else "—",
                     "Winning trades": f"{r['win_rate']:.0%}" if r["n_trades"] else "—",
                     "Biggest single bet": f"{r['biggest_bet_share']:.0%} of the account" if r["n_trades"] else "—",
                     "Random traders beaten": f"{S.beat_share(run, name):.0f} in 100"})
    return pd.DataFrame(rows).set_index("Who")


# ---- 2. was it luck ----------------------------------------------------------------
def luck_chart(S: Summary, run: str, pyramiding: bool = False, agents: list[str] | None = None):
    """1,000 coin-flip traders' results as a histogram, with each prompt and the passive
    index drawn as a vertical line where it lands."""
    r = S.random_returns(run, pyramiding=pyramiding)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.hist(r, bins=45, color="#d9d8d3", edgecolor="white", lw=0.6)
    ymax = ax.get_ylim()[1]
    names = agents or PROMPTS + [INDEX]
    for i, name in enumerate(names):
        v = S.value(run, name, "total_return")
        ax.axvline(v, color=COLOR[name], lw=2.2 if name in PROMPTS else 1.6, ls="-" if name in PROMPTS else "--")
        beaten = S.beat_share(run, name, pyramiding=pyramiding)
        ax.text(v, ymax * (0.96 - 0.11 * i), f"  {name.split(' — ')[0]} {_pct(v)}: beat {beaten:.0f} in 100", color=COLOR[name], fontsize=9.5, va="top",
                fontweight="bold" if name in PROMPTS else "normal", bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0%}"))
    who = "500 coin-flip traders" + (" who were also allowed to keep adding to winners" if pyramiding else "")
    ax.set_title(f"{RUN_LABEL[run]} — where {who} ended up", loc="left")
    ax.set_xlabel("result over the window"); _tidy(ax, "number of random traders")
    fig.tight_layout()
    return fig


# ---- 3. the one bet ---------------------------------------------------------------
def one_bet_chart(S: Summary, run: str = "leaderboard_rules", agent: str = PROMPTS[0]):
    """Every trade of one player as a bar, in the order they were closed. The single
    trade that carried the result stands out on its own."""
    t = S.trades[(S.trades["run"] == run) & (S.trades["agent"] == agent)].sort_values("exit_date").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 4.4))
    colors = ["#2a78d6" if p > 0 else "#b3b2ae" for p in t["pnl"]]
    ax.bar(range(len(t)), t["pnl"], color=colors, width=0.7)
    ax.axhline(0, color=INK2, lw=0.8)
    best = t["pnl"].idxmax()
    share = S.value(run, agent, "biggest_bet_share")
    ax.text(best - 0.5, t.loc[best, "pnl"], f"{t.loc[best, 'symbol']}: +${t.loc[best, 'pnl']:,.0f}  \n{min(share, 1.0):.0%} of the account in one stock  ",
            va="top", ha="right", fontsize=9.5, color=INK)
    ax.set_xticks(range(len(t))); ax.set_xticklabels(t["symbol"], rotation=60, fontsize=8.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    losers = int((t["pnl"] <= 0).sum())
    ax.set_title(f"{agent.split(' — ')[0]} under the leaderboard rules — profit or loss on each of its {len(t)} trades ({losers} lost)", loc="left")
    _tidy(ax, "profit / loss per trade")
    fig.tight_layout()
    return fig


# ---- 4. before / after the rule --------------------------------------------------
def before_after_chart(S: Summary):
    """Each prompt's result under the leaderboard rules and with the 25% limit, side by
    side, with the passive index as the bar to clear."""
    runs = ["leaderboard_rules", "position_limit"]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = np.arange(len(PROMPTS)); w = 0.36
    for j, run in enumerate(runs):
        vals = [S.value(run, p, "total_return") for p in PROMPTS]
        bars = ax.bar(x + (j - 0.5) * w, vals, width=w - 0.04, color=[COLOR[p] for p in PROMPTS], alpha=1.0 if j else 0.45)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + (0.004 if v >= 0 else -0.004), _pct(v), ha="center", va="bottom" if v >= 0 else "top", fontsize=9, color=INK)
    idx = S.value("position_limit", INDEX, "total_return")
    ax.axhline(idx, color=COLOR[INDEX], ls="--", lw=1.4); ax.text(len(PROMPTS) - 0.5, idx, f" hold all 50: {_pct(idx)}", va="bottom", ha="right", color=COLOR[INDEX], fontsize=9.5)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([p.split(' — ')[0] for p in PROMPTS])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0%}"))
    ax.set_title("Same model, same prompts, same days — before and after the 25% position limit", loc="left")
    ax.bar([np.nan], [np.nan], color="#9a9a9a", alpha=0.45, label="leaderboard rules (no limit)"); ax.bar([np.nan], [np.nan], color="#9a9a9a", label="with a 25% position limit")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _tidy(ax, "result over the window")
    fig.tight_layout()
    return fig


# ---- 5. confidence ----------------------------------------------------------------
def confidence_chart(S: Summary):
    """Stated confidence next to the share of trades that actually made money, for every
    prompt in both runs."""
    a = S.agents[S.agents["kind"] == "prompt"].copy()
    a["label"] = [f"{p.split(' — ')[0]}, {'leaderboard rules' if r == 'leaderboard_rules' else '25% limit'}  (n={int(n)})" for p, r, n in zip(a["agent"], a["run"], a["n_trades"])]
    a = a.sort_values(["run", "agent"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    y = np.arange(len(a))
    for i, r in a.iterrows():
        ax.plot([r["win_rate"], r["stated_confidence"]], [i, i], color=GRID, lw=2, zorder=1)
        ax.scatter(r["stated_confidence"], i, color=INK2, s=70, zorder=2)
        ax.scatter(r["win_rate"], i, color=COLOR[r["agent"]], s=90, zorder=3)
        ax.text(r["win_rate"] - 0.02, i, f"{r['win_rate']:.0%}", ha="right", va="center", fontsize=9.5, color=INK)
    ax.text(a["stated_confidence"].iloc[0] + 0.02, 0, "said: 85% sure", color=INK2, fontsize=9.5, va="center")
    ax.set_yticks(y); ax.set_yticklabels(a["label"], fontsize=9.5)
    ax.set_xlim(-0.02, 1.0); ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("share of trades that made money (coloured)  vs  confidence the model stated on those trades (grey)")
    ax.set_title("The model said it was 85% sure on every single trade. Here is how often it was right.", loc="left")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def confidence_facts(S: Summary) -> dict:
    t = S.trades[S.trades["agent"].isin(PROMPTS)]
    return {"n_trades": int(len(t)), "min_conf": float(t["confidence"].min()), "max_conf": float(t["confidence"].max()),
            "share_won": float((t["pnl"] > 0).mean())}
