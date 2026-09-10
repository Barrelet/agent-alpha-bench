"""Generates notebooks/04_the_story.ipynb — the experiment explained for readers who are
not data scientists. Reads data/summary/ (see scripts/export_summary.py); never calls a
model, runs in seconds.

    python scripts/export_summary.py     # once, after the experiments have run
    python scripts/make_notebook_04.py
"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Can an AI trade stocks? I built a test to find out.

*A plain-language walk through one experiment. No maths beyond percentages, no code you need to read — each short code cell just draws the next picture. If you want the full technical version, it is in `03_new_rule_model_comparison.ipynb`; the code is in the `alphabench/` folder.*

**The question.** Websites like Alpha Arena and TradeRank.ai let AI language models trade pretend money and rank them by how much they made. A model at the top of such a list looks clever. But is it? A rising market lifts everyone, a lucky bet lifts anyone, and the leaderboard never asks. I wanted a way to tell skill from luck.

**The short answer, so you can stop here if you like.** Over one summer, with one small AI model and three different sets of instructions, nothing beat the dullest strategy there is: buying all fifty stocks and doing nothing. The one result that looked impressive turned out to be a single lucky bet. And the model said it was 85% sure of every trade it made, while only one trade in seven made money.

The rest of this notebook shows how I know that.""")

code("""import sys
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
from alphabench.report import *          # the pictures; everything is read from data/summary/
S = load_summary(ROOT)
print("Runs in this story:", *[f"{v['label']} — {S.window(k)}" for k, v in S.meta["runs"].items()], sep="\\n  ")""")

md("""## 1. The game

I built a small simulator that plays a stock-trading game, and let different players take turns at it. The rules are the same as on the leaderboards, plus one I added later (section 5):

- **$10,000 of pretend money**, 50 large US companies (Apple, Microsoft, and so on), 60 trading days from June to August 2026.
- **One decision a day**, after the market closes, carried out at the next morning's opening price — so nobody can peek at prices they could not have known.
- **One new position per day, at most ten at a time.** A player may bet on a stock going up (*long*) or down (*short*).
- **Every bet needs a stop-loss**: a price at which the simulator closes it automatically if it goes the wrong way.
- **Every trade costs 0.1%**, like a real broker.
- A player may **add to a bet that is already winning**, but never to one that is losing.

Every player sees the same information: the last few weeks of prices and a few simple statistics for each stock, plus their own account. Nothing else. No news, no earnings, no internet.""")

md("""## 2. The players

**Three AI players.** The same AI model each time (Qwen3, an 8-billion-parameter open model that runs on my laptop), with three different sets of instructions, or *prompts*:

- **Prompt 1** — the instructions copied from the leaderboards: manage the account, explain each bet, set a stop-loss.
- **Prompt 2** — the same, plus some risk rules: how big a bet should be, how far away the stop-loss belongs.
- **Prompt 3** — a full step-by-step procedure: check your existing positions first, score each stock on four criteria, only bet when it clearly beats doing nothing.

**Two players that are not AI at all.** They are there so we can tell whether the AI is doing anything a simpler thing could not:

- **A one-line rule**: buy whatever went up most in the last ten days, sell it when it stops going up.
- **Do nothing**: buy all 50 stocks in equal amounts on day one and hold. Also the S&P 500 index fund, for reference.

**And 1,000 coin-flippers.** These are the key to the whole experiment. Each one plays by the same rules, but decides everything at random: whether to trade today, which stock, up or down, how much. They have no information and no skill. If an AI player finishes ahead of most of them, that means something; if it finishes in the middle of the pack, it has shown nothing that a coin would not.""")

md("""## 3. What happened — under the leaderboard's rules

First I ran the game exactly as the leaderboards do. Here is the value of each player's account over the summer, starting at 100.""")

code("""equity_chart(S, "leaderboard_rules");""")

md("""Prompt 1, the leaderboard's own instructions, finished up 11%. That beats the do-nothing strategy (+5.4%), the S&P 500 (+1.3%) and the simple rule (−4.7%). On a leaderboard, Prompt 1 would be the winner.

The scoreboard adds three columns the leaderboards do not show: how many trades each player made, how many of them made money, and how big its biggest single bet was.""")

code("""scoreboard(S, "leaderboard_rules")""")

md("""Two things stand out. Prompt 1 won only 2 of its 17 trades. And its biggest bet was the *whole account* in one stock. That is worth a closer look.""")

md("""## 4. Where the +11% came from

Here is every trade Prompt 1 made, in order, with its profit or loss.""")

code("""one_bet_chart(S);""")

md("""Fifteen of seventeen trades lost money. One made $5. The seventeenth made $1,786 — and that one trade is the entire result. What happened: the model bought Microsoft, and then, because the rules allow adding to a winning bet, it kept adding, day after day, until almost the whole account was in that one stock. Microsoft then rose 19% in a month. Take that one trade away and Prompt 1 lost money.

Prompt 2 did the same thing with Amazon. Prompt 3, told to size every bet at 10–20% of the account, never piled in, and made nothing.

**Was piling in a clever decision?** This is what the coin-flippers are for. The chart below shows where 500 of them ended up when they, too, were allowed to keep adding to winners. Each bar is a number of random traders who finished with that result; the blue line is Prompt 1.""")

code("""luck_chart(S, "leaderboard_rules", pyramiding=True, agents=[PROMPTS[0], INDEX]);""")

md("""About one random trader in ten did as well as Prompt 1 or better, with no information at all. That is the kind of result a coin produces one time in ten — and Prompt 1 was the best of three prompts I tried, which is roughly how often you would expect to find one. "Beat 90 in 100" sounds strong; against a coin that can pile into a winner, it is one lucky summer.

There is a second, simpler problem: no real fund would ever let a manager put nearly all of the money into one stock. The leaderboard's rules allow it, so their winners can be whoever bet biggest on the stock that happened to go up.""")

md("""## 5. Adding the rule any real account has

So I added one rule to the game: **no single stock may be more than 25% of the account.** The simulator enforces it on every player — the three AI prompts, the simple rule, and all the coin-flippers — and every prompt is told about it. Then I ran the whole summer again.""")

code("""equity_chart(S, "position_limit");""")

code("""scoreboard(S, "position_limit")""")

md("""With the limit in place, the +11% disappears. Prompt 1 tried to add to its positions 37 times; the rule refused 24 of those and cut the rest down to size. It finished at −4.4%, with one winning trade in fourteen. Prompt 2 ended flat, Prompt 3 slightly down. Doing nothing still made +5.4%.

Here are the three prompts before and after the rule, side by side.""")

code("""before_after_chart(S);""")

md("""And here is where they land among 500 coin-flippers playing under the same rule.""")

code("""luck_chart(S, "position_limit");""")

md("""All three prompts sit in the middle of the pack, somewhere between beating 30 and 65 random traders in 100. That is the range a coin-flipper lands in most of the time. The do-nothing strategy beat 91 in 100 — without making a single decision.""")

md("""## 6. Did the model know when it was right?

Every time an AI player made a bet, it also had to say how confident it was, as a number between 0 and 1. This is the part of the experiment I care about most, because a model that *knows* when it is unsure would be useful even if it were not a great trader: you could trust it more when it says 0.95 and less when it says 0.80.

Here is what the model said, next to what actually happened, for every prompt in both runs.""")

code("""confidence_chart(S);""")

md("""The model said it was 85% sure on every trade — all 50 of them across both runs, whatever stock, whatever the situation. In reality between 7% and 25% of its trades made money. Its confidence was not a judgement; it was a number it had learned to write down.

Of the three findings in this notebook, this is the one I would bet survives a longer test. The returns might change with a different summer. A confidence figure that never moves cannot mean anything in any summer.""")

md("""## 7. What this does and does not show

**What it shows.** Over one summer, with one small model, nothing the AI did beat holding the fifty stocks and waiting. The result that looked good was one bet that the rules should not have allowed. The model's confidence carried no information. And a leaderboard that only reports returns would have shown you none of this — it would have shown Prompt 1 at +11% and moved on.

**What it does not show.** That AI cannot trade. This is one 8-billion-parameter model that fits on a laptop; the models on the leaderboards are a hundred times larger. It is sixty days in a calm, rising market. The AI players made between 4 and 17 trades each, which is far too few to measure skill even if it were there. And the information they had — prices and a few statistics — is thin.

**What I would do next.** Run a full year, so each player makes enough trades to measure. Try a larger model. And keep the three tests — a do-nothing benchmark, a thousand coin-flippers, and a check on stated confidence — because they are what turned a leaderboard number into an answer.""")

md("""---

### Glossary

**Long / short.** A long bet makes money if the stock goes up; a short bet makes money if it goes down (you borrow the shares, sell them, and buy them back later, hopefully cheaper).

**Stop-loss.** A price at which a bet is closed automatically to cap the loss.

**Position limit.** A cap on how much of the account may be in one stock. The rule added in section 5.

**Coin-flipper / random trader.** A simulated player that follows all the rules but chooses at random. A thousand of them show the spread of results that pure luck produces, so a real player's result can be placed against it: "beat 90 in 100" means only one random trader in ten did better.

**Do-nothing strategy (equal-weight index).** Buy all 50 stocks in equal amounts on day one and never trade. The bar every active player has to clear.

**Prompt.** The written instructions given to the AI model before it sees the data.

**Stated confidence.** The probability the model attaches to each bet when it makes it. A well-calibrated model that says 0.85 should be right about 85% of the time.

### Where the numbers come from

Everything in this notebook is read from `data/summary/`, a small set of tables written by `scripts/export_summary.py` from the two full runs (`03_model_comparison.ipynb` without the position limit, `03_new_rule_model_comparison.ipynb` with it). Those notebooks contain the complete tables, the code that produced them, and the caveats in full.""")

nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parents[1] / "notebooks" / "04_the_story.ipynb"
nbf.write(nb, out)
print("wrote", out)
