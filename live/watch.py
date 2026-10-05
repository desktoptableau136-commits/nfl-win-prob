"""Builds docs/watch.json: how the games list picks "Watch now".

    .venv\\Scripts\\python -m live.watch

With several games on, the best one to switch to is the one whose win probability is about to
move the most. This script measures, for every snap in recent seasons, how far win probability
travelled over the next 5 minutes of game clock (to the final result if the game ended first),
and what predicted it:

  - how much the game is in doubt (1 at 50/50, 0 when it's decided) and how much time is left:
    a table of the typical 5-minute movement for each combination
  - the next snap's stakes (docs/js/leverage.js), a small extra nudge
  - how exciting the game has been so far turned out not to help, so it isn't used

The page (docs/js/watch.js) looks the table up and ranks live games by the result.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from live import leverage
from live.model import WPModel
from live.train import load

SEASONS = range(2021, 2026)
OUT = Path(__file__).resolve().parent.parent / "docs" / "watch.json"
WINDOW = 300        # seconds of game clock ahead
DOUBT_BINS = 10
TIME_BIN = 300      # regulation time-left bins of 5 minutes (12 of them), plus one column for overtime
FLOOR = 0.01        # log(movement + FLOOR), so near-zero stretches don't dominate


def future_movement(df):
    """WP movement over the next WINDOW seconds of game clock, for every snap (home view, final result at the end)."""
    df["t"] = np.where(df.is_ot == 1, 3600 + 600 - df.game_seconds_remaining, 3600 - df.game_seconds_remaining)
    fut = np.zeros(len(df))
    for _, g in df.groupby("game_id", sort=False):
        idx = g.index.values
        final = g.final_home.values[-1]
        w = np.append(g.wp_home.values, final)
        t = np.append(g.t.values, g.t.values[-1] + 1)
        cum = np.concatenate([[0], np.cumsum(np.abs(np.diff(w)))])
        end = np.maximum(np.searchsorted(t, t[:-1] + WINDOW, side="right") - 1, np.arange(len(idx)) + 1)
        fut[idx] = cum[end] - cum[:-1]
    return fut


def cells(d):
    doubt = np.minimum((d.doubt * DOUBT_BINS).astype(int), DOUBT_BINS - 1)
    col = np.where(d.is_ot == 1, 3600 // TIME_BIN, np.minimum(d.game_seconds_remaining // TIME_BIN, 3600 // TIME_BIN - 1))
    return doubt.values, col.astype(int)


def stakes(sample, model, t):
    """The page's next-snap stakes for 1st-3rd down states (same recipe as leverage.js)."""
    fields = ["score_differential", "game_seconds_remaining", "half_seconds_remaining", "down", "ydstogo",
              "yardline_100", "posteam_timeouts_remaining", "defteam_timeouts_remaining", "posteam_spread",
              "home", "receive_2h_ko", "is_ot"]
    states = sample[fields].to_dict("records")
    rows, owner = [], []
    for i, s in enumerate(states):
        for w, st, keeps in leverage.outcomes(s, t["gains"], t["turnover"], t["secs"]):
            rows.append(st)
            owner.append((i, w, keeps))
    after, before = model.predict(rows), sample.wp_pos.values
    swing = np.zeros(len(states))
    for (i, w, keeps), a in zip(owner, after):
        swing[i] += w * abs((a if keeps else 1 - a) - before[i])
    return swing / t["avg_swing"]


def main():
    model = WPModel()
    df = load()
    df = df[df.season.isin(SEASONS)].sort_values(["game_id", "play_id"]).reset_index(drop=True)
    df["wp_pos"] = model.predict(df.to_dict("records"))
    df["wp_home"] = np.where(df.home == 1, df.wp_pos, 1 - df.wp_pos)
    df["final_home"] = np.where(df.home == 1, df.label, 1 - df.label)
    df["fut"] = future_movement(df)
    df["doubt"] = 1 - np.abs(2 * df.wp_home - 1)
    df["y"] = np.log(df.fut + FLOOR)

    fit, test = df[df.season < SEASONS[-1]], df[df.season == SEASONS[-1]]

    def table(d):
        doubt, col = cells(d)
        return d.y.groupby([doubt, col]).mean()

    # the stakes nudge: fitted on what the table leaves unexplained, on a sample of 1st-3rd downs
    t = json.loads(leverage.OUT.read_text())
    sample = fit[fit.down.between(1, 3) & (fit.is_ot == 0)].sample(20000, random_state=0).copy()
    sample["stakes"] = stakes(sample, model, t)
    tab = table(fit)
    left = sample.y.values - tab.reindex(list(zip(*cells(sample)))).values
    x = np.log(sample.stakes.values + 0.05)
    coef = float(np.polyfit(x, left, 1)[0])

    # check on the held-out season
    check = test[test.down.between(1, 3) & (test.is_ot == 0)].sample(6000, random_state=0).copy()
    check["stakes"] = stakes(check, model, t)
    base = tab.reindex(list(zip(*cells(check)))).fillna(fit.y.mean()).values
    score = base + coef * np.log(check.stakes.values + 0.05)
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, len(check), 50000), rng.integers(0, len(check), 50000)
    pairs = lambda sc: ((sc[i] > sc[j]) == (check.fut.values[i] > check.fut.values[j])).mean()
    print(f"{SEASONS[-1]} check: picking the snap with more movement in the next 5 minutes, out of two random snaps")
    print(f"  next-snap stakes alone   {pairs(check.stakes.values):.1%}  (rank correlation {spearmanr(check.stakes, check.fut).correlation:.2f})")
    print(f"  doubt x time table       {pairs(base):.1%}  (rank correlation {spearmanr(base, check.fut).correlation:.2f})")
    print(f"  table + stakes           {pairs(score):.1%}  (rank correlation {spearmanr(score, check.fut).correlation:.2f})")

    full = table(df)  # final table from all seasons
    n_cols = 3600 // TIME_BIN + 1
    grid = [[round(float(full.get((d, c), np.nan)), 3) for c in range(n_cols)] for d in range(DOUBT_BINS)]
    grid = pd.DataFrame(grid).T.ffill().bfill().T.values.tolist()  # fill any empty cell from its neighbours
    print("Typical 5-minute movement (points of WP), by doubt (rows, decided → 50/50) and minutes left (columns, 60 → 5, then OT):")
    for d in (0, 3, 6, 9):
        print(f"  doubt {d / 10:.1f}+ ", " ".join(f"{100 * (np.exp(v) - FLOOR):4.0f}" for v in grid[d][11::-1] + grid[d][12:]))
    OUT.write_text(json.dumps({
        "seasons": [SEASONS[0], SEASONS[-1]], "window": WINDOW, "floor": FLOOR,
        "doubt_bins": DOUBT_BINS, "time_bin": TIME_BIN,  # column = min(11, game seconds left // 300); 12 = overtime
        "log_move": grid, "stakes_coef": round(coef, 4),
        "avg_move": round(float(df.fut.mean()), 4),
    }))
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
