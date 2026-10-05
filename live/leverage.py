"""Builds docs/leverage.json: the odds behind "what's at stake on the next snap".

    .venv\\Scripts\\python -m live.leverage

The page plays out the next snap many ways (docs/js/leverage.js):

  turnover (chance by down and distance)  → opponent's ball at the spot
  otherwise, yards gained = each of 20 typical gains for this down and distance
      → touchdown, first down, or the next down

and averages how far win probability moves across them: the expected swing. On a 4th down it
uses the 4th-down helper's outcomes for the model's pick instead.

"Stakes" = that expected swing divided by the average for a 1st-3rd down snap, so 1.0x is a
typical play and 3x is a play that matters three times as much. This script measures the
gain/turnover odds from recent seasons and computes that average with the same recipe
(it must stay in step with leverage.js).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import nflreadpy as nfl

from live.model import WPModel
from live.train import load

SEASONS = list(range(2018, 2026))
BASELINE_SEASONS = range(2021, 2026)
OUT = Path(__file__).resolve().parent.parent / "docs" / "leverage.json"
YTG_BUCKETS = [1, 2, 3, 5, 7, 10, 15, 99]  # upper edges: 1, 2, 3, 4-5, 6-7, 8-10, 11-15, 16+
N_GAINS = 20
KICKOFF_YARDLINE = 70
EXPECTED_PAT = 0.95


def bucket(ytg):
    return np.searchsorted(YTG_BUCKETS, ytg)


def odds(seasons):
    """Gain quantiles and turnover chance for runs and passes (sacks included), by down x distance bucket."""
    cols = ["season", "game_id", "play_id", "down", "ydstogo", "play_type", "yards_gained",
            "interception", "fumble_lost", "game_seconds_remaining", "half_seconds_remaining", "qtr"]
    df = nfl.load_pbp(seasons).select(cols).to_pandas()
    snaps = df[df.down.notna()].sort_values(["game_id", "play_id"])
    plays = snaps[snaps.play_type.isin(["run", "pass"]) & snaps.down.between(1, 3)].copy()
    plays["to"] = ((plays.interception == 1) | (plays.fumble_lost == 1)).astype(int)
    plays["b"] = bucket(plays.ydstogo.values)
    qs = (np.arange(N_GAINS) + 0.5) / N_GAINS
    gains, turnover = {}, {}
    for (down, b), g in plays.groupby(["down", "b"]):
        clean = g[g.to == 0].yards_gained
        gains[f"{int(down)}-{b}"] = [round(float(v), 1) for v in np.quantile(clean, qs)]
        turnover[f"{int(down)}-{b}"] = round(float(g.to.mean()), 4)

    # seconds that come off the clock per snap, normally and in the last 2 minutes of a half
    nxt = snaps.groupby(["game_id", "qtr"]).game_seconds_remaining.shift(-1)
    used = (snaps.game_seconds_remaining - nxt).dropna()
    late = snaps.loc[used.index].half_seconds_remaining <= 120
    secs = {"normal": float(used[~late & (used > 0)].median()), "late": float(used[late & (used > 0)].median())}
    print(f"Clock per snap: {secs}")
    print("1st & 10: gains", gains["1-5"][::4], "turnover", turnover["1-5"])
    return gains, turnover, secs


def outcomes(s, gains, turnover, secs):
    """[(chance, next state, offense keeps the ball)] for one 1st-3rd down state. Mirrors leverage.js."""
    t = secs["late"] if s["half_seconds_remaining"] <= 120 else secs["normal"]
    clock = {"game_seconds_remaining": max(1, s["game_seconds_remaining"] - t),
             "half_seconds_remaining": max(1, s["half_seconds_remaining"] - t)}

    def flip(yardline, points=0):
        return {**s, **clock, "down": 1, "ydstogo": min(10, yardline), "yardline_100": yardline,
                "score_differential": -(s["score_differential"] + points),
                "posteam_timeouts_remaining": s["defteam_timeouts_remaining"],
                "defteam_timeouts_remaining": s["posteam_timeouts_remaining"],
                "posteam_spread": -s["posteam_spread"], "home": 1 - s["home"],
                "receive_2h_ko": 1 - s["receive_2h_ko"] if s["game_seconds_remaining"] > 1800 else 0}

    key = f"{int(s['down'])}-{bucket(s['ydstogo'])}"
    p_to = turnover[key]
    out = [(p_to, flip(int(min(99, max(1, 100 - s["yardline_100"])))), False)]
    ytez, ytg = s["yardline_100"], s["ydstogo"]
    for g in gains[key]:
        w = (1 - p_to) / N_GAINS
        spot = ytez - int(np.floor(g + 0.5))  # round half up, like Math.round in leverage.js
        if spot <= 0:
            out.append((w, flip(KICKOFF_YARDLINE, 6 + EXPECTED_PAT), False))
            continue
        spot = min(99, spot)
        if ytez - spot >= ytg:
            out.append((w, {**s, **clock, "down": 1, "ydstogo": min(10, spot), "yardline_100": spot}, True))
        elif s["down"] < 3:
            out.append((w, {**s, **clock, "down": s["down"] + 1, "ydstogo": ytg - (ytez - spot), "yardline_100": spot}, True))
        else:  # 4th down next: in the page that's the 4th-down helper's job; here just the 4th-down state
            out.append((w, {**s, **clock, "down": 4, "ydstogo": ytg - (ytez - spot), "yardline_100": spot}, True))
    return out


def baseline(gains, turnover, secs):
    """Average expected swing of a 1st-3rd down snap in recent seasons, plus a check against what happened."""
    model = WPModel()
    df = load()
    df = df[df.season.isin(BASELINE_SEASONS)].sort_values(["game_id", "play_id"]).reset_index(drop=True)
    p = model.predict(df.to_dict("records"))
    df["wp_home"] = np.where(df.home == 1, p, 1 - p)
    df["real_swing"] = (df.groupby("game_id").wp_home.shift(-1) - df.wp_home).abs()
    sample = df[df.down.between(1, 3) & (df.is_ot == 0) & df.real_swing.notna()].sample(20000, random_state=0)

    fields = ["score_differential", "game_seconds_remaining", "half_seconds_remaining", "down", "ydstogo",
              "yardline_100", "posteam_timeouts_remaining", "defteam_timeouts_remaining", "posteam_spread",
              "home", "receive_2h_ko", "is_ot"]
    states = sample[fields].to_dict("records")
    rows, owner = [], []
    for i, s in enumerate(states):
        for w, st, keeps in outcomes(s, gains, turnover, secs):
            rows.append(st)
            owner.append((i, w, keeps))
    after = model.predict(rows)
    before = model.predict(states)
    expected = np.zeros(len(states))
    for (i, w, keeps), a in zip(owner, after):
        expected[i] += w * abs((a if keeps else 1 - a) - before[i])
    avg = float(expected.mean())
    print(f"Average expected swing per 1st-3rd down snap: {avg:.4f} (actual average: {sample.real_swing.mean():.4f})")
    print(f"Correlation of expected vs actual swing: {np.corrcoef(expected, sample.real_swing)[0, 1]:.2f}")
    sample = sample.assign(lev=expected / avg)
    for lo, hi in ((0, 0.5), (0.5, 1.5), (1.5, 2.5), (2.5, 99)):
        m = sample.lev.between(lo, hi, inclusive="left")
        print(f"  stakes {lo}-{hi}x: {m.mean():.0%} of snaps, actual avg swing {sample.real_swing[m].mean():.4f}")
    return avg


def main():
    gains, turnover, secs = odds(SEASONS)
    avg = baseline(gains, turnover, secs)
    OUT.write_text(json.dumps({
        "seasons": [SEASONS[0], SEASONS[-1]],
        "ytg_buckets": YTG_BUCKETS,  # upper edges; key = "<down>-<bucket index>"
        "gains": gains, "turnover": turnover, "secs": secs,
        "avg_swing": round(avg, 5),
    }))
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
