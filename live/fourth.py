"""Builds docs/fourth.json: the odds behind the 4th-down helper.

    .venv\\Scripts\\python -m live.fourth

For each 4th down the page compares three choices by the win probability they lead to:

  Go for it   P(convert) x WP(1st down after the typical gain)  +  P(fail) x WP(opponent ball at the spot)
  Field goal  P(make) x WP(up 3, kick off)                    +  P(miss) x WP(opponent ball at the kick spot)
  Punt        WP(opponent ball where punts from here usually end up)

It also saves extra point and 2-point success rates, for a kick-or-go-for-2 helper (not on the page yet).

This script measures the ingredients from recent seasons:
  - conversion chance by yards to go (3rd and 4th downs, fitted together; 4th downs are rarer)
  - typical yards gained on a successful conversion
  - field goal make chance by kick distance
  - where the receiving team usually starts after a punt from each spot
  - extra point and 2-point conversion success rates
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import nflreadpy as nfl
from sklearn.linear_model import LogisticRegression

SEASONS = list(range(2018, 2026))
OUT = Path(__file__).resolve().parent.parent / "docs" / "fourth.json"
COLS = ["season", "game_id", "play_id", "posteam", "defteam", "down", "ydstogo", "yardline_100", "goal_to_go",
        "play_type", "yards_gained", "first_down", "touchdown", "fumble_lost", "interception",
        "field_goal_attempt", "field_goal_result", "kick_distance", "punt_attempt", "punt_blocked",
        "half_seconds_remaining", "qtr", "extra_point_attempt", "extra_point_result",
        "two_point_attempt", "two_point_conv_result"]


def conversion(df):
    """P(convert) by yards to go, separately for goal-to-go (the end zone squeezes the field)."""
    tries = df[df.down.isin([3, 4]) & df.play_type.isin(["run", "pass"]) & (df.ydstogo <= 20)
               & (df.half_seconds_remaining > 120)].copy()  # skip hurry-up/garbage situations
    tries["ok"] = ((tries.first_down == 1) | (tries.touchdown == 1)).astype(int)
    x = lambda d: np.column_stack([np.log(d.ydstogo), d.ydstogo, d.goal_to_go, (d.down == 4).astype(int),
                                   d.goal_to_go * np.log(d.ydstogo)])
    lr = LogisticRegression(C=10, max_iter=1000).fit(x(tries), tries.ok)
    ytg = np.arange(1, 21)
    table = lambda goal: lr.predict_proba(x(pd.DataFrame({"ydstogo": ytg, "goal_to_go": goal, "down": 4})))[:, 1]
    conv, conv_goal = table(0), table(1)
    fourth = tries[tries.down == 4]
    print("4th-down conversion  yards: model / actual (tries)")
    for y in (1, 2, 3, 5, 8, 10):
        a = fourth[(fourth.ydstogo == y) & (fourth.goal_to_go == 0)]
        print(f"  4th & {y:<2}  {conv[y - 1]:.2f} / {a.ok.mean():.2f} ({len(a)})")

    # typical gain on a successful (non goal-to-go) conversion: a bit more than needed
    ok = tries[(tries.ok == 1) & (tries.goal_to_go == 0)]
    gain = ok.groupby(ok.ydstogo.clip(upper=15)).yards_gained.median().reindex(range(1, 21), method="ffill")
    return [round(float(v), 3) for v in conv], [round(float(v), 3) for v in conv_goal], [float(v) for v in gain]


def field_goals(df):
    fg = df[(df.field_goal_attempt == 1) & df.kick_distance.notna()].copy()
    fg["made"] = (fg.field_goal_result == "made").astype(int)
    # Up to 58 yards there are plenty of kicks: use the actual make rate, smoothed over ±3 yards.
    # Beyond that, kicks are rare and only the strongest legs try them, so continue with a
    # straight line in log-odds, with the slope of a fit to all kicks (the page doesn't offer FGs past 66).
    dist = np.arange(18, 76)
    rate = fg.groupby("kick_distance").made.agg(["sum", "count"]).reindex(dist, fill_value=0)
    win = lambda c: rate[c].rolling(7, center=True, min_periods=1).sum()
    smooth = np.minimum.accumulate((win("sum") / win("count")).fillna(1).values)
    slope = LogisticRegression(C=10, max_iter=1000).fit(fg[["kick_distance"]].values, fg.made).coef_[0, 0]
    at58 = smooth[dist == 58][0]
    tail = 1 / (1 + np.exp(-(np.log(at58 / (1 - at58)) + slope * (dist - 58))))
    p = np.where(dist <= 58, smooth, tail)
    print("FG make chance  distance: model / actual (tries)")
    for lo in (20, 30, 40, 50, 55, 60):
        a = fg[fg.kick_distance.between(lo, lo + 4)]
        print(f"  {lo}-{lo + 4}  {p[lo + 2 - 18]:.2f} / {a.made.mean():.2f} ({len(a)})")
    return [round(float(v), 3) for v in p]


def punts(df):
    """Where the receiving team starts (their yardline_100) after a punt from each spot."""
    df = df[df.posteam.notna()].sort_values(["game_id", "play_id"])
    nxt = df.groupby("game_id")[["posteam", "yardline_100", "down"]].shift(-1)
    p = df[(df.punt_attempt == 1) & (df.play_type == "punt") & (nxt.posteam == df.defteam) & (nxt.down == 1)].copy()
    p["opp_yl"] = nxt.loc[p.index, "yardline_100"]
    by = p.groupby("yardline_100").opp_yl.median().reindex(range(1, 100)).interpolate(limit_direction="both")
    smooth = by.rolling(5, center=True, min_periods=1).mean()
    print("Punt  from yards-to-end-zone …: receiver's yards to go the other way")
    for y in (90, 80, 70, 60, 50, 40):
        print(f"  {y} → {smooth[y]:.0f}")
    return [round(float(v), 1) for v in smooth]


def conversions(df):
    """Extra point and 2-point conversion success rates (for the kick-or-go-for-2 helper)."""
    xp = df[df.extra_point_attempt == 1]
    two = df[df.two_point_attempt == 1]
    pat = float((xp.extra_point_result == "good").mean())
    two_pt = float((two.two_point_conv_result == "success").mean())
    print(f"Extra point {pat:.3f} ({len(xp)} tries), 2-point conversion {two_pt:.3f} ({len(two)} tries)")
    return round(pat, 3), round(two_pt, 3)


def main():
    df = nfl.load_pbp(SEASONS).select(COLS).to_pandas()
    conv, conv_goal, gain = conversion(df)
    fg = field_goals(df)
    punt = punts(df)
    pat, two_pt = conversions(df)
    OUT.write_text(json.dumps({
        "seasons": [SEASONS[0], SEASONS[-1]],
        "convert": conv, "convert_goal": conv_goal,  # index = yards to go - 1 (1..20)
        "gain": gain,                                 # median yards on a conversion, index = yards to go - 1
        "fg_make": fg, "fg_min_distance": 18,         # index = kick distance - 18
        "punt_opp_yardline": punt,                    # index = yardline_100 - 1 → receiver's yardline_100
        "pat": pat, "two_pt": two_pt,                 # success rates after a touchdown
    }))
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
