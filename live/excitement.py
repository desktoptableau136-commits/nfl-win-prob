"""Builds docs/excitement.json: how much win probability moved in recent games.

    .venv\\Scripts\\python -m live.excitement

"Excitement" = the total distance the home team's WP line travels over a game,
snap to snap, ending at the final result (the same way the page adds up swings).
The page compares a game's total against these percentiles.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from live.model import WPModel
from live.train import load

SEASONS = range(2021, 2026)
OUT = Path(__file__).resolve().parent.parent / "docs" / "excitement.json"


def main():
    df = load()
    df = df[df.season.isin(SEASONS)].sort_values(["game_id", "play_id"])
    p = WPModel().predict(df.to_dict("records"))
    df["wp_home"] = np.where(df.home == 1, p, 1 - p)
    totals = []
    for _, g in df.groupby("game_id"):
        result = g.result.iloc[0]
        end = 1.0 if result > 0 else 0.0 if result < 0 else 0.5
        wp = np.append(g.wp_home.values, end)
        totals.append(np.abs(np.diff(wp)).sum())
    totals = np.array(totals)
    pct = [round(float(v), 3) for v in np.percentile(totals, range(101))]
    OUT.write_text(json.dumps({"seasons": [min(SEASONS), max(SEASONS)], "games": len(totals), "percentiles": pct}))
    print(f"{len(totals)} games, median {np.median(totals):.2f}, 90th pct {np.percentile(totals, 90):.2f}. Saved {OUT}")


if __name__ == "__main__":
    main()
