"""Builds docs/twopoint.json: the odds behind the kick-or-go-for-2 helper.

    .venv\\Scripts\\python -m live.two_point

The tree model is good at the overall win chance but too lumpy to compare a lead of +7 with
+8: that difference is one point of margin, and the trees' steps across score margins are
about as big as the edge being measured. So the decision is made the way 2-point charts are
built: play out the rest of the game possession by possession.

  - each drive ends in a touchdown, a field goal, or nothing (league rates)
  - after a touchdown, the scoring team picks the better try (kick or go for 2), all the way down
  - a field goal that doesn't help (e.g. down 7 on the last drive) is swapped for a try at a touchdown
  - the number of drives left depends on the clock (measured from real games)
  - a tie at the end is a coin flip

The page (docs/js/twopoint.js) runs the same calculation; keep the two in step.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import nflreadpy as nfl

SEASONS = list(range(2018, 2026))
OUT = Path(__file__).resolve().parent.parent / "docs" / "twopoint.json"
MAX_DRIVES = 30
FG_TO_TD = 0.35  # a drive that would have kicked a FG, going for the TD instead, scores one this often
COLS = ["season", "game_id", "play_id", "qtr", "game_seconds_remaining", "fixed_drive", "fixed_drive_result",
        "touchdown", "extra_point_attempt", "extra_point_result", "two_point_attempt", "two_point_conv_result"]


def measure():
    df = nfl.load_pbp(SEASONS).select(COLS).to_pandas()
    reg = df[(df.qtr <= 4) & df.fixed_drive.notna()]

    drives = reg.groupby(["game_id", "fixed_drive"]).fixed_drive_result.first()
    p_td, p_fg = float((drives == "Touchdown").mean()), float((drives == "Field goal").mean())

    xp, two = df[df.extra_point_attempt == 1], df[df.two_point_attempt == 1]
    pat = float((xp.extra_point_result == "good").mean())
    two_pt = float((two.two_point_conv_result == "success").mean())

    # drives left in regulation after a touchdown (starting with the other team's), by minutes left
    last = reg.groupby("game_id").fixed_drive.max()
    td = reg[reg.touchdown == 1].drop_duplicates(["game_id", "fixed_drive"])
    left = (last.reindex(td.game_id).values - td.fixed_drive.values).clip(0, MAX_DRIVES).astype(int)
    minute = np.ceil(td.game_seconds_remaining.values / 60).clip(1, 60).astype(int)
    by_minute = []
    for m in range(1, 61):
        w = 1 if m <= 10 else 2 if m <= 30 else 3  # pool neighbouring minutes where TDs are spread thin
        near = left[np.abs(minute - m) <= w]
        by_minute.append([round(float(v), 4) for v in np.bincount(near, minlength=MAX_DRIVES + 1)[:MAX_DRIVES + 1] / len(near)])
    print(f"Drive ends in TD {p_td:.3f}, FG {p_fg:.3f}; extra point {pat:.3f}, 2-point try {two_pt:.3f}")
    return {"seasons": [SEASONS[0], SEASONS[-1]], "p_td": round(p_td, 4), "p_fg": round(p_fg, 4), "fg_to_td": FG_TO_TD,
            "pat": round(pat, 3), "two_pt": round(two_pt, 3), "drives_left": by_minute}  # drives_left[minute - 1][n]


def solve(t, max_n=MAX_DRIVES, span=80):
    """V[n][ball][lead + span]: chance team A wins with `lead` and n drives left, ball = 0 (A) or 1 (B) next.
    Mirrors solve() in docs/js/twopoint.js."""
    leads = np.arange(-span, span + 1)
    V = np.zeros((max_n + 1, 2, len(leads)))
    end = np.where(leads > 0, 1.0, np.where(leads < 0, 0.0, 0.5))
    V[0, 0] = V[0, 1] = end
    at = lambda arr, d: arr[np.clip(d + span, 0, 2 * span)]
    pk, p2, ptd, pfg, swap = t["pat"], t["two_pt"], t["p_td"], t["p_fg"], t["fg_to_td"]
    for n in range(1, max_n + 1):
        for ball, sign, best in ((0, 1, np.maximum), (1, -1, np.minimum)):
            nxt = V[n - 1, 1 - ball]
            base = at(nxt, leads + 6 * sign)
            kick = pk * at(nxt, leads + 6 * sign + sign) + (1 - pk) * base
            go2 = p2 * at(nxt, leads + 6 * sign + 2 * sign) + (1 - p2) * base
            td = best(kick, go2)
            none = at(nxt, leads)
            fg = best(at(nxt, leads + 3 * sign), swap * td + (1 - swap) * none)
            V[n, ball] = ptd * td + pfg * fg + (1 - ptd - pfg) * none
    return V, span


def choice(t, V, span, lead, minute):
    """Scoring team's chance after kicking / going for 2, `lead` = its lead after the TD, before the try."""
    dist = np.array(t["drives_left"][int(min(60, max(1, minute))) - 1])
    after = lambda d: float(dist @ V[:, 1, d + span])  # the other team has the ball next
    kick = t["pat"] * after(lead + 1) + (1 - t["pat"]) * after(lead)
    go2 = t["two_pt"] * after(lead + 2) + (1 - t["two_pt"]) * after(lead)
    return kick, go2


def chart(t):
    V, span = solve(t)
    print("\nGo for 2 (2) or kick (k), by the scoring team's lead after the TD; margin in WP points if over 1")
    for minute in (55, 40, 25, 15, 10, 6, 3, 1):
        cells = []
        for lead in range(-16, 17):
            kick, go2 = choice(t, V, span, lead, minute)
            d = (go2 - kick) * 100
            cells.append(f"{lead:+d}:{'2' if d > 0 else 'k'}{abs(d):.0f}" if abs(d) >= 1 else f"{lead:+d}:·")
        print(f"{minute:>2} min: " + " ".join(cells))


def main():
    t = measure()
    chart(t)
    OUT.write_text(json.dumps(t))
    print(f"Saved {OUT}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--chart"]:
        chart(json.loads(OUT.read_text()))
    else:
        main()
