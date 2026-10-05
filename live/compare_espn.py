"""Compares our win probability with ESPN's on every game of a finished season.

    .venv\\Scripts\\python -m live.compare_espn            # 2025
    .venv\\Scripts\\python -m live.compare_espn 2026       # after the 2026 season (add it to SEASONS in live/train.py first)

To be a fair test the model must not have seen the season, but the model the site uses is
trained on every season. So this trains a copy the same way (same features, same number of
trees) on the earlier seasons only, puts it in a copy of the site, and replays every game of
the season through the page's own code in a headless browser. That way both models see exactly
the same snaps, read from ESPN's feed the same way the dashboard does.

ESPN's number for a play is its win probability *after* the play; the page already shifts it
onto the next snap, so the two are compared snap for snap.

Prints a Brier score table (lower is better). Paste it into VS_ESPN in docs/index.html
(the "About the model" section) and the table in README.md.
"""
import json
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / ".compare"  # holdout model + site copy (not committed)
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"


def train_holdout(season: int) -> Path:
    """The live model's recipe, trained on every season before `season`, inside a copy of docs/."""
    from live import export_site_model
    from live.model import FEATURES, MODEL_PATH, new_classifier
    from live.train import load

    n_trees = json.loads(MODEL_PATH.with_name("wp_live_meta.json").read_text())["n_trees"]
    df = load()
    train = df[df.season < season]
    print(f"Training a {n_trees}-tree model on {train.season.min()}–{season - 1} ({len(train):,} plays)…")
    model = new_classifier(n_estimators=n_trees, early_stopping_rounds=None)
    model.fit(train[FEATURES], train.label, verbose=False)

    site = WORK / "docs"
    if site.exists():
        shutil.rmtree(site)
    shutil.copytree(ROOT / "docs", site)
    model.save_model(WORK / "wp_holdout.json")
    export_site_model.export(WORK / "wp_holdout.json", site / "model.json")
    return site


def replay_season(site: Path, season: int) -> list[dict]:
    """Every finished game of the season, loaded by the page's own espn.js: our WP and ESPN's per snap."""
    from playwright.sync_api import sync_playwright

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "-d", str(site)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.5)

    def relay(route):
        # a page on localhost can't reach ESPN directly, so Python fetches it and hands it over
        req = urllib.request.Request(route.request.url, headers={"User-Agent": UA})
        for _ in range(3):
            try:
                body = urllib.request.urlopen(req, timeout=30).read()
                break
            except Exception:
                time.sleep(2)
        else:
            return route.abort()
        route.fulfill(status=200, body=body,
                      headers={"content-type": "application/json", "access-control-allow-origin": "*"})

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(user_agent=UA)
            page.route(re.compile(r"https://[a-z.]*espn\.com/.*"), relay)
            page.goto(f"http://127.0.0.1:{port}/")
            page.set_default_timeout(0)
            print(f"Replaying every {season} game through the page (a few minutes)…")
            res = page.evaluate("""async (year) => {
              const { loadGames, loadGame } = await import('./js/espn.js');
              const weeks = [];
              for (let w = 1; w <= 18; w++) weeks.push([w, 2]);
              for (const w of [1, 2, 3, 5]) weeks.push([w, 3]);  // playoffs (week 4 is the Pro Bowl)
              const ids = [];
              for (const [week, seasontype] of weeks) {
                const d = await loadGames({ week, seasontype, year }).catch(() => null);
                for (const g of d?.games || []) if (g.state === 'post') ids.push(g.id);
              }
              const out = [], errors = [];
              let k = 0;
              async function worker() {
                while (k < ids.length) {
                  const id = ids[k++];
                  try {
                    const g = await loadGame(id);
                    out.push({ id, home_won: g.wp_home, points: g.points.filter(x => x.kind !== 'final')
                      .map(x => [x.minute, x.wp_home, x.espn_wp_home ?? null]) });
                  } catch (e) { errors.push([id, String(e)]); }
                }
              }
              await Promise.all([1, 2, 3].map(worker));
              return { out, errors };
            }""", season)
            browser.close()
    finally:
        server.terminate()
    if res["errors"]:
        print(f"{len(res['errors'])} games failed to load, e.g. {res['errors'][:3]}")
    return res["out"]


def score(games: list[dict]):
    rows = [(i, minute, ours, min(max(espn, 0), 1), g["home_won"])
            for i, g in enumerate(games) if g["home_won"] != 0.5  # skip ties
            for minute, ours, espn in g["points"] if ours is not None and espn is not None]
    gid, minute, ours, espn, won = np.array(rows).T
    brier = lambda p, m: float(np.mean((p[m] - won[m]) ** 2))
    phases = [("Whole game", minute >= 0), ("1st quarter", minute < 15), ("2nd quarter", (minute >= 15) & (minute < 30)),
              ("3rd quarter", (minute >= 30) & (minute < 45)), ("4th quarter", (minute >= 45) & (minute < 60)),
              ("Last 5 minutes", (minute >= 55) & (minute <= 60)), ("Overtime", minute > 60)]
    n_games = len(set(gid))
    print(f"\n{n_games} games with a winner, {len(rows):,} snaps. Brier score, lower is better:\n")
    print(f"  {'':16}{'Ours':>8}{'ESPN':>8}")
    table = []
    for name, m in phases:
        if m.any():
            table.append([name, round(brier(ours, m), 3), round(brier(espn, m), 3)])
            print(f"  {name:16}{table[-1][1]:>8.3f}{table[-1][2]:>8.3f}   ({m.sum():,} snaps)")
    closer = sum(brier(ours, gid == g) < brier(espn, gid == g) for g in set(gid))
    print(f"\nOurs was closer in {closer} of {n_games} games.")
    print("\nFor VS_ESPN in docs/index.html:\n" + json.dumps(table))


def main():
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
    WORK.mkdir(exist_ok=True)
    site = train_holdout(season)
    games = replay_season(site, season)
    (WORK / f"games_{season}.json").write_text(json.dumps(games))
    score(games)


if __name__ == "__main__":
    main()
