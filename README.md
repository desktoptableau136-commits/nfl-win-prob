# NFL Win Probability Calculator

Learning project: build an NFL win probability model from scratch, then run it live during games.

## Setup (already done once)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## 1. Learning notebook

```powershell
.venv\Scripts\jupyter lab 01_learn_win_probability.ipynb
```

Run cells top to bottom (Shift+Enter). The first data download takes about 30–60 seconds.

## 2. Live dashboard

The dashboard is a static web page in `docs/`. **Everything runs in your browser.** The page
fetches ESPN's live game feed directly and runs the model (exported to `docs/model.json`)
on your phone. No server does any work, which is what lets GitHub Pages host it for free.

### Run it locally

```powershell
.venv\Scripts\python -m live.server
```

- On this PC: http://localhost:8000
- On your phone (same Wi-Fi): the address the server prints, e.g. `http://192.168.68.61:8000`
  (needs the "Win Prob dashboard (port 8000)" firewall rule on Private networks).

### Host it on GitHub Pages

Push this folder to a GitHub repo, then **Settings → Pages → Deploy from a branch → `main` / `/docs`**.
After about a minute, the dashboard is live at `https://<your-username>.github.io/<repo-name>/`.
To update it, push again.

Pages refresh themselves: every 15 seconds during a live game.
Use the ‹ › arrows to browse other weeks, and open any finished game to replay its chart.
On a live 4th down, the game page shows a **4th-down helper**: your team's chance to win
if it goes for it, kicks a field goal, or punts. Past 4th downs show the model's pick next to what the team did.
Tap any play to find it on the chart. On a phone, use **Add to Home Screen** to open it full screen like an app.

### How it compares with ESPN

Every 2025 game (284 with a winner, about 40,000 snaps) was replayed with a copy of the model
trained only on 2006–2024 (same recipe, 1,519 trees), so it had never seen these games.
Scores are Brier scores (lower is better) on the same snaps. ESPN's number for a play is posted
after the play, so each snap is compared with ESPN's latest number before it:

| | Ours | ESPN |
|---|---|---|
| Whole game | **0.159** | 0.165 |
| 1st quarter | **0.206** | 0.216 |
| 2nd quarter | **0.181** | 0.188 |
| 3rd quarter | **0.149** | 0.153 |
| 4th quarter | **0.109** | 0.110 |
| Last 5 minutes | 0.089 | 0.089 |
| Overtime | **0.174** | 0.177 |

Ours was closer in 188 of the 284 games; its edge is mostly early in games. The same table
is in the dashboard's "About the model" section at the bottom of the games list.

## Retraining the model

```powershell
.venv\Scripts\python -m live.train
```

Trains on 2006–2025, saves `models/wp_live.json`, and re-exports `docs/model.json` for the page.
Re-run after each season (update `SEASONS` in `live/train.py`), then push to update the hosted page.
Then refresh the two lookup files the page uses:

```powershell
.venv\Scripts\python -m live.excitement   # docs/excitement.json: "Excitement" comparison
.venv\Scripts\python -m live.fourth       # docs/fourth.json: 4th-down conversion / field goal / punt odds
```

## Files

| Path | What it is |
|---|---|
| `01_learn_win_probability.ipynb` | Learning notebook: history, Stern formula, logistic regression, XGBoost, game replays, what-if tool |
| `live/model.py` | Model features + training setup (shared by training and export) |
| `live/train.py` | Trains the live model (includes overtime) and exports it for the page |
| `live/export_site_model.py` | Converts the XGBoost model into compact `docs/model.json` |
| `live/excitement.py` | Builds `docs/excitement.json`: how much WP moved in 2021–25 games, for the game summary |
| `live/fourth.py` | Builds `docs/fourth.json`: conversion, field goal and punt odds for the 4th-down helper |
| `live/server.py` | Tiny local web server for `docs/` (home Wi-Fi use) |
| `docs/index.html` | The dashboard page (layout, chart, lists) |
| `docs/js/espn.js` | Reads ESPN's feed and rebuilds the situation before every play |
| `docs/js/fourth.js` | 4th-down helper: WP after going for it, kicking, or punting |
| `docs/js/model.js` | Runs the model's 1,500 decision trees in the browser |
| `docs/manifest.webmanifest`, `docs/icons/` | Lets phones install the page as a home-screen app |

## Known limitations

- ESPN's feed is unofficial and could change without notice.
- Overtime estimates are rougher (only 33 games have been played under the current OT rules).
- Right after a score, the dashboard assumes the kickoff ends up at the receiving team's 30 until the next snap is posted.
- Before an overtime coin toss, it assumes the home team receives.
- The pregame line is the sportsbook's closing line as ESPN reports it. ESPN removes lines from a game's feed a few weeks
  after it's played, so for older games the page reads them from ESPN's odds archive. Games without any line are treated as even.
- The 4th-down helper assumes an average offense, kicker and punter, and uses the typical outcome of each choice
  (e.g. the usual gain on a conversion and the usual punt distance), not every possible outcome.
  Treat differences under about 1–2 points of win probability as toss-ups.
