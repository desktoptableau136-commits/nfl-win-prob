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

## Retraining the model

```powershell
.venv\Scripts\python -m live.train
```

Trains on 2006–2025, saves `models/wp_live.json`, and re-exports `docs/model.json` for the page.
Re-run after each season (update `SEASONS` in `live/train.py`), then push to update the hosted page.

## Files

| Path | What it is |
|---|---|
| `01_learn_win_probability.ipynb` | Learning notebook: history, Stern formula, logistic regression, XGBoost, game replays, what-if tool |
| `live/model.py` | Model features + training setup (shared by training and export) |
| `live/train.py` | Trains the live model (includes overtime) and exports it for the page |
| `live/export_site_model.py` | Converts the XGBoost model into compact `docs/model.json` |
| `live/server.py` | Tiny local web server for `docs/` (home Wi-Fi use) |
| `docs/index.html` | The dashboard page (layout, chart, lists) |
| `docs/js/espn.js` | Reads ESPN's feed and rebuilds the situation before every play |
| `docs/js/model.js` | Runs the model's 1,500 decision trees in the browser |

## Known limitations

- ESPN's feed is unofficial and could change without notice.
- Overtime estimates are rougher (only 33 games have been played under the current OT rules).
- Right after a score, the dashboard assumes the kickoff ends up at the receiving team's 30 until the next snap is posted.
- Before an overtime coin toss, it assumes the home team receives.
- The pregame line is the sportsbook's closing line as ESPN reports it (DraftKings). Games without a line are treated as even.
