# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

NFL live win-probability model and dashboard. Python (in `live/`) trains the model and builds lookup tables.
The dashboard (`docs/`) is a static GitHub Pages site. It fetches ESPN's unofficial JSON feed directly from
the browser and runs the exported XGBoost trees in JS, so no server does any work.

## Commands (Windows, PowerShell; always use the venv)

```powershell
.venv\Scripts\python -m live.server        # serve docs/ at http://localhost:8000 (and on LAN)
.venv\Scripts\python -m live.train         # train on 2006–2025 → models/wp_live.json + docs/model.json
.venv\Scripts\python -m live.fourth        # docs/fourth.json    (4th-down helper odds)
.venv\Scripts\python -m live.leverage      # docs/leverage.json  ("stakes" gain/turnover odds + typical swing)
.venv\Scripts\python -m live.two_point     # docs/twopoint.json  (kick-or-go-for-2; --chart just reprints the chart)
.venv\Scripts\python -m live.excitement    # docs/excitement.json
.venv\Scripts\python -m live.compare_espn 2025   # ~10 min season replay vs ESPN; prints VS_ESPN and CALIBRATION tables
.venv\Scripts\python -m live.stamp         # REQUIRED before pushing any change to docs/js (see below)
```

There is no test suite or linter. Changes are verified ad hoc:
- Python/JS parity checks (e.g. compare the outputs of `live/leverage.py` and `docs/js/leverage.js` on the same states).
- Playwright (chromium + webkit) against `live.server`, with ESPN requests routed through Python urllib.
  Live states are faked by replaying truncated game summaries.
- Use `PYTHONIOENCODING=utf-8` when scripts print non-ASCII (e.g. "→"). Files mix CRLF and LF line endings.

## Deploying

The site is published from `main` / `/docs` on GitHub Pages. Pages lets browsers cache files for 10 minutes.
`live.stamp` rewrites every relative `.js` import in `docs/index.html` and `docs/js/*.js` to `?v=<hash of all scripts>`.
If you skip it, phones (notably iPad Safari) mix cached old modules with new ones and the page hangs on "Loading…".
Any new import must use the `./x.js` or `./js/x.js` form so the stamper matches it.

## Architecture

**Model state.** Every situation is described from the offense's point of view with the raw fields listed in
`live/model.py`: score_differential, game/half seconds remaining, down, ydstogo, yardline_100, timeouts,
posteam_spread, home, receive_2h_ko, is_ot. `add_features` derives `spread_time` and `diff_time_ratio`.
`docs/js/model.js` (`predict(states, neutral)`) must compute the features exactly as `live/model.py` does.
The trees have monotone constraints; field-position monotonicity is what keeps close 4th-down calls stable.
Neutral sites average over both values of `home`.

**Page pipeline.**
- `espn.js` `loadGame()` parses an ESPN summary into a pre-snap state for every play, plus `game.now` for a live game.
  It handles kickoff placeholders after scores and `pat_pending` when a TD's try hasn't been posted.
- `index.html` then chains the enrichers: `loadGame(id).then(addFourthDowns).then(addLeverage).then(addConversions)`.
  Each enricher annotates plays and `game.now` in place and returns the game.
- The games list calls `addLeverage` on every live game to rank them for "Watch now".
- All UI (panels, SVG chart, the About section's `VS_ESPN` and `CALIBRATION` constants) lives in the single file `docs/index.html`.

**Python builds tables; JS mirrors the math.** Each helper has a Python builder that measures the ingredients
from nflverse play-by-play (via `nflreadpy`) and writes a JSON file to `docs/`. A JS module applies the same recipe live.
When you change one side, change the other:
- `live/fourth.py` ↔ `docs/js/fourth.js`: go/FG/punt WP from the typical outcome of each choice.
  `choices()` and `loadTables()` are exported because `leverage.js` reuses them on 4th down.
- `live/leverage.py` `outcomes()` ↔ `docs/js/leverage.js` `outcomes()`: stakes is the expected |ΔWP| over the
  next snap divided by `avg_swing` (a typical 1st–3rd down). Rounding uses `floor(x+0.5)` to match `Math.round`.
- `live/two_point.py` `solve()` ↔ `docs/js/twopoint.js` `solve()`: a drive-by-drive DP over (drives left, ball, lead).
  The tree model is too lumpy across single points of margin to make this decision, so it only sets the overall WP level.
  The DP supplies the difference between kicking and going for 2. It is regulation only.

**Model evaluation.** `live/compare_espn.py` replays a season through the real page with a holdout model trained
without that season. Its cache lives in `.compare/` (gitignored). After rerunning it, paste the printed tables into
`docs/index.html` and the README.

## Conventions

- User-facing text is plain English for a general sports fan. Avoid jargon (say "chance to win", "stakes").
- Commits use the repo-local git identity. Push only when the user says to ("push it").
