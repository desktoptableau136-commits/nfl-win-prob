"""SiriusXM channels for this week's NFL games → docs/sirius.json.

    .venv\\Scripts\\python -m live.sirius

SiriusXM carries both teams' radio calls of every game but reassigns the channel numbers each week,
and its site can't be read from the browser (no CORS). So this scrapes siriusxm.com/sports/nfl and
writes a small table the page reads. A GitHub Actions workflow (.github/workflows/sirius.yml) runs
it daily and commits the file when it changes. Standard library only, so the workflow needs no installs.

The page embeds its schedule as JSON (see parse()), which is read instead of the rendered HTML.
"""
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PAGE = "https://www.siriusxm.com/sports/nfl"
TEAMS = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
OUT = Path(__file__).resolve().parent.parent / "docs" / "sirius.json"


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (nfl-win-prob)"})
    return urllib.request.urlopen(req, timeout=30).read().decode("utf-8")


def team_abbrs():
    """ESPN's full team names → abbreviations, the key the page uses ("Philadelphia Eagles" → "PHI")."""
    teams = json.loads(get(TEAMS))["sports"][0]["leagues"][0]["teams"]
    return {t["team"]["displayName"]: t["team"]["abbreviation"] for t in teams}


def flight_data(page):
    """The page is a Next.js app; its data arrives as JS strings pushed onto self.__next_f. Join and unescape them."""
    chunks = re.findall(r'self\.__next_f\.push\(\[1,"((?:[^"\\]|\\.)*)"\]\)', page)
    return "".join(json.loads('"' + c + '"') for c in chunks)


def parse(page, abbr):
    """Each game is an object {"key":"game-…","start":ISO time,"variant":"matchup","awayTeam":…,"homeTeam":…}, where
    a team's listenOptions list its own feed ("away"/"home": "channel" is satellite, "internetChannel" the app)
    plus any "national" and "spanish" feeds. The same game can appear more than once on the page."""
    data, dec = flight_data(page), json.JSONDecoder()
    games, seen = [], set()
    for m in re.finditer(r'\{"key":"game-', data):
        try:
            g, _ = dec.raw_decode(data, m.start())
        except ValueError:
            continue
        if g.get("variant") != "matchup" or g["key"] in seen:
            continue
        seen.add(g["key"])
        away, home = g["awayTeam"]["name"].strip(), g["homeTeam"]["name"].strip()
        if away not in abbr or home not in abbr:
            print(f"skipping unknown team: {away} @ {home}")
            continue
        feeds = {}
        for side in ("awayTeam", "homeTeam"):
            for o in g[side].get("listenOptions", []):
                if str(o.get("channel", "")).isdigit():
                    feeds.setdefault(o["key"], int(o["channel"]))
        games.append({
            "date": g["start"][:10],  # local (Eastern) date of kickoff
            "away": abbr[away], "home": abbr[home],
            "away_ch": feeds.get("away"),
            "home_ch": feeds.get("home"),
            "national": feeds.get("national"),
            "spanish": feeds.get("spanish"),
        })
    return games


def main():
    games = parse(get(PAGE), team_abbrs())
    if not games:
        raise SystemExit("No games found on the SiriusXM page; its layout may have changed. Leaving sirius.json alone.")
    old = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    if old.get("games") == games:
        print(f"{len(games)} games, unchanged")
        return
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    OUT.write_text(json.dumps({"updated": stamp, "games": games}, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(games)} games → {OUT.name}")
    for g in games:
        print(f"  {g['date']} {g['away']:>3} {g['away_ch']} @ {g['home']:<3} {g['home_ch']}"
              f"{'  national ' + str(g['national']) if g['national'] else ''}{'  español ' + str(g['spanish']) if g['spanish'] else ''}")


if __name__ == "__main__":
    main()
