"""SiriusXM channels for this week's NFL games → docs/sirius.json.

    .venv\\Scripts\\python -m live.sirius

SiriusXM carries both teams' radio calls of every game but reassigns the channel numbers each week,
and its site can't be read from the browser (no CORS). So this scrapes siriusxm.com/sports/nfl and
writes a small table the page reads. A GitHub Actions workflow (.github/workflows/sirius.yml) runs
it daily and commits the file when it changes. Standard library only, so the workflow needs no installs.

Each game on the page carries aria-labels like
    "Listen to Philadelphia Eagles on 263"                                   (satellite channel, the radio number)
    "Listen to Philadelphia Eagles @ Jacksonville Jaguars on home channel 814"     (app channel; identifies the game)
    >National< … >CH 88<                                                       (extra feeds, as visible text)
under a heading like "Sunday, October 11 · 9:30 AM EDT".
"""
import json
import re
import urllib.request
from datetime import date, datetime, timezone
from html import unescape
from pathlib import Path

PAGE = "https://www.siriusxm.com/sports/nfl"
TEAMS = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
OUT = Path(__file__).resolve().parent.parent / "docs" / "sirius.json"
MONTHS = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August",
                                      "September", "October", "November", "December"], 1)}


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (nfl-win-prob)"})
    return urllib.request.urlopen(req, timeout=30).read().decode("utf-8")


def team_abbrs():
    """ESPN's full team names → abbreviations, the key the page uses ("Philadelphia Eagles" → "PHI")."""
    teams = json.loads(get(TEAMS))["sports"][0]["leagues"][0]["teams"]
    return {t["team"]["displayName"]: t["team"]["abbreviation"] for t in teams}


def game_date(heading, today):
    """'Sunday, October 11 · 9:30 AM EDT' → the date, picking the year nearest today (the page spans New Year's)."""
    m = re.search(r"(\w+) (\d+)", heading.split(",", 1)[1])
    d = date(today.year, MONTHS[m.group(1)], int(m.group(2)))
    if (d - today).days > 180:
        d = d.replace(year=d.year - 1)
    elif (today - d).days > 180:
        d = d.replace(year=d.year + 1)
    return d


def parse(page, abbr, today):
    text = unescape(page.replace("<!-- -->", ""))
    games, heading = [], None
    # each date heading is followed by its games, each game a <li aria-label="Home vs Away">
    for kind, body in re.findall(r'(dateTitle[^>]*>|<li aria-label=")(.*?)(?=dateTitle[^>]*>|<li aria-label="|$)', text, re.S):
        if kind.startswith("dateTitle"):
            heading = body.split("<", 1)[0]
            continue
        feeds = re.findall(r'Listen to ([^"@]+?) @ ([^"]+?) on (\w+) channel (\d+)', body)
        if not feeds or heading is None:
            continue
        away, home = feeds[0][0].strip(), feeds[0][1].strip()
        if away not in abbr or home not in abbr:
            print(f"skipping unknown team: {away} @ {home}")
            continue
        sat = {t.strip(): ch for t, ch in re.findall(r'Listen to ([^"@]+?) on (\d+)"', body)}  # team → satellite channel
        # national and Spanish feeds: the aria-label may give the app channel, so read the visible "CH n" after the label
        extra = lambda label: (m := re.search(rf">{label}<.*?>CH (\d+)<", body, re.S)) and int(m.group(1))
        games.append({
            "date": game_date(heading, today).isoformat(),
            "away": abbr[away], "home": abbr[home],
            "away_ch": int(sat[away]) if away in sat else None,
            "home_ch": int(sat[home]) if home in sat else None,
            "national": extra("National"),
            "spanish": extra("Espa[nñ]ol"),
        })
    return games


def main():
    games = parse(get(PAGE), team_abbrs(), date.today())
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
