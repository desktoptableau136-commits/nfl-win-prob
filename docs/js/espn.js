// Reads ESPN's public (unofficial) NFL JSON feeds in the browser and rebuilds the
// game situation before every play, in the same shape the model was trained on.
// Port of live/espn.py (the Python reference version); keep the two in step.
//
// ESPN conventions (checked against real games):
// - a play's awayScore/homeScore are the score AFTER the play
// - a play's start/end hold down, distance, yardsToEndzone and the team with the ball
// - odds `spread` is the HOME team's line: -3.5 means home favored by 3.5

import { predict } from "./model.js";

const SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard";
const SUMMARY_URL = id => `https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event=${id}`;

const NOT_SNAPS = new Set(["Kickoff", "Timeout", "Official Timeout", "End Period", "End of Half", "End of Game",
  "End of Regulation", "Two-minute warning", "Coin Toss", "Extra Point Good", "Extra Point Missed",
  "Two Point Pass", "Two Point Rush", "Defensive 2pt Conversion"]);
const KICKOFF_YARDLINE = 70;  // typical start after a kickoff under the 2025 rules (own 30)
const EXPECTED_PAT = 0.95;    // points a TD is "really" worth before the extra point is posted

// ---------- fetching ----------
const cache = new Map();  // url -> {time, promise}
async function fetchJSON(url, maxAgeMs = 10000) {
  const hit = cache.get(url);
  if (hit && Date.now() - hit.time < maxAgeMs) return hit.promise;
  const promise = fetch(url, { cache: "no-store" }).then(r => {
    if (!r.ok) throw new Error(`ESPN ${r.status}`);
    return r.json();
  });
  cache.set(url, { time: Date.now(), promise });
  promise.catch(() => cache.delete(url));
  return promise;
}

// ---------- pregame line ----------
/** Points the HOME team is favored by (negative = underdog). Uses the sportsbook's
 *  closing line, which stays fixed once the game starts and is what the model was
 *  trained on; before the line closes, the current line. */
function homeSpread(odds) {
  for (const o of odds || []) {
    const close = Number.parseFloat(o.pointSpread?.home?.close?.line);
    if (Number.isFinite(close)) return -close;
    if (o.spread !== null && o.spread !== undefined) return -Number(o.spread);
  }
  return null;
}

// ---------- small helpers ----------
function clockSeconds(display) {
  const [m, s] = String(display || "0:00").split(":");
  return Number(m) * 60 + Math.floor(Number(s || 0));
}
/** [game_seconds_remaining, half_seconds_remaining]; in OT both are the OT clock. */
function gameClock(period, secs) {
  if (period >= 5) return [secs, secs];
  const game = (4 - period) * 900 + secs;
  return [game, period <= 2 ? game - 1800 : game];
}
function teamInfo(c) {
  const t = c.team;
  return {
    id: String(t.id), abbr: t.abbreviation, name: t.shortDisplayName || t.displayName,
    color: "#" + (t.color || "666666"), alt_color: "#" + (t.alternateColor || "999999"),
    logo: t.logo || (t.logos || [{}])[0].href, score: Number(c.score || 0),
  };
}
const valid = pos => !!(pos && pos.team && pos.down >= 1 && pos.down <= 4
  && pos.yardsToEndzone >= 1 && pos.yardsToEndzone <= 99 && (pos.distance || 0) >= 1);

// ESPN's yardsToEndzone is wrong on punts (it repeats the yard line, so a punt from your own 6
// looks like 4th down at the opponent's 6). The down-and-distance text ("4th & 17 at NYJ 6") is right.
function toEndzone(pos, abbrOf) {
  const m = (pos.downDistanceText || "").match(/ at (?:([A-Z]+) )?(\d+)$/);
  if (!m) return pos.yardsToEndzone;
  const n = Number(m[2]);
  return !m[1] ? 50 : m[1] === abbrOf[pos.team.id] ? 100 - n : n;
}

/** The opening kickoff from each side; averaged since we don't know the coin toss yet. */
function pregameStates(spreadHome) {
  const fav = spreadHome || 0;
  const base = { score_differential: 0, game_seconds_remaining: 3600, half_seconds_remaining: 1800,
    down: 1, ydstogo: 10, yardline_100: KICKOFF_YARDLINE, posteam_timeouts_remaining: 3,
    defteam_timeouts_remaining: 3, receive_2h_ko: 0, is_ot: 0 };
  return [{ ...base, home: 1, posteam_spread: fav }, { ...base, home: 0, posteam_spread: -fav }];
}

// ---------- parsing ----------
/** Turns an ESPN summary into teams, status, and a list of pre-snap situations.
 *  Each situation has the model's raw fields (offense's view) plus display info. */
function parseGame(summary, liveSituation = null) {
  const comp = summary.header.competitions[0];
  const gameId = String(summary.header.id);
  const teams = Object.fromEntries(comp.competitors.map(c => [c.homeAway, teamInfo(c)]));
  const homeId = teams.home.id, awayId = teams.away.id;
  const abbrToId = { [teams.home.abbr]: homeId, [teams.away.abbr]: awayId };
  const abbrOf = { [homeId]: teams.home.abbr, [awayId]: teams.away.abbr };
  const status = comp.status;
  const state = status.type.state;  // pre / in / post
  const period = status.period || 0;
  const clockNow = clockSeconds(status.displayClock ?? "15:00");
  const spreadHome = homeSpread(summary.pickcenter || summary.odds);
  const isPostseason = summary.header.season?.type === 3;
  const otLen = isPostseason ? 900 : 600;

  const byId = new Map();
  for (const d of summary.drives?.previous || []) for (const p of d.plays || []) byId.set(p.id, p);
  for (const p of summary.drives?.current?.plays || []) byId.set(p.id, p);
  const plays = [...byId.values()].sort((a, b) => Number(a.sequenceNumber || 0) - Number(b.sequenceNumber || 0));

  const espnWp = Object.fromEntries((summary.winprobability || []).map(w => [w.playId, w.homeWinPercentage]));
  let timeouts = { [homeId]: 3, [awayId]: 3 };
  const openingReceiver = plays.find(p => valid(p.start))?.start.team.id ?? null;
  let score = { [homeId]: 0, [awayId]: 0 };
  let lastPeriod = 1;
  let pendingKickoff = null;  // [scoring team, points, wasSafety] until the next kickoff or snap
  let gained = { [homeId]: 0, [awayId]: 0 };
  const situations = [];

  function situation(posteam, down, dist, ytez, per, secs, extraPoints = {}, display = {}) {
    const defteam = posteam === homeId ? awayId : homeId;
    const [gsr, hsr] = gameClock(per, secs);
    const fav = spreadHome || 0;
    return {
      score_differential: score[posteam] + (extraPoints[posteam] || 0) - score[defteam] - (extraPoints[defteam] || 0),
      game_seconds_remaining: gsr, half_seconds_remaining: hsr,
      down, ydstogo: dist, yardline_100: ytez,
      posteam_timeouts_remaining: timeouts[posteam], defteam_timeouts_remaining: timeouts[defteam],
      posteam_spread: posteam === homeId ? fav : -fav,
      home: posteam === homeId ? 1 : 0,
      receive_2h_ko: per <= 2 && openingReceiver !== null && posteam !== openingReceiver ? 1 : 0,
      is_ot: per >= 5 ? 1 : 0,
      posteam, period: per, clock: `${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, "0")}`,
      minute: per <= 4 ? (3600 - gsr) / 60 : 60 + (per - 5) * otLen / 60 + (otLen - secs) / 60,
      home_score: score[homeId], away_score: score[awayId], ...display,
    };
  }

  for (const p of plays) {
    const per = p.period?.number ?? lastPeriod;
    if (per !== lastPeriod) {
      if (per === 3 || per >= 5) {  // timeouts reset for the 2nd half (3) and OT (2)
        const n = per === 3 ? 3 : 2;
        timeouts = { [homeId]: n, [awayId]: n };
      }
      lastPeriod = per;
    }
    const kind = p.type?.text || "";
    const text = p.text || "";
    if (kind === "Timeout") {
      const m = text.match(/Timeout #\d by (\w+)/);
      if (m && m[1] in abbrToId) timeouts[abbrToId[m[1]]] = Math.max(0, timeouts[abbrToId[m[1]]] - 1);
    }
    const start = p.start || {};
    if (!NOT_SNAPS.has(kind) && valid(start)) {
      situations.push(situation(start.team.id, start.down, start.distance, toEndzone(start, abbrOf),
        per, clockSeconds(p.clock?.displayValue), {}, {
          play_id: p.id, text, kind, espn_wp_home: espnWp[p.id], down_text: start.downDistanceText || "",
          scoring: !!p.scoringPlay, turnover: !!p.isTurnover,
        }));
    }
    const prev = score;
    score = { [homeId]: Number(p.homeScore ?? prev[homeId]), [awayId]: Number(p.awayScore ?? prev[awayId]) };
    gained = { [homeId]: score[homeId] - prev[homeId], [awayId]: score[awayId] - prev[awayId] };
    const scorer = [homeId, awayId].find(t => gained[t] > 0);
    if (scorer) pendingKickoff = [scorer, gained[scorer], (kind + text).toLowerCase().includes("safety")];
    else if (valid(p.end) && kind !== "Timeout" && kind !== "Official Timeout") pendingKickoff = null;
  }

  // --- the situation right now ---
  teams.home.score = score[homeId];
  teams.away.score = score[awayId];
  let now = null;
  if (state === "in" && plays.length) {
    const last = plays[plays.length - 1];
    const end = last.end || {};
    if (liveSituation) {  // scoreboard has the official timeout counts
      timeouts = { [homeId]: liveSituation.homeTimeouts ?? timeouts[homeId],
                   [awayId]: liveSituation.awayTimeouts ?? timeouts[awayId] };
    }
    const kind = last.type?.text || "";
    const lastEnd = [...plays].reverse().map(p => p.end).find(valid) || null;
    const endOfRegulation = kind === "End of Regulation" || (kind === "End Period" && last.period?.number === 4);

    if (kind === "End of Half") {
      timeouts = { [homeId]: 3, [awayId]: 3 };
      const receiver = [homeId, awayId].find(t => t !== openingReceiver) ?? homeId;
      now = situation(receiver, 1, 10, KICKOFF_YARDLINE, 3, 900, {}, { down_text: "2nd-half kickoff" });
    } else if (endOfRegulation && score[homeId] === score[awayId]) {
      // overtime coin toss hasn't happened yet; assume the home team receives
      timeouts = { [homeId]: 2, [awayId]: 2 };
      now = situation(homeId, 1, 10, KICKOFF_YARDLINE, 5, otLen, {}, { down_text: "Overtime kickoff" });
    } else if (pendingKickoff && !valid(end)) {
      // a team just scored and the kickoff hasn't happened yet
      const [scorer, points, safety] = pendingKickoff;
      const other = scorer === homeId ? awayId : homeId;
      const receiver = safety ? scorer : other;  // after a safety the scoring team receives
      const pendingPat = points === 6 ? { [scorer]: EXPECTED_PAT } : {};
      now = situation(receiver, 1, 10, KICKOFF_YARDLINE, period, clockNow, pendingPat, { down_text: "Kickoff" });
    } else if (valid(end) || lastEnd) {
      const e = valid(end) ? end : lastEnd;
      now = situation(e.team.id, e.down, e.distance, toEndzone(e, abbrOf), period, clockNow, {},
        { down_text: e.downDistanceText || "" });
    } else {  // nothing usable yet, e.g. right before the opening kickoff
      now = situation(homeId, 1, 10, KICKOFF_YARDLINE, Math.max(period, 1), clockNow, {}, { down_text: "Kickoff" });
    }
    now.text = "Now";
    now.kind = "now";
  }

  return { id: gameId, state, detail: status.type.shortDetail || "", period, home: teams.home, away: teams.away,
    neutral: !!comp.neutralSite, spread_home: spreadHome, postseason: isPostseason, situations, now };
}

// ---------- adding win probability ----------
async function homeWp(states, neutral) {
  const p = await predict(states, neutral);
  return states.map((s, i) => s.home === 1 ? p[i] : 1 - p[i]);
}
async function pregameWp(spreadHome, neutral) {
  const [receive, kick] = await homeWp(pregameStates(spreadHome), neutral);
  return (receive + kick) / 2;
}
function finalWp(g) {
  const h = g.home.score, a = g.away.score;
  return h > a ? 1 : h < a ? 0 : 0.5;
}

/** Everything the game page needs: teams, status, WP now, and a WP point per snap. */
export async function loadGame(gameId, liveSituation = null) {
  const game = parseGame(await fetchJSON(SUMMARY_URL(gameId)), liveSituation);
  const points = [...game.situations, ...(game.now ? [game.now] : [])];
  (await homeWp(points, game.neutral)).forEach((wp, i) => { points[i].wp_home = wp; });

  if (game.state === "pre") {
    game.wp_home = await pregameWp(game.spread_home, game.neutral);
  } else if (game.state === "post") {
    game.wp_home = finalWp(game);
    points.push({ minute: Math.max(60, ...points.map(p => p.minute)), wp_home: game.wp_home, text: "Final",
      kind: "final", home_score: game.home.score, away_score: game.away.score });
  } else {
    game.wp_home = points.length ? points[points.length - 1].wp_home : await pregameWp(game.spread_home, game.neutral);
  }
  // how much each snap moved the needle (home view): WP before the next snap minus WP before this one
  for (let i = 0; i + 1 < points.length; i++) points[i].swing = points[i + 1].wp_home - points[i].wp_home;
  game.points = points;
  delete game.situations;
  return game;
}

/** The week's games for the list page, with WP for each. */
export async function loadGames({ week, seasontype, year } = {}) {
  const params = new URLSearchParams(Object.entries({ week, seasontype, dates: year }).filter(([, v]) => v));
  const sb = await fetchJSON(`${SCOREBOARD_URL}${params.size ? "?" + params : ""}`, 15000);
  const games = await Promise.all((sb.events || []).map(async event => {
    const comp = event.competitions[0];
    const teams = Object.fromEntries(comp.competitors.map(c => [c.homeAway, teamInfo(c)]));
    const state = event.status.type.state;
    const neutral = !!comp.neutralSite;
    const g = { id: event.id, state, detail: event.status.type.shortDetail || "", date: event.date,
      home: teams.home, away: teams.away, neutral, possession: comp.situation?.possession,
      down_text: comp.situation?.shortDownDistanceText, spread_home: homeSpread(comp.odds) };
    if (state === "pre") g.wp_home = await pregameWp(g.spread_home, neutral);
    else if (state === "post") g.wp_home = finalWp(g);
    else g.wp_home = await loadGame(event.id, comp.situation).then(x => x.wp_home, () => null);
    return g;
  }));
  const order = { in: 0, pre: 1, post: 2 };
  games.sort((a, b) => (order[a.state] ?? 3) - (order[b.state] ?? 3) || String(a.date).localeCompare(String(b.date)));
  return { week: sb.week?.number, seasontype: sb.season?.type, year: sb.season?.year, games };
}
