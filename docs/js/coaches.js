// Coach report card: every 4th down and every try after a touchdown this season, graded by the
// same helpers the game page uses (fourth.js, twopoint.js), and added up by team.
// Finished games never change, so each one is graded once and kept in this browser.
import { loadGame, seasonGames } from "./espn.js?v=c92a5518";
import { addFourthDowns } from "./fourth.js?v=c92a5518";
import { addConversions } from "./twopoint.js?v=c92a5518";

const VERSION = new URL(import.meta.url).searchParams.get("v") || "dev";  // new scripts → grade again
const STORE = "coach-calls";
const CLEAR = 0.01;     // calls closer than 1 point of win probability don't count either way (same as the game page)
const WORKERS = 4;
const BOLDNESS = { punt: 0, fg: 1, go: 2, kick: 0, two: 1 };

function readStore() {
  try {
    const x = JSON.parse(localStorage.getItem(STORE));
    if (x?.v === VERSION && x.games) return x;
  } catch (e) {}
  return { v: VERSION, games: {} };
}
function writeStore(x) { try { localStorage.setItem(STORE, JSON.stringify(x)); } catch (e) {} }

/** The decisions in one finished game, each { team, kind, best, did, margin, lost, lead, period, clock, down_text }.
 *  lost = win probability the call gave up against the model's pick (0 when they agreed). */
export function decisions(game) {
  const out = [], home = game.home.id, away = game.away.id;
  const lead = (team, p, extra = 0) => (team === home ? p.home_score - p.away_score : p.away_score - p.home_score) + extra;
  const where = p => ({ period: p.period, clock: p.clock });
  for (const p of game.points) {
    const f = p.fourth;
    if (f?.did && f.margin !== null && !/kneel|spike/i.test(p.text || "")) {
      const wp = Object.fromEntries(f.options.map(o => [o.key, o.wp]));
      if (wp[f.did] !== undefined)
        out.push({ team: p.posteam, kind: "4th", best: f.best, did: f.did, margin: +f.margin.toFixed(4),
          lost: +(wp[f.best] - wp[f.did]).toFixed(4), lead: lead(p.posteam, p), down_text: p.down_text, ...where(p) });
    }
    const t = p.pat;
    if (t?.did) out.push({ team: t.team, kind: "try", best: t.best, did: t.did, margin: +t.margin.toFixed(4),
      lost: t.did === t.best ? 0 : +t.margin.toFixed(4), lead: lead(t.team, p, 6), down_text: "Try after TD", ...where(p) });
  }
  return out;
}

/** Whether a call counts (it wasn't a toss-up), and which way a disagreement leaned. */
const counts = d => (d.did === d.best ? d.margin : d.lost) >= CLEAR;
const lean = d => d.did === d.best ? null : BOLDNESS[d.did] < BOLDNESS[d.best] ? "timid" : "bold";

/** Adds up the graded games by team. */
function summarize(season, graded) {
  const teams = {};
  const row = id => (teams[id] ??= { team: season.teams[id], games: 0, calls: 0, agreed: 0, lost: 0,
    timid: { n: 0, lost: 0 }, bold: { n: 0, lost: 0 }, misses: [] });
  for (const { game, g } of graded) {
    g.teams.forEach(id => row(id).games++);
    for (const d of g.calls) {
      if (!counts(d)) continue;
      const r = row(d.team), way = lean(d);
      r.calls++;
      if (!way) { r.agreed++; continue; }
      r.lost += d.lost; r[way].n++; r[way].lost += d.lost;
      r.misses.push({ ...d, way, game: game.id, week: game.week, opp: season.teams[g.teams.find(id => id !== d.team)] });
    }
  }
  const rows = Object.values(teams).filter(r => r.team);
  rows.forEach(r => { r.perGame = r.games ? r.lost / r.games : 0; r.misses.sort((a, b) => b.lost - a.lost); });
  rows.sort((a, b) => a.perGame - b.perGame);
  const sum = k => rows.reduce((t, r) => t + (k.includes(".") ? r[k.split(".")[0]][k.split(".")[1]] : r[k]), 0);
  return { year: season.year, current: season.current, rows,
    league: { calls: sum("calls"), agreed: sum("agreed"), timid: sum("timid.n"), bold: sum("bold.n"),
              timidLost: sum("timid.lost"), boldLost: sum("bold.lost") } };
}

/** The season's report card. Grades any finished games it hasn't seen before; onProgress(report, done, total) as it goes. */
export async function seasonReport(year, onProgress = () => {}) {
  const season = await seasonGames(year);
  const finished = season.games.filter(x => x.state === "post");
  const store = readStore();
  const graded = () => finished.filter(x => store.games[x.id]).map(x => ({ game: x, g: store.games[x.id] }));
  const todo = finished.filter(x => !store.games[x.id]);
  let done = finished.length - todo.length, failed = 0, sinceSave = 0;
  onProgress(summarize(season, graded()), done, finished.length);
  const work = async () => {
    for (let x; (x = todo.shift());) {
      try {
        const g = await loadGame(x.id, null, { keep: false }).then(addFourthDowns).then(addConversions);
        store.games[x.id] = { teams: [g.home.id, g.away.id], calls: decisions(g) };
        if (++sinceSave >= 8) { writeStore(store); sinceSave = 0; onProgress(summarize(season, graded()), done + 1, finished.length); }
      } catch (e) { console.error(e); failed++; }
      done++;
    }
  };
  await Promise.all(Array.from({ length: WORKERS }, work));
  writeStore(store);
  return { ...summarize(season, graded()), failed, games: finished.length };
}
