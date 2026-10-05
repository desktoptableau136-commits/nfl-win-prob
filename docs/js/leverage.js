// "What's at stake on the next snap": plays the next snap out many ways and averages how far
// win probability moves. The odds come from docs/leverage.json (see live/leverage.py, which
// uses the same recipe to measure a typical snap; keep the two in step).
import { predict } from "./model.js";
import { choices, loadTables as loadFourth } from "./fourth.js";

const KICKOFF_YARDLINE = 70;
const EXPECTED_PAT = 0.95;

let tablesPromise = null;
const loadTables = () => (tablesPromise ??= fetch("leverage.json").then(r => {
  if (!r.ok) throw new Error(`leverage.json: ${r.status}`);
  return r.json();
}));

function nextSnap(s, t) {
  const secs = s.half_seconds_remaining <= 120 ? t.secs.late : t.secs.normal;
  const clock = { game_seconds_remaining: Math.max(1, s.game_seconds_remaining - secs),
                  half_seconds_remaining: Math.max(1, s.half_seconds_remaining - secs) };
  const keep = (down, ydstogo, spot) => ({ ...s, ...clock, down, ydstogo, yardline_100: spot });
  const flip = (spot, points = 0) => ({ ...s, ...clock, down: 1, ydstogo: Math.min(10, spot), yardline_100: spot,
    score_differential: -(s.score_differential + points),
    posteam_timeouts_remaining: s.defteam_timeouts_remaining, defteam_timeouts_remaining: s.posteam_timeouts_remaining,
    posteam_spread: -s.posteam_spread, home: 1 - s.home, receive_2h_ko: s.period <= 2 ? 1 - s.receive_2h_ko : 0 });
  return { keep, flip };
}

/** [chance, next state, offense keeps the ball?] for a 1st-3rd down snap. Mirrors outcomes() in live/leverage.py. */
function outcomes(s, t) {
  const { keep, flip } = nextSnap(s, t);
  const key = `${s.down}-${t.ytg_buckets.findIndex(edge => s.ydstogo <= edge)}`;
  const pTo = t.turnover[key], ytez = s.yardline_100, ytg = s.ydstogo;
  const out = [[pTo, flip(Math.min(99, Math.max(1, 100 - ytez))), false]];
  for (const g of t.gains[key]) {
    const w = (1 - pTo) / t.gains[key].length;
    let spot = ytez - Math.round(g);
    if (spot <= 0) { out.push([w, flip(KICKOFF_YARDLINE, 6 + EXPECTED_PAT), false]); continue; }
    spot = Math.min(99, spot);
    if (ytez - spot >= ytg) out.push([w, keep(1, Math.min(10, spot), spot), true]);
    else out.push([w, keep(s.down + 1, ytg - (ytez - spot), spot), true]);
  }
  return out;
}

/** A few outcomes worth showing, as [label, next state, offense keeps the ball?]. */
function scenarios(s, t) {
  const { keep, flip } = nextSnap(s, t);
  const ytez = s.yardline_100, ytg = s.ydstogo, out = [["Touchdown", flip(KICKOFF_YARDLINE, 6 + EXPECTED_PAT), false]];
  if (ytg < ytez) out.push(["First down", keep(1, Math.min(10, ytez - ytg), ytez - ytg), true]);
  out.push([s.down < 3 ? "No gain" : "No gain, 4th down next", keep(s.down + 1, ytg, ytez), true]);
  out.push(["Turnover", flip(Math.min(99, Math.max(1, 100 - ytez))), false]);
  return out;
}

const LABELS = [[0.5, "Low"], [1.5, "Normal"], [2.5, "High"], [Infinity, "Very high"]];

/** Adds `stakes` to game.now during a live game:
 *  { index (1 = typical snap), label, wp (offense's now), scenarios: [{label, wp}] (offense's view) }. */
export async function addLeverage(game) {
  const s = game.now;
  if (game.state !== "in" || !s || !(s.down >= 1 && s.down <= 4) || !(s.yardline_100 >= 1)) return game;
  const t = await loadTables();
  let outs, shown = [];
  if (s.down === 4) {
    // the model's pick from the 4th-down helper, played out
    const opts = choices(s, await loadFourth());
    const wps = await predict(opts.flatMap(o => o.outcomes.map(([, st]) => st)), game.neutral);
    let i = 0;
    const scored = opts.map(o => ({ o, wp: o.outcomes.reduce((sum, [p, , keeps]) => { const w = wps[i++]; return sum + p * (keeps ? w : 1 - w); }, 0) }));
    outs = scored.sort((a, b) => b.wp - a.wp)[0].o.outcomes;
  } else {
    outs = outcomes(s, t);
    shown = scenarios(s, t);
  }
  const states = [s, ...outs.map(([, st]) => st), ...shown.map(([, st]) => st)];
  const wps = await predict(states, game.neutral);
  const now = wps[0];
  const offense = (w, keeps) => keeps ? w : 1 - w;
  const swing = outs.reduce((sum, [p, , keeps], i) => sum + p * Math.abs(offense(wps[1 + i], keeps) - now), 0);
  const index = swing / t.avg_swing;
  s.stakes = { index, swing, label: LABELS.find(([hi]) => index < hi)[1], wp: now,
    scenarios: shown.map(([label, , keeps], i) => ({ label, wp: offense(wps[1 + outs.length + i], keeps) })) };
  return game;
}
