// 4th-down helper: compares going for it, kicking a field goal, and punting by the
// win probability each choice leads to. The odds come from docs/fourth.json (see live/fourth.py).
import { predict } from "./model.js?v=734f5105";

const KICKOFF_YARDLINE = 70;  // same assumption as espn.js: the receiving team starts at its own 30
const EXPECTED_PAT = 0.95;
const MAX_FG_DISTANCE = 66;
const MIN_PUNT_YARDLINE = 30; // nobody punts from inside the opponent's 30
const SNAP_TO_KICK = 17;      // kick distance = yards to the end zone + 7 (snap) + 10 (end zone)

let tablesPromise = null;
export const loadTables = () => (tablesPromise ??= fetch("fourth.json").then(r => {
  if (!r.ok) throw new Error(`fourth.json: ${r.status}`);
  return r.json();
}));

const at = (arr, i) => arr[Math.max(0, Math.min(arr.length - 1, i))];

// The next situation, from the point of view of whoever has the ball afterwards.
function after(s, { flip, yardline, points = 0, secs }) {
  const clock = { game_seconds_remaining: Math.max(1, s.game_seconds_remaining - secs),
                  half_seconds_remaining: Math.max(1, s.half_seconds_remaining - secs) };
  const field = { down: 1, ydstogo: Math.min(10, yardline), yardline_100: yardline };
  if (!flip) return { ...s, ...clock, ...field, score_differential: s.score_differential + points };
  return { ...s, ...clock, ...field,
    score_differential: -(s.score_differential + points),
    posteam_timeouts_remaining: s.defteam_timeouts_remaining, defteam_timeouts_remaining: s.posteam_timeouts_remaining,
    posteam_spread: -s.posteam_spread, home: 1 - s.home,
    receive_2h_ko: s.period <= 2 ? 1 - s.receive_2h_ko : 0 };
}

/** The choices for one 4th-down situation. Each outcome is [chance, next situation, offense keeps the ball?]. */
export function choices(s, t) {
  const ytg = s.ydstogo, ytez = s.yardline_100, out = [];

  const goal = ytez <= ytg;
  const pConv = at(goal ? t.convert_goal : t.convert, ytg - 1);
  const gain = Math.max(ytg, at(t.gain, ytg - 1));
  const success = goal || ytez - gain <= 0
    ? [after(s, { flip: true, yardline: KICKOFF_YARDLINE, points: 6 + EXPECTED_PAT, secs: 6 }), false]  // touchdown
    : [after(s, { flip: false, yardline: ytez - gain, secs: 6 }), true];
  out.push({ key: "go", label: "Go for it", chance: pConv, chanceText: "to convert",
    outcomes: [[pConv, ...success], [1 - pConv, after(s, { flip: true, yardline: 100 - ytez, secs: 6 }), false]] });

  const dist = ytez + SNAP_TO_KICK;
  if (dist <= MAX_FG_DISTANCE) {
    const pMake = at(t.fg_make, dist - t.fg_min_distance);
    out.push({ key: "fg", label: `${dist}-yard field goal`, chance: pMake, chanceText: "to make",
      outcomes: [[pMake, after(s, { flip: true, yardline: KICKOFF_YARDLINE, points: 3, secs: 5 }), false],
                 // a miss gives the ball back at the spot of the kick, or the 20 if that's closer to the end zone
                 [1 - pMake, after(s, { flip: true, yardline: Math.min(80, 100 - (ytez + 7)), secs: 5 }), false]] });
  }

  if (ytez >= MIN_PUNT_YARDLINE) {
    const opp = Math.round(at(t.punt_opp_yardline, ytez - 1));
    const spot = opp > 50 ? `their own ${100 - opp}` : opp < 50 ? `your ${opp}` : "midfield";
    out.push({ key: "punt", label: "Punt", chance: null, chanceText: `they usually start at ${spot}`,
      outcomes: [[1, after(s, { flip: true, yardline: opp, secs: 8 }), false]] });
  }
  return out;
}

/** What the play actually was, from ESPN's play type. */
function whatTheyDid(kind) {
  const k = (kind || "").toLowerCase();
  if (!k || k.includes("penalty") || k.includes("timeout")) return null;
  if (k.includes("punt")) return "punt";
  if (k.includes("field goal")) return "fg";
  return "go";
}

/** Adds `fourth` to every 4th-down point (and the live "now" point):
 *  { options: [{key, label, wp, chance, chanceText}], best, margin, did }. wp = offense's win chance. */
export async function addFourthDowns(game) {
  const spots = game.points.filter(p => p.down === 4 && p.ydstogo > 0 && p.yardline_100 > 0 && (p.play_id || p.kind === "now"));
  if (!spots.length) return game;
  const t = await loadTables();
  const all = spots.map(s => choices(s, t));
  const states = all.flatMap(opts => opts.flatMap(o => o.outcomes.map(([, st]) => st)));
  const wps = await predict(states, game.neutral);
  let i = 0;
  spots.forEach((s, n) => {
    const options = all[n].map(o => ({
      key: o.key, label: o.label, chance: o.chance, chanceText: o.chanceText,
      wp: o.outcomes.reduce((sum, [p, , keeps]) => { const w = wps[i++]; return sum + p * (keeps ? w : 1 - w); }, 0),
    }));
    const ranked = [...options].sort((a, b) => b.wp - a.wp);
    s.fourth = { options, best: ranked[0].key, margin: ranked.length > 1 ? ranked[0].wp - ranked[1].wp : null,
                 did: s.kind === "now" ? null : whatTheyDid(s.kind) };
  });
  return game;
}
