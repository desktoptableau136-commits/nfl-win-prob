// Kick or go for 2 after a touchdown. The decision plays out the rest of the game drive by drive
// (see live/two_point.py, which runs the same calculation and explains why the tree model
// isn't used for it); the odds come from docs/twopoint.json.

const SPAN = 80;  // leads from -80 to +80

let tablesPromise = null;
const load = () => (tablesPromise ??= fetch("twopoint.json").then(r => {
  if (!r.ok) throw new Error(`twopoint.json: ${r.status}`);
  return r.json();
}).then(t => ({ ...t, V: solve(t) })));

/** V[n][ball][lead + SPAN]: chance team A wins with `lead` and n drives left, ball 0 = A, 1 = B. Mirrors solve() in live/two_point.py. */
function solve(t) {
  const L = 2 * SPAN + 1, maxN = t.drives_left[0].length - 1;
  const at = (arr, d) => arr[Math.max(0, Math.min(2 * SPAN, d + SPAN))];
  const end = Float64Array.from({ length: L }, (_, i) => i > SPAN ? 1 : i < SPAN ? 0 : 0.5);
  const V = [[end, end]];
  for (let n = 1; n <= maxN; n++) {
    V.push([0, 1].map(ball => {
      const sign = ball === 0 ? 1 : -1, best = ball === 0 ? Math.max : Math.min, nxt = V[n - 1][1 - ball];
      return Float64Array.from({ length: L }, (_, i) => {
        const d = i - SPAN;
        const base = at(nxt, d + 6 * sign);
        const kick = t.pat * at(nxt, d + 7 * sign) + (1 - t.pat) * base;
        const go2 = t.two_pt * at(nxt, d + 8 * sign) + (1 - t.two_pt) * base;
        const td = best(kick, go2), none = at(nxt, d);
        const fg = best(at(nxt, d + 3 * sign), t.fg_to_td * td + (1 - t.fg_to_td) * none);
        return t.p_td * td + t.p_fg * fg + (1 - t.p_td - t.p_fg) * none;
      });
    }));
  }
  return V;
}

/** Scoring team's chance after each try. `lead` = its lead after the TD, before the try. */
function decide(t, lead, gameSeconds) {
  const dist = t.drives_left[Math.min(60, Math.max(1, Math.ceil(gameSeconds / 60))) - 1];
  const after = d => dist.reduce((sum, p, n) => sum + p * t.V[n][1][Math.max(0, Math.min(2 * SPAN, d + SPAN))], 0);
  return { kick: t.pat * after(lead + 1) + (1 - t.pat) * after(lead),
           two: t.two_pt * after(lead + 2) + (1 - t.two_pt) * after(lead) };
}

/** What the team tried after the touchdown, from ESPN's play text. */
function tried(text) {
  const x = (text || "").toLowerCase();
  if (/two[- ]point|2[- ]p(oin)?t/.test(x)) return "two";
  if (x.includes("extra point") || x.includes("kick is")) return "kick";
  return null;
}

/** Adds `pat` to regulation touchdown snaps, and to the live "now" point while the try is pending:
 *  { options: [{key, label, chance, chanceText, wp}], best, margin, did, team }.
 *  The margin comes from the drive-by-drive calculation; on the live panel, `wp` is the page's
 *  win chance for the scoring team (with the usual extra point) plus that difference. */
export async function addConversions(game) {
  const home = game.home.id, away = game.away.id, pts = game.points;
  const spots = [];  // [point, scoring team, lead after the TD, seconds left]
  pts.forEach((p, i) => {
    const next = pts[i + 1];
    if (!p.play_id || !p.scoring || p.is_ot || !next || next.home_score === undefined) return;
    const gained = { [home]: next.home_score - p.home_score, [away]: next.away_score - p.away_score };
    const scorer = [home, away].find(t => gained[t] >= 6 && gained[t] <= 8);
    if (!scorer || !/touchdown/i.test(`${p.kind} ${p.text}`)) return;
    const mine = scorer === home ? p.home_score : p.away_score, theirs = scorer === home ? p.away_score : p.home_score;
    spots.push([p, scorer, mine + 6 - theirs, p.game_seconds_remaining]);
  });
  const now = game.now;
  if (game.state === "in" && now?.pat_pending && !now.is_ot) {
    const scorer = now.pat_pending, s = { [home]: game.home.score, [away]: game.away.score };
    spots.push([now, scorer, s[scorer] - s[scorer === home ? away : home], now.game_seconds_remaining]);
  }
  if (!spots.length) return game;
  const t = await load();
  for (const [p, scorer, lead, secs] of spots) {
    const d = decide(t, lead, secs);
    const level = p.kind === "now" ? (scorer === home ? p.wp_home : 1 - p.wp_home) : null;
    const options = [
      { key: "kick", label: "Kick the extra point", chance: t.pat, chanceText: "to make", wp: level },
      { key: "two", label: "Go for 2", chance: t.two_pt, chanceText: "to convert", wp: level === null ? null : Math.min(1, Math.max(0, level + d.two - d.kick)) }];
    p.pat = { options, best: d.two > d.kick ? "two" : "kick", margin: Math.abs(d.two - d.kick), team: scorer,
              did: p.kind === "now" ? null : tried(p.text) };
  }
  return game;
}

export const _test = { load, decide };
