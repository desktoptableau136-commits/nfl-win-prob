// "Watch now": how far a live game's win probability is likely to move over the next 5 minutes
// of game clock. It looks up how much the game is in doubt and how much time is left in
// docs/watch.json (see live/watch.py), nudged by the next snap's stakes (leverage.js).

let tablesPromise = null;
const loadTables = () => (tablesPromise ??= fetch("watch.json").then(r => {
  if (!r.ok) throw new Error(`watch.json: ${r.status}`);
  return r.json();
}));

/** Adds `watch` to a live game: { move (expected WP movement), index (1 = a typical 5 minutes) }. Run after addLeverage. */
export async function addWatch(game) {
  const s = game.now;
  if (game.state !== "in" || !s || !Number.isFinite(game.wp_home)) return game;
  const t = await loadTables();
  const doubt = Math.min(t.doubt_bins - 1, Math.floor((1 - Math.abs(2 * game.wp_home - 1)) * t.doubt_bins));
  const cols = t.log_move[0].length;
  const col = s.is_ot ? cols - 1 : Math.min(cols - 2, Math.floor(s.game_seconds_remaining / t.time_bin));
  const log = t.log_move[doubt][col] + t.stakes_coef * Math.log((s.stakes?.index ?? 1) + 0.05);
  const move = Math.max(0, Math.exp(log) - t.floor);
  game.watch = { move, index: move / t.avg_move };
  return game;
}
