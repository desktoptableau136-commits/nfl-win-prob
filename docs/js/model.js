// Win probability model, evaluated in the browser.
// Mirrors live/model.py; the trees come from docs/model.json (see live/export_site_model.py).

let modelPromise = null;

export function loadModel() {
  modelPromise ??= fetch("model.json").then(r => {
    if (!r.ok) throw new Error(`model.json: ${r.status}`);
    return r.json();
  }).then(m => ({
    features: m.features,
    base: m.base_margin,
    trees: m.trees.map(([left, right, feat, value, defLeft]) => ({
      left: Int32Array.from(left), right: Int32Array.from(right), feat: Int32Array.from(feat),
      value: Float32Array.from(value), defLeft: Uint8Array.from(defLeft),
    })),
  }));
  return modelPromise;
}

// Same as add_features() in live/model.py: overtime counts as "end of game" for these.
function featureRow(s, features) {
  const elapsed = s.is_ot === 1 ? 1 : (3600 - s.game_seconds_remaining) / 3600;
  const row = { ...s, spread_time: s.posteam_spread * Math.exp(-4 * elapsed),
                diff_time_ratio: s.score_differential / Math.exp(-4 * elapsed) };
  // XGBoost compares in 32-bit floats; do the same so splits land identically
  return Float32Array.from(features, f => row[f] ?? NaN);
}

function margin(model, x) {
  let total = model.base;
  for (const t of model.trees) {
    let n = 0;
    while (t.left[n] !== -1) {
      const v = x[t.feat[n]];
      n = Number.isNaN(v) ? (t.defLeft[n] ? t.left[n] : t.right[n]) : v < t.value[n] ? t.left[n] : t.right[n];
    }
    total += t.value[n];
  }
  return total;
}

const sigmoid = z => 1 / (1 + Math.exp(-z));

/** WP for the team with the ball in each state. At a neutral site the
 *  home-field flag is meaningless, so we average both values of it. */
export async function predict(states, neutralSite = false) {
  const model = await loadModel();
  return states.map(s => {
    if (!neutralSite) return sigmoid(margin(model, featureRow(s, model.features)));
    const h = sigmoid(margin(model, featureRow({ ...s, home: 1 }, model.features)));
    const a = sigmoid(margin(model, featureRow({ ...s, home: 0 }, model.features)));
    return (h + a) / 2;
  });
}
