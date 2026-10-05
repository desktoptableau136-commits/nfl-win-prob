"""Trains the live model: the notebook's XGBoost, plus overtime.

    .venv\\Scripts\\python -m live.train

Picks the number of trees on 2024, reports a test score on 2025, then refits on
every season and saves models/wp_live.json.
"""
import numpy as np
import pandas as pd
import nflreadpy as nfl
from sklearn.metrics import brier_score_loss, log_loss

from live import export_site_model
from live.model import FEATURES, MODEL_PATH, add_features, new_classifier, save_meta

SEASONS = list(range(2006, 2026))
COLS = ["season", "game_id", "play_id", "posteam", "home_team", "qtr", "game_seconds_remaining",
        "half_seconds_remaining", "down", "ydstogo", "yardline_100", "score_differential",
        "posteam_timeouts_remaining", "defteam_timeouts_remaining", "spread_line", "result", "vegas_wp"]


def load() -> pd.DataFrame:
    raw = nfl.load_pbp(SEASONS).select(COLS).to_pandas()
    df = raw[raw.posteam.notna() & raw.down.notna() & raw.game_seconds_remaining.notna()
             & raw.spread_line.notna()].copy()
    is_home = df.posteam == df.home_team
    df["home"] = is_home.astype(int)
    # ties (only possible after OT) count as "did not win"
    df["label"] = np.where(is_home, df.result > 0, df.result < 0).astype(int)
    df["posteam_spread"] = np.where(is_home, df.spread_line, -df.spread_line)
    opening_receiver = df.sort_values(["game_id", "play_id"]).groupby("game_id").posteam.first()
    df["receive_2h_ko"] = ((df.qtr <= 2) & (df.posteam != df.game_id.map(opening_receiver))).astype(int)
    df["is_ot"] = (df.qtr >= 5).astype(int)
    return add_features(df)


def grade(name, d, p):
    ot = (d.is_ot == 1).values
    late = ((d.game_seconds_remaining < 300) & (d.is_ot == 0)).values
    print(f"{name:20s} Brier {brier_score_loss(d.label, p):.4f}  log loss {log_loss(d.label, np.clip(p, 1e-6, 1 - 1e-6)):.4f}"
          f"  final 5 min {brier_score_loss(d.label[late], p[late]):.4f}  OT {brier_score_loss(d.label[ot], p[ot]):.4f} ({ot.sum()} plays)")


def main():
    df = load()
    print(f"{len(df):,} plays, {df.game_id.nunique():,} games, {df[df.is_ot == 1].game_id.nunique()} with overtime")
    train, valid, test = df[df.season <= 2023], df[df.season == 2024], df[df.season == 2025]

    clf = new_classifier()
    clf.fit(train[FEATURES], train.label, eval_set=[(valid[FEATURES], valid.label)], verbose=False)
    n_trees = clf.best_iteration + 1
    print(f"Trees: {n_trees}\nTest season 2025:")
    grade("our model", test, clf.predict_proba(test[FEATURES])[:, 1])
    bench = test[test.vegas_wp.notna()]
    grade("nflfastR vegas_wp", bench, bench.vegas_wp.values)

    final = new_classifier(n_estimators=n_trees, early_stopping_rounds=None)
    final.fit(df[FEATURES], df.label, verbose=False)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    final.save_model(MODEL_PATH)
    save_meta({"seasons": [SEASONS[0], SEASONS[-1]], "n_trees": n_trees})
    print(f"Saved {MODEL_PATH}")
    print(f"Saved {export_site_model.export()} (for the web page)")


if __name__ == "__main__":
    main()
