"""Win probability model shared by training (train.py) and the live dashboard.

Every situation is described from the point of view of the team with the ball,
using these raw fields:

    score_differential, game_seconds_remaining, half_seconds_remaining,
    down, ydstogo, yardline_100, posteam_timeouts_remaining,
    defteam_timeouts_remaining, posteam_spread, home, receive_2h_ko, is_ot

In overtime, game_seconds_remaining and half_seconds_remaining are the time
left in the OT period.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "wp_live.json"

FEATURES = ["score_differential", "diff_time_ratio", "posteam_spread", "spread_time",
            "game_seconds_remaining", "half_seconds_remaining", "down", "ydstogo", "yardline_100",
            "posteam_timeouts_remaining", "defteam_timeouts_remaining", "home", "receive_2h_ko", "is_ot"]
# Common-sense rules the trees must follow. Without the field-position ones the model can wobble
# (e.g. right after kickoff), which is harmless on a chart but flips close 4th-down calls.
MONOTONE = {"score_differential": 1, "diff_time_ratio": 1, "posteam_spread": 1, "spread_time": 1,
            "posteam_timeouts_remaining": 1, "defteam_timeouts_remaining": -1,
            "yardline_100": -1, "ydstogo": -1, "down": -1}


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    """Adds the time-interaction features. Overtime is treated as 'end of game' for these."""
    d = d.copy()
    elapsed = np.where(d.is_ot == 1, 1.0, (3600 - d.game_seconds_remaining) / 3600)
    d["spread_time"] = d.posteam_spread * np.exp(-4 * elapsed)
    d["diff_time_ratio"] = d.score_differential / np.exp(-4 * elapsed)
    return d


def new_classifier(n_estimators=3000, early_stopping_rounds=100) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=n_estimators, learning_rate=0.03, max_depth=5, min_child_weight=50,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist", random_state=0,
        monotone_constraints=tuple(MONOTONE.get(f, 0) for f in FEATURES),
        eval_metric="logloss", early_stopping_rounds=early_stopping_rounds,
    )


class WPModel:
    def __init__(self, path: Path = MODEL_PATH):
        self.clf = xgb.XGBClassifier()
        self.clf.load_model(path)

    def predict(self, states: list[dict], neutral_site: bool = False) -> np.ndarray:
        """WP for the team with the ball in each state. At a neutral site the
        home-field flag is meaningless, so we average both values of it."""
        if not states:
            return np.array([])
        d = add_features(pd.DataFrame(states))
        if not neutral_site:
            return self.clf.predict_proba(d[FEATURES])[:, 1]
        p_home = self.clf.predict_proba(d.assign(home=1)[FEATURES])[:, 1]
        p_away = self.clf.predict_proba(d.assign(home=0)[FEATURES])[:, 1]
        return (p_home + p_away) / 2


def save_meta(info: dict):
    MODEL_PATH.with_name("wp_live_meta.json").write_text(json.dumps({"features": FEATURES, **info}, indent=2))
