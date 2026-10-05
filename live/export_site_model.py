"""Exports models/wp_live.json into the compact format the web page reads (docs/model.json).

    .venv\\Scripts\\python -m live.export_site_model

train.py runs this automatically after training.

Format: {"features": [...], "base_margin": float, "trees": [[left, right, feature, value, default_left], ...]}
Each tree is five parallel lists, one entry per node. A node with left == -1 is a
leaf, and its `value` is the leaf's score. Otherwise go left when
x[feature] < value (compared as 32-bit floats, like XGBoost), or follow
default_left when the feature is missing.
"""
import json
import math
from pathlib import Path

import numpy as np

from live.model import FEATURES, MODEL_PATH

SITE_MODEL = Path(__file__).resolve().parent.parent / "docs" / "model.json"


def f32(x: float) -> str:
    """Shortest decimal that round-trips to the same 32-bit float."""
    return np.format_float_positional(np.float32(x), unique=True, trim="-")


def export(src: Path = MODEL_PATH, dst: Path = SITE_MODEL) -> Path:
    learner = json.loads(src.read_text())["learner"]
    assert learner["feature_names"] == FEATURES, "model features don't match live/model.py"
    assert learner["objective"]["name"] == "binary:logistic"
    base_score = float(learner["learner_model_param"]["base_score"].strip("[]"))
    trees = []
    for t in learner["gradient_booster"]["model"]["trees"]:
        lists = [
            ",".join(map(str, t["left_children"])),
            ",".join(map(str, t["right_children"])),
            ",".join(map(str, t["split_indices"])),
            ",".join(f32(v) for v in t["split_conditions"]),
            ",".join(map(str, t["default_left"])),
        ]
        trees.append("[" + ",".join(f"[{x}]" for x in lists) + "]")
    dst.parent.mkdir(exist_ok=True)
    dst.write_text(
        '{"features":' + json.dumps(FEATURES)
        + ',"base_margin":' + repr(math.log(base_score / (1 - base_score)))
        + ',"trees":[\n' + ",\n".join(trees) + "]}\n"
    )
    return dst


if __name__ == "__main__":
    out = export()
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
