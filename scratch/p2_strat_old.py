"""(vòng private) Đo chiến thuật CŨ trên nhãn MỚI với thông tin đúng (scenes.json), đích = goal_ref.rc nếu có."""
import sys, collections
sys.path.insert(0, "src")
from common import *
import strategy

def legs_true(s):
    m = s["mission"]; w = world_from_scene(s)
    legs = []
    if m["via"]:
        legs.append([tuple(m["via_ref"]["rc"])] if m["via_ref"] else list(w["landmarks"][m["via"]]))
    legs.append([tuple(m["goal_ref"]["rc"])] if m["goal_ref"] else list(w["landmarks"][m["goal"]]))
    return legs

for split in ["train", "validation"]:
    rows, labels, scenes = load_split(split)
    preds = []
    for s in scenes:
        L = legs_true(s)
        strategy.legs_for = lambda w, m, L=L: L
        w, m = world_from_scene(s), mission_from_scene(s)
        preds += [strategy.predict(w, m, r) for r in range(10)]
    score, per = macro_accuracy(rows, labels, preds)
    print(f"{split}: macro {score:.4f}  " + " ".join(f"R{r}={v:.3f}" for r, v in per.items()))
