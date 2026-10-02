"""Đo module chiến thuật khi được cho thông tin ĐÚNG (scenes.json): trần trên của cả hệ thống."""
import sys
from common import *
from strategy import predict

for split in ["train", "validation"]:
    rows, labels, scenes = load_split(split)
    preds = []
    for s in scenes:
        w, m = world_from_scene(s), mission_from_scene(s)
        preds += [predict(w, m, r) for r in range(10)]
    score, per = macro_accuracy(rows, labels, preds)
    print(f"{split}: macro acc {score:.4f}  per robot {[round(v, 4) for v in per.values()]}")
    bad = [(rows[i]["id"]) for i in range(len(rows)) if preds[i] != labels[i]]
    print("   sai:", len(bad), bad[:20])
