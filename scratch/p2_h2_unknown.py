"""Bộ đếm tổng hợp: với cảnh có đích là TÊN LẠ (không phải mô tả qua bản đồ), phân bố độ tự tin loại đích (validation vs test).
Chỉ đếm, không in câu test."""
import sys, pickle, collections, numpy as np
sys.path.insert(0, "src")
from common import *
import nlp3
p = nlp3.MissionParser3.load(OUT / "nlp3_final.pkl")
for split in ("validation", "test"):
    rows = load_json(DATA / split / "observations.json")
    W = pickle.load(open(CACHE / f"world_{split}_v1final.pkl", "rb"))
    n = len(rows) // 10; pm = []; c = collections.Counter()
    for i in range(n):
        w = W[rows[i * 10]["image"]]["world"]
        if w is None: continue
        L = w["landmarks"]
        r = p.parse(rows[i * 10]["mission"], {t for t, v in L.items() if v}, L)
        if r.get("goal_ref") and r["goal_ref"][0] == "pos": continue
        if r.get("goal_known"): c["known"] += 1; continue
        d = np.asarray(r["goal_dist"]); pm.append(float(d.max()))
        c["goal type not on map"] += r["goal"] not in {t for t, v in L.items() if v}
    pm = np.array(pm)
    print(split, n, dict(c), "unknown", len(pm), "| P max: <0.5", (pm < .5).sum(), "<0.7", (pm < .7).sum(), "<0.9", (pm < .9).sum(), "mean", pm.mean().round(3))
