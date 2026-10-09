"""Kiểm tra R0 KHÔNG cần nhãn trên validation và test: luật từ điển R0 có trả về rỗng (rơi sang nhánh dự phòng) không, bước dự đoán
(trong file nộp) có nằm trên đường ít đoạn nhất của bản đồ CV không. Chỉ đếm tổng hợp.
    python scratch/p2_r0_check.py [file dự đoán test]"""
import sys, json, pickle, collections
sys.path.insert(0, "src")
from common import *
import nlp3
from pipeline_p2 import legs_from_mission, cv_worlds
from strat_ml import SG, lm_excl, theta_of, StrategyML, REL
import numpy as np
PRED = sys.argv[1] if len(sys.argv) > 1 else "outputs/private_result/predictions_p19_sel.json"
for split, mode in (("validation", "trainonly"), ("test", "final")):
    rows = load_json(DATA / split / "observations.json")
    worlds = cv_worlds(split, "v1final")
    p = nlp3.MissionParser3.load(OUT / f"nlp3_{mode}.pkl")
    cfg = json.load(open(OUT / ("strategy_hybrid_final.json" if mode == "final" else "strategy_hybrid.json"), encoding="utf-8"))["0"]
    preds = json.load(open(PRED)) if split == "test" else None
    if split == "validation":
        labels = load_split("validation")[1]
    c = collections.Counter()
    for si in range(len(rows) // 10):
        w = worlds[rows[si * 10]["image"]]["world"]
        if w is None:
            c["cv lỗi"] += 1; continue
        L = w["landmarks"]
        r = p.parse(rows[si * 10]["mission"], {t for t, v in L.items() if v}, L)
        m = nlp3.resolve3(r, L, w); legs = legs_from_mission(w, m)
        if not legs[-1]:
            c["không có đích"] += 1; continue
        d = StrategyML._lex_move(w, legs, 0, cfg, False, m["urgent"], m["fragile"])
        c["n"] += 1
        if d is None:
            c["luật R0 trả rỗng"] += 1
        q = SG(w, False, frozenset()).q(theta_of(), legs)
        b = min(q); short = [k for k in range(4) if q[k] <= b + 1e-6]
        c["có đường"] += np.isfinite(b)
        y = preds[si * 10] if preds else labels[si * 10]
        c["bước (nộp/nhãn) trên đường ngắn nhất"] += y in short
        c["luật R0 = bước nộp/nhãn"] += d == y
        c["số bước ngắn nhất >= 2"] += len(short) >= 2
    n = c["n"]
    print(split, {k: (v if k == "n" else round(v / n, 3)) for k, v in sorted(c.items())})
