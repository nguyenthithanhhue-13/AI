"""Bộ đếm TỔNG HỢP (không in câu test): hai bộ đọc (chỉ học train / final) bất đồng ở thành phần nào trên validation và test;
và trên validation, bên nào đúng."""
import sys, pickle, collections
sys.path.insert(0, "src"); sys.path.insert(0, "scratch")
from common import *
import nlp3
from pipeline_p2 import legs_from_mission
pa = nlp3.MissionParser3.load(OUT / "nlp3_trainonly.pkl"); pb = nlp3.MissionParser3.load(OUT / "nlp3_final.pkl")
for split in ("validation", "test"):
    rows = load_json(DATA / split / "observations.json")
    W = pickle.load(open(CACHE / f"world_{split}_v1final.pkl", "rb"))
    truth = None
    if split == "validation":
        from p2_lib import load
        truth = load("validation")
    c = collections.Counter()
    for i in range(len(rows) // 10):
        w = W[rows[i * 10]["image"]]["world"]
        if w is None:
            continue
        L = w["landmarks"]; t = rows[i * 10]["mission"]
        ra = nlp3.resolve3(pa.parse(t, {k for k, v in L.items() if v}, L), L, w)
        rb = nlp3.resolve3(pb.parse(t, {k for k, v in L.items() if v}, L), L, w)
        la = legs_from_mission(w, ra); lb = legs_from_mission(w, rb)
        d = []
        if sorted(la[-1]) != sorted(lb[-1]): d.append("đích")
        if (len(la) == 2) != (len(lb) == 2) or (len(la) == 2 and sorted(la[0]) != sorted(lb[0])): d.append("ghé")
        if ra["urgent"] != rb["urgent"]: d.append("gấp")
        if ra["fragile"] != rb["fragile"]: d.append("vỡ")
        for k in d:
            c[k] += 1
        c["bất đồng"] += bool(d); c["n"] += 1
        if d and truth:
            x = truth[i]; tl = sorted(map(tuple, x["legs"][-1]))
            c["đích: train-only đúng"] += sorted(la[-1]) == tl; c["đích: final đúng"] += sorted(lb[-1]) == tl
    print(split, dict(c))
