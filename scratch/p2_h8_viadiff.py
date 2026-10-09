"""Bộ đếm TỔNG HỢP trên test (không in câu): kiểu bất đồng giữa hai bộ đọc về ĐIỂM GHÉ và ĐÍCH; tên đã biết hay lạ."""
import sys, pickle, collections
sys.path.insert(0, "src")
from common import *
import nlp3
from pipeline_p2 import legs_from_mission
pa = nlp3.MissionParser3.load(OUT / "nlp3_trainonly.pkl"); pb = nlp3.MissionParser3.load(OUT / "nlp3_final.pkl")
for split in ("validation", "test"):
    rows = load_json(DATA / split / "observations.json")
    W = pickle.load(open(CACHE / f"world_{split}_v1final.pkl", "rb"))
    c = collections.Counter()
    for i in range(len(rows) // 10):
        w = W[rows[i * 10]["image"]]["world"]
        if w is None:
            continue
        L = w["landmarks"]; t = rows[i * 10]["mission"]
        a = pa.parse(t, {k for k, v in L.items() if v}, L); b = pb.parse(t, {k for k, v in L.items() if v}, L)
        if a.get("via") != b.get("via"):
            k = "ghé: A có / B không" if a.get("via") and not b.get("via") else "ghé: A không / B có" if b.get("via") and not a.get("via") else "ghé: khác loại"
            c[k] += 1
            c[k + " | B via_known=" + str(bool(b.get("via_known")))] += 1
        if a["goal"] != b["goal"]:
            c["đích khác loại"] += 1
            c["đích khác | A known=%s B known=%s" % (bool(a.get("goal_known")), bool(b.get("goal_known")))] += 1
            if a.get("via") == b["goal"] or b.get("via") == a["goal"]:
                c["đích<->ghé đổi chỗ"] += 1
    print(split, dict(sorted(c.items())))
