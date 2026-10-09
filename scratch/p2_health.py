"""Bộ đếm SỨC KHỎE tổng hợp của hệ thống trên validation và test (chỉ đếm, không in nội dung câu / ảnh test).
So phân bố để tìm khâu nào hỏng riêng trên test."""
import sys, pickle, collections, numpy as np
sys.path.insert(0, "src")
from common import *
import nlp3
from pipeline_p2 import legs_from_mission

p = nlp3.MissionParser3.load(OUT / "nlp3_final.pkl")
for split in ("validation", "test"):
    rows = load_json(DATA / split / "observations.json")
    W = pickle.load(open(CACHE / f"world_{split}_v1final.pkl", "rb"))
    c = collections.Counter(); n = len(rows) // 10
    nl = []; nn = []; dup = []
    for i in range(n):
        img = rows[i * 10]["image"]; w = W[img]["world"]
        if w is None:
            c["CV lỗi"] += 1; continue
        L = w["landmarks"]
        nlm = sum(len(v) for v in L.values()); nl.append(nlm); nn.append(len(w["adj"]))
        dup.append(sum(1 for v in L.values() if len(v) > 1))
        c["số địa điểm ngoài 8-14"] += not (8 <= nlm <= 14)
        r = p.parse(rows[i * 10]["mission"], {t for t, v in L.items() if v}, L)
        m = nlp3.resolve3(r, L, w)
        legs = legs_from_mission(w, m)
        c["mô tả qua bản đồ (đã chốt vị trí)"] += bool(r.get("goal_ref")) and r["goal_ref"][0] == "pos"
        c["có cụm mô tả nhưng không chốt được"] += ("mapref" in r) and not (r.get("goal_ref") and r["goal_ref"][0] == "pos")
        c["có điểm ghé"] += len(legs) == 2
        c["đích là tên đã biết"] += bool(r.get("goal_known"))
        c["gấp"] += bool(m["urgent"]); c["dễ vỡ"] += bool(m["fragile"])
        c["đích có 2 bản, không tham chiếu"] += len(legs[-1]) > 1
        c["tham chiếu chọn bản"] += bool(m.get("goal_ref")) and m["goal_ref"][0] != "pos"
        c["robot đứng trên địa điểm"] += w["robot"] in {q for v in L.values() for q in v}
    print(f"== {split}: {n} cảnh | số địa điểm TB {np.mean(nl):.2f} | số giao lộ TB {np.mean(nn):.1f} | loại 2 bản TB {np.mean(dup):.2f}")
    for k, v in sorted(c.items()):
        print(f"   {k:38s} {v:5d}  {v / n:.3f}")
