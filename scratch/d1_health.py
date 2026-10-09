"""Bộ đếm tổng hợp trên test (không in câu nào): tìm nhóm bất thường còn lại sau bản 0,9778.
Dùng bản đồ test đã cache trên máy này (đọc bằng MLP cũ; chỉ để chẩn đoán phần đọc câu)."""
import sys, pickle, re
sys.path.insert(0, "src")
from collections import Counter
import numpy as np
from common import *
import nlp2
from nlp2 import *
from strategy import q_values, INF
p = pickle.load(open(OUT / "nlp2_final.pkl", "rb"))
REL = r"\b(gan|xa|sat|canh|ke|giap|doi dien|cach|lien)\b"
for sp in ("validation", "test"):
    rows = load_json(DATA / sp / "observations.json"); worlds = pickle.load(open(f"cache/world_{sp}_final.pkl", "rb"))
    c = Counter(); n = 0
    for si in range(len(rows) // 10):
        w = worlds[rows[si * 10]["image"]]["world"]
        if w is None: continue
        n += 1
        present = {x for x, v in w["landmarks"].items() if v}
        t = rows[si * 10]["mission"]
        m = p.parse(t, present); r = resolve_with_map(m, w["landmarks"], w)
        ments = p._mentions(t)[0]
        c["A. không tìm thấy địa điểm nào trong câu"] += not ments
        c["B. robot thường không có đường tới đích (sau mọi luật)"] += min(q_values(w, r, 0)) == INF
        if not m["goal_known"]:
            gd = m["goal_dist"]; k = int(np.argmax(gd))
            c["C1. đích là tên lạ"] += 1
            c["C2.   loại đoán không có trên bản đồ (chắc chắn đoán sai trước khi lọc)"] += PLACE_TYPES[k] not in present
            pp = np.sort(np.array([gd[PLACE_TYPES.index(x)] for x in present]) / sum(gd[PLACE_TYPES.index(x)] for x in present))[::-1]
            c["C3.   sau khi lọc theo bản đồ vẫn mơ hồ (nhất - nhì < 0.5)"] += (pp[0] - (pp[1] if len(pp) > 1 else 0)) < 0.5
        if m["via"] is not None and not m["via_known"] and m["via_dist"] is not None:
            vd = m["via_dist"]; k = int(np.argmax(vd))
            c["D1. điểm ghé là tên lạ"] += 1
            c["D2.   loại đoán không có trên bản đồ"] += PLACE_TYPES[k] not in present
        c["E. có điểm ghé"] += r["via"] is not None
        for key in ("goal", "via"):
            if r[key] is None: continue
            nd = len(w["landmarks"].get(r[key], []))
            if nd >= 2 and r[key + "_ref"] is None:
                c[f"F1. {key} có 2 bản, không có hướng"] += 1
                cand = [x for x in ments if x["type"] == r[key]] if m[key + "_known"] else [x for x in ments if x["type"] is None]
                c[f"F2.   ...mà sau tên có từ gần/xa/cách trong 8 từ"] += any(re.search(REL, " ".join(x["toks"][x["j"]:x["j"] + 8])) for x in cand)
                c[f"F3.   ...mà sau tên có từ hướng (bắc/nam/trên/dưới/trái/phải...)"] += any(re.search(r"\b(bac|nam|dong|tay|tren|duoi|trai|phai)\b", " ".join(x["toks"][x["j"]:x["j"] + 6])) for x in cand)
            if nd >= 2 and r[key + "_ref"] is not None: c[f"F0. {key} có 2 bản, có hướng"] += 1
        c["G. gấp"] += bool(r["urgent"]); c["H. dễ vỡ"] += bool(r["fragile"])
        # tên đã biết không bị coi là gây nhiễu mà loại không có trên bản đồ
        nun = sum(x["type"] is None for x in ments)
        c[f"I. số tên lạ trong câu >= 2"] += nun >= 2
        roles_goal = [x for x in ments if x["type"] is None]
        c["J. câu có tên lạ nhưng đích lẫn điểm ghé đều là tên ĐÃ BIẾT (tên lạ bị coi là gây nhiễu / mốc)"] += bool(roles_goal) and m["goal_known"] and (m["via"] is None or m["via_known"])
    print(f"\n== {sp}: {n} cảnh")
    for k in sorted(c): print(f"   {c[k]:4d} ({c[k] / n:5.1%})  {k}")
nlp2.save_embed_cache()
