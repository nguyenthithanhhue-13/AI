"""Khi đường ngắn nhất buộc có đặc trưng X (đông / đi ngang địa điểm) mà đường dài hơn 2 đoạn tránh được, robot đi vòng bao
nhiêu phần trăm? Tách theo điều kiện cảnh.  python scratch/p2_a29_avoid.py"""
import sys, collections, numpy as np
sys.path.insert(0, "scratch")
from p2_lib import *
from p2_budget import first_q
from p2_a20_dagperc import bfs_dist

D = load("train")[:700]
st = collections.defaultdict(collections.Counter)
for x in D:
    w = x["w"]; s = w["robot"]
    for legged in (False, True):
        V2 = bfs_dist(w, x["legs"][-1], legged)
        V = bfs_dist(w, x["legs"][0], legged, {v: V2.get(v) for v in x["legs"][0]}) if len(x["legs"]) == 2 else V2
        for feat, th in (("đông", [0, 1, 0, 0, 0, 0, 0]), ("địa điểm", [0, 0, 0, 1, 0, 0, 0]), ("đông+đđ", [0, 1, 0, 1, 0, 0, 0])):
            th = np.array(th, float)
            m0 = min(first_q(x, th, 0, legged)); m2 = min(first_q(x, th, 2, legged))
            if not (m0 < INF and m2 < m0):
                continue
            for r in ([4] if legged else [1, 2, 3, 5, 6, 7, 8, 9]):
                y = x["y"][r]; n2 = (s[0] + DRC[y][0], s[1] + DRC[y][1])
                det = V.get(n2, 99) >= V.get(s, 99)
                for c, v in (("tất cả", True), ("mưa", w["rain"]), ("khô", not w["rain"]), ("gấp", x["m"]["urgent"]),
                             ("không gấp", not x["m"]["urgent"]), ("dễ vỡ", x["m"]["fragile"]), ("không dễ vỡ", not x["m"]["fragile"])):
                    if v:
                        st[(r, feat)][c + " n"] += 1; st[(r, feat)][c + " vòng"] += det
for (r, feat), c in sorted(st.items()):
    print(f"R{r} tránh {feat:8s}: " + "  ".join(f"{k} {c[k + ' vòng']}/{c[k + ' n']}" for k in
          ("tất cả", "mưa", "khô", "gấp", "không gấp", "dễ vỡ", "không dễ vỡ")))
