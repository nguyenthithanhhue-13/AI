"""Các ca robot ĐI VÒNG (bước của nhãn không nằm trên đường ít đoạn nhất): so đặc trưng đường ngắn nhất theo bước nhãn
với đường ngắn nhất tổng thể (min / max trên DAG theo số đoạn của từng bước).
    python scratch/p2_a27_detour.py <robot>"""
import sys, collections, numpy as np
sys.path.insert(0, "scratch")
from p2_lib import *
from p2_a20_dagperc import Dag, FN, IX, NF, bfs_dist

r = int(sys.argv[1])
D = load("train")
leg = r == 4
st = collections.Counter(); n = 0


def per_move(g, d, sign):
    """Giá trị nhỏ nhất của (sign · đặc trưng) trên các đường ít đoạn nhất bắt đầu bằng d (tính riêng từng đặc trưng)."""
    res = {}
    for k in ("crowd", "cover", "normal", "oneway", "tR", "tL", "tB"):
        th = np.zeros(NF); th[IX[k]] = sign
        res[k] = g.first_moves(th)
    return res


KEYS = ["crowd", "cover", "lm", "turns", "oneway"]
for x in D:
    w = x["w"]; s = w["robot"]; y = x["y"][r]
    g = Dag(x, leg)
    V = g.V[0]
    n2 = (s[0] + DRC[y][0], s[1] + DRC[y][1])
    if V.get(n2, 99) < V.get(s, 99):
        continue          # không đi vòng
    n += 1
    # đường tốt nhất của bước nhãn: số đoạn ít nhất từ n2 (cộng 1), so với tối ưu
    st["thêm đoạn " + str(V.get(n2, 99) + 1 - V.get(s, 99))] += 1
    # đặc trưng nhỏ nhất của các đường ngắn nhất tổng thể
    lmset = {p for v in w["landmarks"].values() for p in v} - {p for L in x["legs"] for p in L}
    fm = {}
    for k in ("crowd", "cover"):
        th = np.zeros(NF); th[IX[k]] = 1 if k == "crowd" else -1
        fm[k] = min(c for c, f in g.first_moves(th).values()) if g.first_moves(th) else None
    th = np.zeros(NF)
    for t in PLACES: th[IX["lm_" + t]] = 1
    fm["lm"] = min(c for c, f in g.first_moves(th).values())
    th = np.zeros(NF); th[IX["tR"]] = th[IX["tL"]] = th[IX["tB"]] = th[IX["tR0"]] = th[IX["tL0"]] = th[IX["tB0"]] = 1
    fm["turns"] = min(c for c, f in g.first_moves(th).values())
    st["đường ngắn nhất buộc: đông>=1"] += fm["crowd"] >= 1
    st["đường ngắn nhất buộc: mái che=0"] += fm["cover"] == 0
    st["đường ngắn nhất buộc: đi ngang địa điểm>=1"] += fm["lm"] >= 1
    st["đường ngắn nhất buộc: rẽ>=2"] += fm["turns"] >= 2
    st["bước nhãn là quay đầu"] += REL[w["heading"]][y] == "B"
    st["bước nhãn: đoạn " + w["adj"][s][y][0]] += 1
print(f"R{r}: số ca đi vòng {n}/{len(D)}")
for k, v in sorted(st.items()):
    print(f"   {k:40s} {v:4d}  {v / max(n, 1):.3f}")
