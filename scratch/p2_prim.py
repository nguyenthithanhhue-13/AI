"""Tìm TIÊU CHÍ CHÍNH cho một robot: tổng có trọng số mà nhãn LUÔN nằm trong tập tối ưu (như "số đoạn" của R0).
Quét rộng (đông, địa điểm, mái che, thường, rẽ phải, rẽ trái, quay đầu, bậc thang), theo nhóm điều kiện; in top theo tỉ lệ
"nhãn ∈ argmin" và kích thước trung bình tập argmin (nhỏ = phân biệt tốt).
    python scratch/p2_prim.py <robot> <khóa> [n]"""
import sys, itertools, collections, numpy as np
sys.path.insert(0, "scratch")
from p2_fast import *

r = int(sys.argv[1]); key = sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 300
NM = lambda x: x["s"]["style"] == "night"
KF = {"all": lambda x: "all", "w": lambda x: x["s"]["weather"], "nd_w": lambda x: ("đêm" if NM(x) else "ngày") + "/" + x["s"]["weather"],
      "nd": lambda x: "đêm" if NM(x) else "ngày", "v": lambda x: "ghé=" + str(bool(x["m"]["via"])),
      "w_v": lambda x: x["s"]["weather"] + "/ghé=" + str(bool(x["m"]["via"])),
      "nd_f": lambda x: ("đêm" if NM(x) else "ngày") + "/dv=" + str(x["m"]["fragile"]),
      "nd_u": lambda x: ("đêm" if NM(x) else "ngày") + "/gấp=" + str(x["m"]["urgent"])}[key]
TR = load("train")
groups = collections.defaultdict(list)
for x in TR: groups[KF(x)].append(x)
GRID = list(itertools.product([0, 1, 2, 4], [0, 1, 2, 4], [-0.5, 0, 1], [0, 0.5, 2], [0, 0.5, 2], [0, 1, 4]))   # đông, đđ, mái, phải, trái, quay
for k in sorted(groups):
    D = groups[k][:N]
    G = [SG(x["w"], r == 4, lm_excl(x)) for x in D]
    res = []
    for cr, lm, cv, tr, tl, tb in GRID:
        th = theta_of(crowd=cr, lm=lm, cover=cv, tR=tr, tL=tl, tB=tb)
        ins = 0; sz = 0
        for g, x in zip(G, D):
            a = argmins(g.q(th, x["legs"]), 1e-6)
            ins += x["y"][r] in a; sz += len(a)
        res.append((ins / len(D), -sz / len(D), (cr, lm, cv, tr, tl, tb)))
    res.sort(reverse=True)
    print(f"R{r} {key}={k} n={len(D)}: " + " | ".join(f"ins={a:.3f} cỡ={-b:.2f} (đông,đđ,mái,phải,trái,quay)={p}" for a, b, p in res[:3]), flush=True)
