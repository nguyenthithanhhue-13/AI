"""R9: điểm bước = khoảng cách (Manhattan | số đoạn BFS | Euclid) từ giao lộ kế tới điểm đến + phạt; hòa -> thứ tự tương đối.
Tách theo nhóm điều kiện; đo train / validation."""
import sys, itertools, math, collections
sys.path.insert(0, "scratch")
from p2_lib import *
from p2_a20_dagperc import bfs_dist

KEY = sys.argv[1] if len(sys.argv) > 1 else "nd"
ROBOT = int(sys.argv[2]) if len(sys.argv) > 2 else 9
KF = {"nd": lambda x: "đêm" if x["s"]["style"] == "night" else "ngày", "none": lambda x: "all",
      "nd_w": lambda x: ("đêm" if x["s"]["style"] == "night" else "ngày") + x["s"]["weather"],
      "nd_f": lambda x: ("đêm" if x["s"]["style"] == "night" else "ngày") + str(x["m"]["fragile"]),
      "nd_f_u": lambda x: ("đêm" if x["s"]["style"] == "night" else "ngày") + str(x["m"]["fragile"]) + str(x["m"]["urgent"]),
      "nd_f_w": lambda x: ("đêm" if x["s"]["style"] == "night" else "ngày") + str(x["m"]["fragile"]) + x["s"]["weather"],
      "w": lambda x: x["s"]["weather"], "nd_u": lambda x: ("đêm" if x["s"]["style"] == "night" else "ngày") + str(x["m"]["urgent"])}[KEY]


def prep(D):
    out = []
    for x in D:
        w = x["w"]; s = w["robot"]; tg = x["legs"][0]; h = w["heading"]
        V = bfs_dist(w, tg, False)
        lm = {p for v in w["landmarks"].values() for p in v} - {p for L in x["legs"] for p in L}
        mv = []
        for d, i in w["adj"].get(s, {}).items():
            if not edge_ok(i, False):
                continue
            n2 = (s[0] + DRC[d][0], s[1] + DRC[d][1])
            mv.append((d, min(abs(n2[0] - t[0]) + abs(n2[1] - t[1]) for t in tg), V.get(n2, 50),
                       min(math.hypot(n2[0] - t[0], n2[1] - t[1]) for t in tg),
                       i[0] == "crowded", n2 in lm, i[0] == "covered", REL[h][d]))
        out.append((mv, x["y"][ROBOT], KF(x)))
    return out


TR = prep(load("train")); VA = prep(load("validation"))
ORD = ["".join(o) for o in itertools.permutations("SRLB")]


def acc(P, di, a, b, c, t, o):
    ok = 0
    for mv, y, _ in P:
        sc = {z[0]: z[1 + di] + a * z[4] + b * z[5] + c * z[6] + (0 if z[7] == "S" else t) for z in mv}
        mn = min(sc.values())
        A = [d for d, v in sc.items() if v <= mn + 1e-9]
        rel = {z[0]: z[7] for z in mv}
        ok += min(A, key=lambda d: o.index(rel[d])) == y
    return ok


tot_t = tot_v = 0
BEST = {}
for g in sorted({z[2] for z in TR}):
    T = [z for z in TR if z[2] == g]; V = [z for z in VA if z[2] == g]
    best = max((acc(T, di, a, b, c, t, o), di, a, b, c, t, o) for di in (0, 1, 2)
               for a, b, c, t in itertools.product([0, 1, 2, 3, 5], [0, 1, 2, 3, 5], [-1, -0.5, 0, 0.5], [0, 0.25, 0.5])
               for o in ("SRLB", "SLRB", "RSLB", "LSRB", "SRBL"))
    v = acc(V, *best[1:])
    BEST[g] = best[1:]
    tot_t += best[0]; tot_v += v
    print(f"{g}: train {best[0]}/{len(T)}  validation {v}/{len(V)}  (khoảng cách={['Manhattan', 'BFS', 'Euclid'][best[1]]}, đông={best[2]}, đđ={best[3]}, mái={best[4]}, rẽ={best[5]}, {best[6]})", flush=True)
import json; json.dump(BEST, open(f"cache/p2_greedy_r{ROBOT}_{KEY}.json", "w"), ensure_ascii=False)
print(f"R{ROBOT} {KEY}: train {tot_t/len(TR):.4f}  validation {tot_v/len(VA):.4f}")
