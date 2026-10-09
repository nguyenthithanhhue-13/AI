"""R9 tham lam theo đêm / ngày: điểm = Manhattan + a*đông + b*vào địa điểm + c*mái che + rẽ; hòa -> thứ tự tương đối tốt nhất."""
import sys, itertools, collections
sys.path.insert(0, "scratch")
from p2_lib import *
man = lambda p, t: abs(p[0] - t[0]) + abs(p[1] - t[1])
def prep(D):
    out = []
    for x in D:
        w = x["w"]; s = w["robot"]; tg = x["legs"][0]; h = w["heading"]
        lm = {p for v in w["landmarks"].values() for p in v} - {p for L in x["legs"] for p in L}
        mv = [(d, min(man((s[0]+DRC[d][0], s[1]+DRC[d][1]), t) for t in tg), i[0] == "crowded", (s[0]+DRC[d][0], s[1]+DRC[d][1]) in lm, i[0] == "covered", REL[h][d])
              for d, i in w["adj"].get(s, {}).items() if edge_ok(i, False)]
        out.append((mv, x["y"][9], "đêm" if x["s"]["style"] == "night" else "ngày"))
    return out
TR = prep(load("train")); VA = prep(load("validation"))
RELO = ["".join(o) for o in itertools.permutations("SRLB")]
def acc(P, a, b, c, t, o):
    ok = 0
    for mv, y, _ in P:
        sc = {d: m + a * cr + b * l + c * cv + (0 if rl == "S" else t) for d, m, cr, l, cv, rl in mv}
        mn = min(sc.values()); A = [d for d, v in sc.items() if v <= mn + 1e-9]
        rel = {d: rl for d, *_, rl in mv}
        ok += min(A, key=lambda d: o.index(rel[d])) == y
    return ok
for g in ("đêm", "ngày"):
    T = [z for z in TR if z[2] == g]; V = [z for z in VA if z[2] == g]
    best = max(((acc(T, a, b, c, t, o), a, b, c, t, o) for a, b, c, t in itertools.product([0, 1, 1.5, 2, 2.5, 3, 5], [0, 1, 1.5, 2, 2.5, 3, 5], [-1, -0.5, 0, 0.5, 1], [0, 0.25, 0.5]) for o in ["SRLB", "SLRB", "RSLB", "LSRB"]))
    print(g, f"train {best[0]}/{len(T)} = {best[0]/len(T):.3f}", "validation", f"{acc(V, *best[1:])}/{len(V)} = {acc(V, *best[1:])/len(V):.3f}", best[1:])
