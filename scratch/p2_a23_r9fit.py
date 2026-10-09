"""R9: điểm bước d = Manhattan(giao lộ kế, điểm đến gần nhất) + a*đông + b*vào địa điểm + c*mái che + rẽ; hòa -> thứ tự."""
import sys, math, collections, itertools
sys.path.insert(0, "scratch")
from p2_lib import *
D = load(sys.argv[1] if len(sys.argv) > 1 else "train"); r = 9
man = lambda p, t: abs(p[0] - t[0]) + abs(p[1] - t[1])
pre = []
for x in D:
    w = x["w"]; s = w["robot"]; tg = x["legs"][0]; h = w["heading"]
    lm = {p for v in w["landmarks"].values() for p in v} - {p for L in x["legs"] for p in L}
    mv = []
    for d, info in w["adj"].get(s, {}).items():
        if not edge_ok(info, False): continue
        n2 = (s[0] + DRC[d][0], s[1] + DRC[d][1])
        mv.append((d, min(man(n2, t) for t in tg), info[0] == "crowded", n2 in lm, info[0] == "covered", REL[h][d]))
    pre.append((mv, x["y"][r], x))
RELO = list(itertools.permutations("SRLB")); ABS = list(itertools.permutations(range(4)))
best = []
for a, b, c, t in itertools.product([0, 1, 1.5, 2, 2.5, 3, 4, 5], [0, 1, 1.5, 2, 2.5, 3, 4, 5], [-1, -0.5, 0, 0.5, 1], [0, 0.25, 0.5]):
    ok = 0; amb = 0; ins = 0
    res_rel = collections.Counter()
    for mv, y, x in pre:
        sc = {d: m + a * cr + b * l + c * cv + (0 if rl == "S" else t) for d, m, cr, l, cv, rl in mv}
        mn = min(sc.values())
        A = [d for d, v in sc.items() if v <= mn + 1e-9]
        ins += y in A; ok += A == [y]
    best.append((ins, ok, a, b, c, t))
best.sort(reverse=True)
for z in best[:8]: print(z)
