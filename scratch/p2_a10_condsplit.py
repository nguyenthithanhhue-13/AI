"""Ca hòa của một robot: thứ tự hướng (tuyệt đối / tương đối) tốt nhất trong từng nhóm điều kiện."""
import sys, collections, itertools
sys.path.insert(0, "scratch")
from p2_lib import *
r = int(sys.argv[1])
single, multi = pickle.load(open(f"cache/p2_tiesplit_r{r}.pkl", "rb"))
ABS = list(itertools.permutations(range(4))); RELO = list(itertools.permutations("SRLB"))
def best(L):
    ra = collections.Counter(); rr = collections.Counter()
    for x, a, y, _ in L:
        h = x["w"]["heading"]
        for o in ABS: ra[o] += min(a, key=lambda d: o.index(d)) == y
        for o in RELO: rr[o] += min(a, key=lambda d: o.index(REL[h][d])) == y
    n = len(L) or 1
    (oa, va), = ra.most_common(1); (orr, vr), = rr.most_common(1)
    return f"n={len(L):4d} tuyệt đối {''.join('UDLR'[i] for i in oa)} {va/n:.3f} | tương đối {''.join(orr)} {vr/n:.3f}"
L = single + multi
conds = {
 "rain": lambda x: x["w"]["rain"], "urgent": lambda x: x["m"]["urgent"], "fragile": lambda x: x["m"]["fragile"],
 "via": lambda x: bool(x["m"]["via"]), "heading": lambda x: ACTIONS[x["w"]["heading"]], "style": lambda x: x["s"]["style"],
 "goal_ref": lambda x: (x["m"]["goal_ref"] or {}).get("kind"), "rows": lambda x: x["s"]["grid"]["rows"],
}
for name, f in conds.items():
    g = collections.defaultdict(list)
    for t in L: g[f(t[0])].append(t)
    for k in sorted(g, key=str): print(f"{name:8s}={str(k):12s} {best(g[k])}")
