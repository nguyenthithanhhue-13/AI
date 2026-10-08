import sys, itertools
sys.path.insert(0, "scratch")
from p2_fast import *
TR = [x for x in load("train") if not x["m"]["via"]][:500]; VA = [x for x in load("validation") if not x["m"]["via"]]
def run(D, p, o="SRLB"):
    ok = 0
    for x in D:
        a = argmins(SG(x["w"], False, lm_excl(x)).q(theta_of(**p), x["legs"]), 1e-6)
        ok += bool(a) and min(a, key=lambda d: o.index(REL[x["w"]["heading"]][d])) == x["y"][8]
    return ok
res = []
for tl, tb, tr, cr, lm in itertools.product([0, 1, 3, 10, 100], [0, 2, 10, 100], [0, 0.25, 1], [0, 1, 3], [0, 1, 3]):
    p = dict(tL=tl, tB=tb, tR=tr, crowd=cr, lm=lm)
    res.append((run(TR, p), p))
res.sort(key=lambda z: -z[0])
for s, p in res[:6]:
    print(f"train {s}/{len(TR)}={s/len(TR):.3f}  validation {run(VA, p)}/{len(VA)}  {p}", flush=True)
