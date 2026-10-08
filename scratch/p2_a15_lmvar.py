import sys, collections, itertools
sys.path.insert(0, "scratch")
from p2_lib import *
r = int(sys.argv[1]); D = load("train"); leg = r == 4; BIG = 1000.0
def lmset(x, var):
    w = x["w"]; m = x["m"]
    allp = {p for v in w["landmarks"].values() for p in v}
    if var == "all": return allp
    if var == "excl_legs": return allp - {p for L in x["legs"] for p in L}
    if var == "excl_types": return allp - set(w["landmarks"].get(m["goal"], [])) - set(w["landmarks"].get(m["via"], []) if m["via"] else [])
for var in ("all", "excl_legs", "excl_types"):
    inset = 0; nt = 0; ties = []
    for x in D:
        L = lmset(x, var)
        ec = lambda info, n, d, L=L: BIG + ((n[0] + DRC[d][0], n[1] + DRC[d][1]) in L)
        q = qvals(x["w"], x["legs"], ec, mk_tcost(), leg)
        a = argmins(q, 1e-6); y = x["y"][r]
        inset += y in a
        if len(a) > 1 and y in a: ties.append((x, a, y))
    print(var, "trong argmin", inset, "còn hòa", len(ties), flush=True)
    pickle.dump(ties, open(f"cache/p2_ties_lm_{var}_r{r}.pkl", "wb"))
