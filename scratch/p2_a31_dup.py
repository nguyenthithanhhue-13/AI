import sys, json, numpy as np, collections
sys.path.insert(0, "scratch")
from p2_fast import *
D = load("train")
st = collections.Counter()
for r in range(1, 10):
    p = json.load(open(f"cache/p2_cd_r{r}_all.txt"))["p"] if r != 9 else {"lm": 2, "crowd": 2}
    th = theta_of(**p)
    for x in D[:1000]:
        g = SG(x["w"], r == 4, lm_excl(x))
        a = argmins(g.q(th, x["legs"]), 1e-6)
        dup = any(len(L) > 1 for L in x["legs"])
        k = "nhiều bản" if dup else "một điểm đến"
        st[(r, k, "n")] += 1; st[(r, k, "ok")] += x["y"][r] in a
for r in range(1, 10):
    print(f"R{r}: " + "  ".join(f"{k}: {st[(r, k, 'ok')]}/{st[(r, k, 'n')]} = {st[(r, k, 'ok')]/max(1, st[(r, k, 'n')]):.3f}" for k in ("một điểm đến", "nhiều bản")))
