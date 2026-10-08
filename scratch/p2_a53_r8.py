import sys, collections
sys.path.insert(0, "scratch")
from p2_lib import *
from p2_a20_dagperc import bfs_dist
c = collections.Counter()
for x in load("train"):
    w = x["w"]; s = w["robot"]; h = w["heading"]
    V = bfs_dist(w, x["legs"][-1], False)
    if len(x["legs"]) == 2: V = bfs_dist(w, x["legs"][0], False, {v: V.get(v) for v in x["legs"][0]})
    via = "ghé" if x["m"]["via"] else "không ghé"
    for r in (0, 8, 1, 2):
        y = x["y"][r]; n2 = (s[0] + DRC[y][0], s[1] + DRC[y][1])
        sp = V.get(n2, 99) < V.get(s, 99)
        c[(r, via, "n")] += 1; c[(r, via, "rel" + REL[h][y])] += 1; c[(r, via, "ngắn nhất")] += sp
for r in (0, 8, 1, 2):
    for via in ("ghé", "không ghé"):
        n = c[(r, via, "n")]
        print(f"R{r} {via:9s} n={n}  trên đường ngắn nhất {c[(r, via, 'ngắn nhất')]/n:.3f}  " + " ".join(f"{k}={c[(r, via, 'rel'+k)]/n:.2f}" for k in "SRLB"))
