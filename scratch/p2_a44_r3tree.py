import sys, numpy as np
sys.path.insert(0, "scratch")
from p2_lib import *
from p2_a20_dagperc import bfs_dist
from sklearn.tree import DecisionTreeClassifier, export_text
r = int(sys.argv[1]) if len(sys.argv) > 1 else 3
X, Y = [], []
NAMES = ["rain", "night", "urg", "frag", "via", "back_crowd", "back_cover", "back_normal", "back_lm", "back_stairs", "dirU", "dirD", "dirL", "dirR",
         "n_alt", "alt_crowd", "alt_cover", "dist", "n_lm", "rows", "cols", "sun_style"]
for split in ("train",):
    for x in load(split):
        w = x["w"]; s = w["robot"]; h = w["heading"]; b = OPP[h]
        info = w["adj"].get(s, {}).get(b)
        if not info or not edge_ok(info, r == 4): continue
        V = bfs_dist(w, x["legs"][-1], r == 4)
        if len(x["legs"]) == 2: V = bfs_dist(w, x["legs"][0], r == 4, {v: V.get(v) for v in x["legs"][0]})
        n2 = (s[0] + DRC[b][0], s[1] + DRC[b][1])
        others = [d for d, i in w["adj"].get(s, {}).items() if edge_ok(i, r == 4) and d != b]
        bo = min((V.get((s[0] + DRC[d][0], s[1] + DRC[d][1]), 99) for d in others), default=99)
        if not (V.get(n2, 99) < bo): continue
        L = {p for v in w["landmarks"].values() for p in v} - {p for Lg in x["legs"] for p in Lg}
        alt = [w["adj"][s][d][0] for d in others]
        X.append([w["rain"], x["s"]["style"] == "night", x["m"]["urgent"], x["m"]["fragile"], bool(x["m"]["via"]),
                  info[0] == "crowded", info[0] == "covered", info[0] == "normal", n2 in L, info[1], b == 0, b == 1, b == 2, b == 3,
                  len(others), sum(a == "crowded" for a in alt), sum(a == "covered" for a in alt), V.get(s, 0),
                  sum(len(v) for v in w["landmarks"].values()), x["s"]["grid"]["rows"], x["s"]["grid"]["cols"], x["s"]["style"] == "classic"])
        Y.append(x["y"][r] == b)
X = np.array(X, float); Y = np.array(Y)
t = DecisionTreeClassifier(max_depth=4, min_samples_leaf=15).fit(X, Y)
print(f"R{r}: {len(Y)} ca, quay đầu {Y.mean():.2f}, cây độ sâu 4 đúng {t.score(X, Y):.3f}")
print(export_text(t, feature_names=NAMES))
