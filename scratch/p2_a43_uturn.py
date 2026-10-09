import sys, collections
sys.path.insert(0, "scratch")
from p2_lib import *
from p2_a20_dagperc import bfs_dist
D = load("train")
st = collections.defaultdict(collections.Counter)
for x in D:
    w = x["w"]; s = w["robot"]; h = w["heading"]; b = OPP[h]
    info = w["adj"].get(s, {}).get(b)
    if not info or not edge_ok(info, False): continue
    V = bfs_dist(w, x["legs"][-1], False)
    if len(x["legs"]) == 2: V = bfs_dist(w, x["legs"][0], False, {v: V.get(v) for v in x["legs"][0]})
    n2 = (s[0] + DRC[b][0], s[1] + DRC[b][1])
    others = [d for d, i in w["adj"].get(s, {}).items() if edge_ok(i, False) and d != b]
    best_other = min((V.get((s[0] + DRC[d][0], s[1] + DRC[d][1]), 99) for d in others), default=99)
    if not (V.get(n2, 99) < best_other): continue      # quay đầu là bước ngắn nhất duy nhất
    gap = best_other - V.get(n2, 99)
    for r in range(10):
        if r == 4: continue
        for k in ("tất cả", "mưa" if w["rain"] else "khô", "đêm" if x["s"]["style"] == "night" else "ngày", f"chênh {gap}",
                  "gấp" if x["m"]["urgent"] else "không gấp", "dễ vỡ" if x["m"]["fragile"] else "không dễ vỡ"):
            st[r][k + " n"] += 1; st[r][k + " quay"] += x["y"][r] == b
keys = ["tất cả", "mưa", "khô", "đêm", "ngày", "chênh 2", "chênh 4", "gấp", "không gấp", "dễ vỡ", "không dễ vỡ"]
for r in sorted(st):
    print(f"R{r}: " + "  ".join(f"{k} {st[r][k + ' quay']}/{st[r][k + ' n']}" for k in keys))
