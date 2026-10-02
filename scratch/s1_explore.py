"""Khám phá chiến thuật: kiểm tra tham chiếu không gian và giả thuyết 'ít đoạn đường nhất'."""
import sys, pickle
from collections import Counter
sys.path.insert(0, "src")
from common import *
from strategy import *

rows, labels, scenes = load_split("train")
W = [world_from_scene(s) for s in scenes]
M = [mission_from_scene(s) for s in scenes]
Y = [labels[i * 10:(i + 1) * 10] for i in range(len(scenes))]

# 1) tham chiếu không gian có khớp rc trong scenes.json?
ok = bad = 0
for s, w, m in zip(scenes, W, M):
    for k in ("goal", "via"):
        ref = s["mission"][k + "_ref"]
        if ref:
            c = resolve_targets(w, m[k], m[k + "_ref"])
            if c == [tuple(ref["rc"])]:
                ok += 1
            else:
                bad += 1
                if bad < 5: print("BAD", ref, w["landmarks"][m[k]], w["landmarks"].get(ref["anchor"]))
print("ref resolve ok", ok, "bad", bad)
print("goal type count", Counter(len(w["landmarks"][m["goal"]]) for w, m in zip(W, M)))
print("label legal (edge exists, not closed, not wrong-way):",
      Counter((r, edge_cost(w["adj"][w["robot"]].get(y, ("closed", 0, 0)), {}, r == 4) < INF)
              for w, ys in zip(W, Y) for r, y in enumerate(ys) if True).most_common(30))

# 2) mỗi robot: nhãn có nằm trong tập bước đi ngắn nhất (ít đoạn) không?
for r in range(10):
    inset = uniq = n = 0
    rels = Counter()
    for w, m, ys in zip(W, M, Y):
        q = path_q(w, legs_for(w, m), {}, r == 4)
        best = min(q)
        ties = [d for d in range(4) if q[d] == best]
        n += 1
        inset += ys[r] in ties
        if len(ties) > 1 and ys[r] in ties:
            rels[("".join(sorted(REL[w["heading"]][d] for d in ties)), REL[w["heading"]][ys[r]])] += 1
    print(f"robot {r}: label in min-hop set {inset / n:.3f}")
    if r in (0, 4):
        for k, v in sorted(rels.items()): print("   ", k, v)
print("robots agree stats:", Counter(len(set(ys)) for ys in Y))
