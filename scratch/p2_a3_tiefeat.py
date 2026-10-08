"""Soi các ca HÒA của một robot (chi phí chính cho trước): đặc trưng nào của bước được chọn khác các bước hòa còn lại.
    python scratch/p2_a3_tiefeat.py <robot>"""
import sys, collections, math
sys.path.insert(0, "scratch")
from p2_lib import *

r = int(sys.argv[1])
D = load("train")
ec, tc = mk_ecost(), mk_tcost()


def path_targets(w, legs, ec, tc, legged):
    """Đích (cuối) và điểm ghé (nếu có) tối ưu cho từng bước đầu: trả về dict d -> (via_node, goal_node)."""
    out = {}
    s = w["robot"]
    for d, info in w["adj"].get(s, {}).items():
        if not edge_ok(info, legged):
            continue
        best = (INF, None)
        if len(legs) == 1:
            for g in legs[0]:
                q = qvals(w, [[g]], ec, tc, legged)
                if q[d] < best[0] - 1e-9:
                    best = (q[d], (None, g))
        else:
            for v in legs[0]:
                for g in legs[1]:
                    q = qvals(w, [[v], [g]], ec, tc, legged)
                    if q[d] < best[0] - 1e-9:
                        best = (q[d], (v, g))
        out[d] = best
    return out


feats = collections.Counter(); n = 0
rows = []
for x in D:
    w = x["w"]; y = x["y"][r]
    q = qvals(w, x["legs"], ec, tc, r == 4)
    a = argmins(q)
    if len(a) < 2:
        continue
    n += 1
    pt = path_targets(w, x["legs"], ec, tc, r == 4)
    s = w["robot"]; h = w["heading"]
    f = {}
    for d in a:
        v, g = pt[d][1]
        t = v if v is not None else g
        n2 = (s[0] + DRC[d][0], s[1] + DRC[d][1])
        f[d] = dict(man=abs(n2[0] - t[0]) + abs(n2[1] - t[1]), euc=math.hypot(n2[0] - t[0], n2[1] - t[1]),
                    st=w["adj"][s][d][0], rel=REL[h][d], vert=d in (0, 1),
                    axis_big=(abs(t[0] - s[0]) >= abs(t[1] - s[1])) == (d in (0, 1)),
                    deg=len(w["adj"].get(n2, {})), next_rc=n2)
    rows.append((x, a, y, f))
    feats["euc_min"] += f[y]["euc"] == min(f[d]["euc"] for d in a)
    feats["euc_max"] += f[y]["euc"] == max(f[d]["euc"] for d in a)
    feats["man_min"] += f[y]["man"] == min(f[d]["man"] for d in a)
    feats["axis_big"] += f[y]["axis_big"]
    feats["vert"] += f[y]["vert"]
    feats["horiz"] += not f[y]["vert"]
    feats["deg_max"] += f[y]["deg"] == max(f[d]["deg"] for d in a)
    feats["deg_min"] += f[y]["deg"] == min(f[d]["deg"] for d in a)
    for k in ("normal", "crowded", "covered"):
        if any(f[d]["st"] == k for d in a) and any(f[d]["st"] != k for d in a):
            feats[f"st={k} có/không"] += 1
            feats[f"st={k} chọn"] += f[y]["st"] == k
    feats["rel_" + f[y]["rel"]] += 1
    feats["rc_min"] += f[y]["next_rc"] == min(f[d]["next_rc"] for d in a)
    feats["rc_max"] += f[y]["next_rc"] == max(f[d]["next_rc"] for d in a)
print("số ca hòa", n)
for k, v in sorted(feats.items()):
    print(f"  {k:22s} {v:5d}  {v/n:.3f}")
pickle.dump(rows, open(f"cache/p2_ties_r{r}.pkl", "wb"))
