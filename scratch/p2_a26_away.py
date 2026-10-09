"""Mỗi robot: tỉ lệ nhãn đi XA đích theo số đoạn (BFS), tách theo điều kiện; và mức trùng nhãn với R0."""
import sys, collections
sys.path.insert(0, "scratch")
from p2_lib import *
from p2_a20_dagperc import bfs_dist
D = load("train")
st = collections.defaultdict(collections.Counter)
for x in D:
    w = x["w"]; s = w["robot"]
    for r in range(10):
        V = bfs_dist(w, x["legs"][-1], r == 4)
        if len(x["legs"]) == 2:
            V = bfs_dist(w, x["legs"][0], r == 4, {v: V.get(v) for v in x["legs"][0]})
        y = x["y"][r]; n2 = (s[0] + DRC[y][0], s[1] + DRC[y][1])
        away = V.get(n2, 99) >= V.get(s, 99)
        for k, v in (("all", 1), ("rain", w["rain"]), ("urg", x["m"]["urgent"]), ("frag", x["m"]["fragile"])):
            st[r][(k, bool(v), "tot")] += 1; st[r][(k, bool(v), "away")] += away
        st[r]["same_as_R0"] += y == x["y"][0]
for r in range(10):
    c = st[r]
    f = lambda k, v: f"{c[(k, v, 'away')] / max(1, c[(k, v, 'tot')]):.3f}"
    print(f"R{r}: đi xa {f('all', True)} | mưa {f('rain', True)} khô {f('rain', False)} | gấp {f('urg', True)} không {f('urg', False)} | dễ vỡ {f('frag', True)} không {f('frag', False)} | trùng R0 {c['same_as_R0']/len(D):.3f}")
