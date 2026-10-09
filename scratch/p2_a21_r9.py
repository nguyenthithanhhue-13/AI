"""R9 (tham lam): bước của nhãn có làm giảm khoảng cách (Manhattan / Euclid) tới điểm đến kế tiếp không; điểm đến là bản nào."""
import sys, math, collections, itertools
sys.path.insert(0, "scratch")
from p2_lib import *
D = load("train"); r = 9
c = collections.Counter()
for x in D:
    w = x["w"]; s = w["robot"]; y = x["y"][r]; tg = x["legs"][0]
    legal = [d for d, info in w["adj"].get(s, {}).items() if edge_ok(info, False)]
    n2 = (s[0] + DRC[y][0], s[1] + DRC[y][1])
    c["nhãn hợp lệ"] += y in legal
    man = lambda p, t: abs(p[0] - t[0]) + abs(p[1] - t[1])
    dm = min(man(n2, t) for t in tg) - min(man(s, t) for t in tg)
    c[f"dMan={dm}"] += 1
    best = min(min(man((s[0] + DRC[d][0], s[1] + DRC[d][1]), t) for t in tg) for d in legal)
    c["nhãn đạt min Manhattan (bản gần nhất)"] += min(man(n2, t) for t in tg) == best
    eb = min(min(math.hypot(s[0] + DRC[d][0] - t[0], s[1] + DRC[d][1] - t[1]) for t in tg) for d in legal)
    c["nhãn đạt min Euclid"] += abs(min(math.hypot(n2[0] - t[0], n2[1] - t[1]) for t in tg) - eb) < 1e-9
print(len(D), sorted(c.items()))
