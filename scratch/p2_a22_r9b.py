import sys, math, collections
sys.path.insert(0, "scratch")
from p2_lib import *
D = load("train"); r = 9
c = collections.Counter(); n = 0
man = lambda p, t: abs(p[0] - t[0]) + abs(p[1] - t[1])
for x in D:
    w = x["w"]; s = w["robot"]; y = x["y"][r]; tg = x["legs"][0]; h = w["heading"]
    lm = {p for v in w["landmarks"].values() for p in v} - {p for L in x["legs"] for p in L}
    legal = [d for d, info in w["adj"].get(s, {}).items() if edge_ok(info, False)]
    M = {d: min(man((s[0] + DRC[d][0], s[1] + DRC[d][1]), t) for t in tg) for d in legal}
    best = min(M.values())
    if M[y] == best: continue
    n += 1
    for d in legal:
        if M[d] != best: continue
        n2 = (s[0] + DRC[d][0], s[1] + DRC[d][1]); info = w["adj"][s][d]
        c["bị bỏ: đoạn " + info[0]] += 1
        c["bị bỏ: vào địa điểm"] += n2 in lm
        c["bị bỏ: quan hệ " + REL[h][d]] += 1
        c["bị bỏ: ngõ cụt (bậc 1)"] += len([1 for dd, ii in w["adj"].get(n2, {}).items() if edge_ok(ii, False)]) == 1
    n2 = (s[0] + DRC[y][0], s[1] + DRC[y][1]); info = w["adj"][s][y]
    c["chọn: đoạn " + info[0]] += 1
    c["chọn: vào địa điểm"] += n2 in lm
    c["chọn: quan hệ " + REL[h][y]] += 1
    c["chọn: dMan " + str(M[y] - best)] += 1
print("số ca bỏ bước tham lam:", n)
for k, v in sorted(c.items()): print(f"   {k:30s} {v}")
