"""So sánh các robot trên cùng cảnh: robot khó (4, 7-fragile, 9) giống robot nào nhất?"""
import sys, os
sys.path.insert(0, "src")
from collections import Counter
from common import *
from strategy import *

rows, labels, scenes = load_split("train")
W = [world_from_scene(s) for s in scenes]
M = [mission_from_scene(s) for s in scenes]
Y = [labels[i * 10:(i + 1) * 10] for i in range(len(scenes))]


def agree(r, cond):
    idx = [i for i in range(len(Y)) if cond(i)]
    return [round(sum(Y[i][r] == Y[i][k] for i in idx) / len(idx), 3) for k in range(10)], len(idx)


print("R7 fragile     ", agree(7, lambda i: M[i]["fragile"]))
print("R7 frag, rain  ", agree(7, lambda i: M[i]["fragile"] and W[i]["rain"]))
print("R7 frag, dry   ", agree(7, lambda i: M[i]["fragile"] and not W[i]["rain"]))
print("R7 frag, urgent", agree(7, lambda i: M[i]["fragile"] and M[i]["urgent"]))
print("R4             ", agree(4, lambda i: True))
print("R4 no stairs in scene", agree(4, lambda i: not any(e["stairs"] for e in scenes[i]["edges"])))
print("R9             ", agree(9, lambda i: True))
print("R9 no via      ", agree(9, lambda i: not M[i]["via"]))

# R7 fragile: có tránh bậc thang/đường nào không? xem trạng thái của đoạn đầu tiên được chọn
c = Counter()
for i in range(len(Y)):
    if M[i]["fragile"]:
        st = W[i]["adj"][W[i]["robot"]][Y[i][7]]
        st0 = W[i]["adj"][W[i]["robot"]][Y[i][0]]
        c[(st[0], st0[0], Y[i][7] == Y[i][0])] += 1
print("R7 fragile first-edge status vs R0:", sorted(c.items(), key=lambda t: -t[1]))
# R7 fragile với các mô hình: nhãn có trong tập min của crowd=2?
for P in [{"crowd": 2}, {"crowd": 2, "cover": 0}, {}]:
    ins = n = 0
    rel = Counter()
    for i in range(len(Y)):
        if not M[i]["fragile"]: continue
        q = path_q(W[i], legs_for(W[i], M[i]), P, False)
        b = min(q); ties = [d for d in range(4) if q[d] <= b + 1e-6]
        n += 1; ins += Y[i][7] in ties
        if Y[i][7] not in ties:
            rel[round(q[Y[i][7]] - b, 2)] += 1
    print(P, "label in argmin set", ins / n, "gap dist", sorted(rel.items())[:12])
