"""Robot 9 v2: chọn MỘT đích gần nhất (Manhattan) từ vị trí robot, rồi mới tham lam một bước."""
import sys, itertools
sys.path.insert(0, "src")
from common import *
from strategy import *

rows, labels, scenes = load_split("train")
W = [world_from_scene(s) for s in scenes]
M = [mission_from_scene(s) for s in scenes]
Y = [labels[i * 10 + 9] for i in range(len(scenes))]
man = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])


def greedy_q(w, m, P, tie="first"):
    targets = legs_for(w, m)[0]
    s = w["robot"]
    if tie == "first":
        t = min(targets, key=lambda t: man(s, t))            # hòa -> phần tử đầu (thứ tự hàng, cột)
    elif tie == "last":
        t = min(reversed(targets), key=lambda t: man(s, t))
    q = [INF] * 4
    for d, info in w["adj"].get(s, {}).items():
        c = edge_cost(info, P, False)
        if c == INF: continue
        n2 = (s[0] + DRC[d][0], s[1] + DRC[d][1])
        q[d] = P.get("lam", 1) * man(n2, t) + c - 1
    return q


if __name__ == "__main__":
    res = []
    for crowd, cover, lam, tie in itertools.product([0, 0.25, 0.5, 1, 1.5, 2, 3, 5], [0, -0.3, 0.3], [1, 100], ["first", "last"]):
        P = {"crowd": crowd, "cover": cover, "lam": lam}
        ok = sum(pick(greedy_q(w, m, P, tie), w["heading"]) == y for w, m, y in zip(W, M, Y))
        res.append((ok / len(Y), P, tie))
    res.sort(key=lambda t: -t[0])
    for r in res[:10]: print(r)
    P = res[0][1]
    for i, (w, m, y) in enumerate(zip(W, M, Y)):
        q = greedy_q(w, m, P, res[0][2])
        if pick(q, w["heading"]) != y:
            st = {ACTIONS[d]: info[0] + ("+s" if info[1] else "") + ("" if info[2] else "+wrongway") for d, info in w["adj"][w["robot"]].items()}
            print(i, "robot", w["robot"], ACTIONS[w["heading"]], "label", ACTIONS[y], "q", q, legs_for(w, m), st)
