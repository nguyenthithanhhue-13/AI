"""Cảnh train mà bước đi THẬT bị TRỘI hoàn toàn (mọi đường của nó đều có đường của bước khác <= mọi đặc trưng, kể cả rẽ bước đầu)
-> không bộ trọng số không âm nào giải thích được (mái che xét cả 2 dấu). Đếm theo robot + tả đặc trưng các cảnh đó."""
import sys, pickle, collections, numpy as np
sys.path.insert(0, "scratch")
from p2_lib import *
D = load("train") + load("validation")
P = {leg: pickle.load(open(f"cache/p2_pareto_train_{leg}.pkl", "rb")) + pickle.load(open(f"cache/p2_pareto_validation_{leg}.pkl", "rb")) for leg in (0, 1)}
FIRST = {"S": [0, 0, 0], "R": [1, 0, 0], "L": [0, 1, 0], "B": [0, 0, 1]}
def full(rel, F, sign):
    G = F.astype(float).copy(); G[:, 2] *= sign
    return np.hstack([G, np.tile(FIRST[rel], (len(G), 1))])
for r in range(10):
    leg = int(r == 4); c = collections.Counter(); ex = []
    for i, x in enumerate(D):
        fr = P[leg][i]; y = x["y"][r]
        if y not in fr:
            c["nhãn không đi được"] += 1; continue
        dom_all = True
        for sign in (1, -1):
            Fy = full(fr[y][0], fr[y][1], sign)
            others = np.vstack([full(fr[d][0], fr[d][1], sign) for d in fr if d != y]) if len(fr) > 1 else np.zeros((0, 11))
            # bước thật bị trội nếu MỌI đường p của nó có q (bước khác) với q <= p và q != p
            dom = len(others) > 0 and all(((others <= p).all(1) & (others < p).any(1)).any() for p in Fy)
            dom_all &= dom
        c["bị trội"] += dom_all; c["n"] += 1
        if dom_all and len(ex) < 3:
            ex.append(i)
    print(f"R{r}: bước thật bị trội hoàn toàn {c['bị trội']}/{c['n']} = {c['bị trội'] / c['n']:.3f}; nhãn không đi được {c['nhãn không đi được']}")
