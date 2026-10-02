from s6_diag import *
from s11_greedy import greedy_q

diag(7, {"turn": 1.25, "turnB": 14, "crowd": 5.5}, cond=lambda i: M[i]["fragile"], show=5)
print("--- R9 misses")
P = {"crowd": 0.5}
for i in range(len(Y)):
    w, m = W[i], M[i]
    q = greedy_q(w, m, "man", P, 2)
    p = pick(q, w["heading"]); y = Y[i][9]
    if p != y:
        legs = legs_for(w, m)
        st = {ACTIONS[d]: info[0] + ("+s" if info[1] else "") + ("" if info[2] else "+wrongway") for d, info in w["adj"][w["robot"]].items()}
        print(i, "robot", w["robot"], ACTIONS[w["heading"]], "label", ACTIONS[y], "pred", ACTIONS[p], "q", [round(x, 1) for x in q], "targets", legs, st)
