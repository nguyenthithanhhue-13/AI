"""Giả thuyết NGÂN SÁCH ĐI VÒNG: robot chỉ xét các đường có số đoạn <= ngắn nhất + B, và chọn đường có chi phí sở thích
(theta · đặc trưng) nhỏ nhất trong số đó (đặc trưng: số đoạn, đông, mái che, địa điểm đi ngang, rẽ phải / trái / quay đầu).
Quy hoạch động lùi theo số đoạn còn lại trên (chặng, giao lộ, hướng).
    python scratch/p2_budget.py"""
import sys, numpy as np
sys.path.insert(0, "scratch")
from p2_lib import *
from p2_a20_dagperc import bfs_dist

FN = ["hop", "crowd", "cover", "lm", "tR", "tL", "tB"]


def first_q(x, theta, B, legged):
    w = x["w"]; legs = x["legs"]
    V2 = bfs_dist(w, legs[-1], legged)
    V1 = bfs_dist(w, legs[0], legged, {v: V2.get(v) for v in legs[0]}) if len(legs) == 2 else V2
    s, h0 = w["robot"], w["heading"]
    if s not in V1:
        return [INF] * 4
    T = V1[s] + B
    L = {p for v in w["landmarks"].values() for p in v} - {p for Lg in legs for p in Lg}
    nodes = list(w["adj"].keys())
    K = len(legs)
    trans = []      # (n, d, n2, base_cost)
    for n in nodes:
        for d, info in w["adj"][n].items():
            if not edge_ok(info, legged):
                continue
            n2 = (n[0] + DRC[d][0], n[1] + DRC[d][1])
            c = theta[0] + theta[1] * (info[0] == "crowded") + theta[2] * (info[0] == "covered") + theta[3] * (n2 in L)
            trans.append((n, d, n2, c))
    tc = {"S": 0.0, "R": theta[4], "L": theta[5], "B": theta[6]}
    # C[k][(n, h)] với t đoạn còn lại; khởi tạo t = 0
    def terminal(prev, t):
        C = [dict() for _ in range(K)]
        for k in reversed(range(K)):
            for n in legs[k]:
                for h in range(4):
                    v = 0.0 if k == K - 1 else C[k + 1].get((n, h), INF)
                    if v < C[k].get((n, h), INF):
                        C[k][(n, h)] = v
        return C
    C = terminal(None, 0)
    hist = [C]
    for t in range(1, T):
        Cn = [dict() for _ in range(K)]
        for k in range(K):
            for (n, d, n2, c) in trans:
                v2 = hist[-1][k].get((n2, d), INF)
                if v2 == INF:
                    continue
                for h in range(4):
                    v = c + tc[REL[h][d]] + v2
                    if v < Cn[k].get((n, h), INF):
                        Cn[k][(n, h)] = v
        # được phép kết thúc chặng sớm (đã tới đích / ghé)
        for k in reversed(range(K)):
            for n in legs[k]:
                for h in range(4):
                    v = 0.0 if k == K - 1 else Cn[k + 1].get((n, h), INF)
                    if v < Cn[k].get((n, h), INF):
                        Cn[k][(n, h)] = v
        hist.append(Cn)
    q = [INF] * 4
    for d, info in w["adj"].get(s, {}).items():
        if not edge_ok(info, legged):
            continue
        n2 = (s[0] + DRC[d][0], s[1] + DRC[d][1])
        c = theta[0] + theta[1] * (info[0] == "crowded") + theta[2] * (info[0] == "covered") + theta[3] * (n2 in L)
        v = hist[T - 1][0].get((n2, d), INF)
        if v < INF:
            q[d] = c + tc[REL[h0][d]] + v
    return q


if __name__ == "__main__":
    D = load("train")[:300]
    HYPS = {
        "đông trước, rồi số đoạn": [1, 100, 0, 0, 0, 0, 0],
        "đông+địa điểm, rồi số đoạn": [1, 100, 0, 100, 0, 0, 0],
        "mái che nhiều nhất": [1, 0, -100, 0, 0, 0, 0],
        "địa điểm ít nhất": [1, 0, 0, 100, 0, 0, 0],
        "rẽ ít nhất": [1, 0, 0, 0, 100, 100, 100],
        "đông 3 + đp 3": [1, 3, 0, 3, 0, 0, 0],
    }
    for name, th in HYPS.items():
        th = np.array(th, float)
        ins = np.zeros(10); uni = np.zeros(10)
        for x in D:
            for legged in (False, True):
                q = first_q(x, th, 2, legged)
                a = argmins(q, 1e-6)
                for r in ([4] if legged else [1, 2, 3, 5, 6, 7, 8, 9]):
                    ins[r] += x["y"][r] in a; uni[r] += a == [x["y"][r]]
        print(f"{name:28s} trong argmin " + " ".join(f"R{r}={ins[r]/len(D):.2f}" for r in range(1, 10)) +
              " | duy nhất " + " ".join(f"{uni[r]/len(D):.2f}" for r in range(1, 10)), flush=True)
