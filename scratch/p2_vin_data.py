"""Dữ liệu cho BỘ TÌM ĐƯỜNG KHẢ VI (học hàm chi phí bằng mạng): mỗi cảnh -> mảng cố định.
  trạng thái s = giao_lộ * 4 + hướng_vừa_đi (giao lộ = hàng * 9 + cột, tối đa 81), s = 324 là trạng thái "chết".
  nxt[s, d]  : trạng thái kế khi đi hướng d (324 nếu không đi được; bản thường và bản có chân)
  E[s, d, :] : đặc trưng bước chuyển = [đông, mái che, thường, bậc thang, vào địa điểm (trừ đích / ghé), loại địa điểm (10),
               rẽ S/R/L/B so với hướng vừa đi]
  goal / via : mặt nạ giao lộ đích / điểm ghé; start = robot * 4 + hướng mũi; C = điều kiện cảnh; y = nhãn 10 robot.
    python scratch/p2_vin_data.py train|validation"""
import sys, numpy as np
sys.path.insert(0, "scratch")
from p2_lib import *

PL = PLACES
NF = 4 + 1 + 10 + 4


def encode(x, legged):
    w = x["w"]
    tg = {p for L in x["legs"] for p in L}
    lt = {p: PL.index(t) for t, v in w["landmarks"].items() for p in v if p not in tg}
    nxt = np.full((325, 4), 324, np.int16)
    E = np.zeros((325, 4, NF), np.uint8)
    for n, dd in w["adj"].items():
        if not (0 <= n[0] < 9 and 0 <= n[1] < 9):
            continue
        for d, info in dd.items():
            if not edge_ok(info, legged):
                continue
            n2 = (n[0] + DRC[d][0], n[1] + DRC[d][1])
            if not (0 <= n2[0] < 9 and 0 <= n2[1] < 9):
                continue
            st = info[0]
            base = [st == "crowded", st == "covered", st == "normal", bool(info[1]), n2 in lt] + [lt.get(n2) == k for k in range(10)]
            for h in range(4):
                s = (n[0] * 9 + n[1]) * 4 + h
                nxt[s, d] = (n2[0] * 9 + n2[1]) * 4 + d
                rel = REL[h][d]
                E[s, d] = base + [rel == "S", rel == "R", rel == "L", rel == "B"]
    return nxt, E


def cond(x):
    m = x["m"]; s = x["s"]
    mapg = bool(m["goal_ref"]) and m["goal_ref"]["kind"] in ("anchor_near", "north_most", "south_most", "west_most", "east_most")
    return np.array([x["w"]["rain"], s["style"] == "night", m["urgent"], m["fragile"], bool(m["via"]), mapg] +
                    [m["goal"] == t for t in PL] + [m["via"] == t for t in PL], np.float32)


if __name__ == "__main__":
    split = sys.argv[1]
    D = load(split)
    N = len(D)
    NX = np.zeros((N, 325, 4), np.int16); NXL = np.zeros((N, 325, 4), np.int16)
    EE = np.zeros((N, 325, 4, NF), np.uint8); EL = np.zeros((N, 325, 4, NF), np.uint8)
    G = np.zeros((N, 81), bool); V = np.zeros((N, 81), bool); ST = np.zeros(N, np.int16); C = np.zeros((N, 26), np.float32)
    Y = np.zeros((N, 10), np.int8)
    for i, x in enumerate(D):
        NX[i], EE[i] = encode(x, False); NXL[i], EL[i] = encode(x, True)
        for p in x["legs"][-1]:
            G[i, p[0] * 9 + p[1]] = True
        if len(x["legs"]) == 2:
            for p in x["legs"][0]:
                V[i, p[0] * 9 + p[1]] = True
        rb = x["w"]["robot"]
        ST[i] = (rb[0] * 9 + rb[1]) * 4 + x["w"]["heading"]
        C[i] = cond(x); Y[i] = x["y"]
    np.savez_compressed(f"cache/p2_vin_{split}.npz", nxt=NX, nxtl=NXL, E=EE, EL=EL, goal=G, via=V, start=ST, C=C, y=Y)
    print(split, N, "xong")
