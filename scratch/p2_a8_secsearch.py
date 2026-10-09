"""Dò chi phí PHỤ dạng tổng có trọng số (đông, mái che, bậc thang, rẽ phải / trái / quay đầu, có tính bước rẽ đầu hay không)
xếp sau chi phí chính (thứ tự từ điển), rồi thứ tự hướng cho hòa tuyệt đối. Tối ưu từng tọa độ trên các cảnh có hòa.
    python scratch/p2_a8_secsearch.py <robot> [primary json]"""
import sys, collections, itertools, json, time
sys.path.insert(0, "scratch")
from p2_lib import *

r = int(sys.argv[1])
prim = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {"e": {}, "t": {}}
D = load("train")
leg = r == 4
e0 = mk_ecost(**prim["e"]); t0 = mk_tcost(**prim["t"])
BIG = 1000.0
# chỉ giữ cảnh có hòa theo chi phí chính
T = []
for x in D:
    q = qvals(x["w"], x["legs"], e0, t0, leg)
    a = argmins(q)
    if len(a) > 1 and x["y"][r] in a:
        T.append(x)
print("số cảnh hòa:", len(T), flush=True)
KEYS = ["crowd", "cover", "normal", "stairs", "tR", "tL", "tB", "f0"]
GRID = [-2, -1, -0.5, -0.2, 0, 0.2, 0.5, 1, 2]


def score(p):
    def ec(i, n, d):
        st = i[0]
        return BIG * e0(i, n, d) + (p["crowd"] if st == "crowded" else p["cover"] if st == "covered" else p["normal"]) + (p["stairs"] if i[1] else 0)
    tt = {"S": 0.0, "R": p["tR"], "L": p["tL"], "B": p["tB"]}
    tc = lambda h, d: BIG * t0(h, d) + tt[REL[h][d]]
    ok = 0; amb = 0
    for x in T:
        # bước rẽ đầu: nhân f0 cho phần phụ (phần chính giữ nguyên)
        w = x["w"]
        q = qvals(w, x["legs"], ec, lambda h, d: tc(h, d), leg, first_turn=1.0)
        if p["f0"] != 1:
            h0 = w["heading"]
            q = [v - (1 - p["f0"]) * tt[REL[h0][d]] if v < INF else v for d, v in enumerate(q)]
        a = argmins(q, 1e-6)
        ok += a == [x["y"][r]]
        amb += len(a) > 1
    return ok, amb


p = {k: 0.0 for k in KEYS}; p["f0"] = 1.0
best = score(p)
print("bắt đầu", best, flush=True)
t = time.time()
for rnd in range(3):
    changed = False
    for k in KEYS:
        grid = [0.0, 1.0] if k == "f0" else GRID
        for v in grid:
            if v == p[k]:
                continue
            q = dict(p, **{k: v})
            s = score(q)
            if s[0] > best[0]:
                best, p, changed = s, q, True
                print(f"  vòng {rnd} {k}={v}: đúng duy nhất {s[0]}/{len(T)}  còn hòa {s[1]}  ({time.time()-t:.0f}s)", flush=True)
    if not changed:
        break
print("KẾT QUẢ", p, best, len(T))
