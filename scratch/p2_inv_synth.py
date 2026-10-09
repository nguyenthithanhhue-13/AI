"""Kiểm tra bộ học ngược bằng nhãn GIẢ sinh từ theta biết trước."""
import sys, numpy as np
sys.path.insert(0, "scratch")
import p2_inv
from p2_inv import G, FN, NF, run
from p2_lib import load
D = load("train")[:1000]
true = np.zeros(NF); true[0] = 1
vals = {"crowd": 3.0, "cover": -0.4, "tR": 0.3, "tL": 0.8, "tB": 3.0}
for k, v in vals.items(): true[FN.index(k)] = v
rng = np.random.default_rng(1)
for t in p2_inv.TYPES: true[FN.index("lm_" + t)] = float(rng.choice([0.5, 1, 2, 4]))
for x in D:
    g = G(x, False)
    res = g.solve(true)
    y = min(res, key=lambda d: res[d][0]) if res else 0
    x["y"] = [y] * 10
acc, th = run(1, "all", 12, lr=0.02, D=D, verbose=False)
print("độ đúng trên nhãn giả:", acc)
print("thật :", " ".join(f"{n}={v:.2f}" for n, v in zip(FN, true)))
print("học  :", " ".join(f"{n}={v:.2f}" for n, v in zip(FN, th)))
