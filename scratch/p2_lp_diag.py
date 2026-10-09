"""Soi nhóm khó của một robot bằng LP biên cực đại (scratch/p2_lp.py): trọng số tìm được + các cảnh train sai (đường rẻ nhất của từng
bước theo w). python scratch/p2_lp_diag.py <robot> <key> "<nhóm repr>" [n]"""
import sys
r, KEY, G = sys.argv[1], sys.argv[2], sys.argv[3]; NS = int(sys.argv[4]) if len(sys.argv) > 4 else 8
sys.argv = ["x", r, KEY]
g = {"__name__": "x"}
exec(open("scratch/p2_lp.py", encoding="utf-8").read().split('\nif __name__ == "__main__":\n')[0], g)
import numpy as np
TR = g["prep"]("train")
items = [it for it in TR if str(g["gkey"](it[0])) == G]
w = g["solve"](items, np.r_[1.0, np.zeros(21)])
PLs = ["lib", "dor", "spo", "cli", "can", "par", "lec", "lab", "off", "gat"]
N = ["hop", "đông", "mái", "lm", "bậc", "R", "L", "B", "khôngmái"] + PLs + ["R0", "L0", "B0"]
print("w:", " ".join(f"{N[k]}={w[k]:.2f}" for k in range(22) if w[k] > 1e-6))
n = 0
for x, mv in items:
    y = x["y"][int(r)]; p = g["predict"](mv, w)
    if p == y:
        continue
    n += 1
    if n > NS:
        continue
    print("==", x["s"]["scene_id"], "mũi", "UDLR"[x["w"]["heading"]], "nhãn", "UDLR"[y], "đoán", "UDLR"[p])
    for d, F in mv:
        c = F @ w; j = int(np.argmin(c)); f = F[j]
        print(f"   {'UDLR'[d]} {'NHÃN' if d == y else ('đoán' if d == p else '    ')} chi phí {c[j]:6.2f} | " +
              " ".join(f"{N[k]}={int(f[k])}" for k in range(22) if f[k]))
print("sai", n, "/", len(items))
