"""MỔ mạng chi phí đã học (scratch/p2_vin2.py): chi phí một bước đi thẳng trên đường "thường", vào giao lộ có địa điểm loại k
(so với giao lộ trống), dưới từng tổ hợp điều kiện cảnh -> bảng phạt theo loại địa điểm. Cũng in phạt đông / mái che / rẽ.
    python scratch/p2_vin_probe.py <robot> [tag]"""
import sys, glob, os
r = int(sys.argv[1]); TAG = sys.argv[2] if len(sys.argv) > 2 else ""
sys.argv = [sys.argv[0], str(r), "1", "128" if TAG == "big" else "64"]
import numpy as np, torch
exec(open("scratch/p2_vin2.py", encoding="utf-8").read().split("t0 = time.time()")[0])
nets = []
for f in sorted(glob.glob(f"cache/p2_vin2{TAG}_r{r}_s[0-9].pt")):
    n = Net().to(dev); n.load_state_dict(torch.load(f)); nets.append(n.eval())
PL = ["library", "dorm", "sports", "clinic", "canteen", "parking", "lecture", "lab", "office", "gate"]


def cost(e, C):
    E = torch.tensor(np.array(e, np.float32))[None].to(dev); Ct = torch.tensor(np.array(C, np.float32))[None].to(dev)
    with torch.no_grad():
        return float(np.mean([n.cost(E, Ct).item() for n in nets]))


def step(st="normal", lmk=None, rel="S"):
    e = [st == "crowded", st == "covered", st == "normal", 0, lmk is not None] + [lmk == k for k in range(10)] + [rel == x for x in "SRLB"]
    return e + [0] * (NF - len(e))


def cond(rain=0, night=0, urg=0, frag=0, via=0, mapg=0, goal=None):
    c = [rain, night, urg, frag, via, mapg] + [goal == t for t in PL] + [0] * 10
    return c + [0] * (NC - len(c))


COMBOS = [(n, u, v, f, w) for n in (0, 1) for u in (0, 1) for v in (0, 1) for f in (0, 1) for w in (0, 1)]
print(f"R{r} ({len(nets)} mạng). cột: chi phí bước thường | +đông | +mái che | +rẽ R/L/B | phạt theo loại địa điểm")
print("đêm gấp ghé vỡ mưa | base  crowd cover  tR   tL   tB  | " + " ".join(f"{t[:5]:>5s}" for t in PL))
for n, u, v, f, w in COMBOS:
    C = cond(rain=w, night=n, urg=u, frag=f, via=v, goal="library")
    b = cost(step(), C)
    row = [b, cost(step("crowded"), C) - b, cost(step("covered"), C) - b] + [cost(step(rel=x), C) - b for x in "RLB"]
    lm = [cost(step(lmk=k), C) - b for k in range(10)]
    print(f" {n}   {u}   {v}   {f}   {w}  | " + " ".join(f"{x:5.2f}" for x in row) + " | " + " ".join(f"{x:5.2f}" for x in lm))
