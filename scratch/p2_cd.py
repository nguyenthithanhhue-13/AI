"""Dò tham số chi phí cho MỘT robot bằng tối ưu từng tọa độ (dải rộng), mục tiêu = số cảnh argmin duy nhất đúng
+ 0,5 × số cảnh nhãn nằm trong tập hòa. Ghi kết quả vào cache/p2_cd_r<robot>_<cond>.txt
    python scratch/p2_cd.py <robot> [cond] [n_scenes]"""
import sys, os, json, time, numpy as np
sys.path.insert(0, "scratch")
from p2_fast import *

import pickle
WI = pickle.load(open("cache/p2_wicon_train.pkl", "rb"))
r = int(sys.argv[1]); cond = sys.argv[2] if len(sys.argv) > 2 else "all"; N = int(sys.argv[3]) if len(sys.argv) > 3 else 700
C = {"all": lambda x: True, "rain": lambda x: x["w"]["rain"], "dry": lambda x: not x["w"]["rain"],
     "urg": lambda x: x["m"]["urgent"], "nourg": lambda x: not x["m"]["urgent"],
     "frag": lambda x: x["m"]["fragile"], "nofrag": lambda x: not x["m"]["fragile"],
     "sun": lambda x: WI.get(x["s"]["scene_id"]) == "sun", "cloud": lambda x: WI.get(x["s"]["scene_id"]) == "cloud",
     "classic": lambda x: x["s"]["style"] == "classic", "night": lambda x: x["s"]["style"] == "night",
     "print": lambda x: x["s"]["style"] == "print", "sketch": lambda x: x["s"]["style"] == "sketch"}[cond]
D = [x for x in load("train") if C(x)][:N]
G = [SG(x["w"], r == 4, lm_excl(x)) for x in D]
Y = [x["y"][r] for x in D]
GRID = {"crowd": [0, 0.5, 1, 2, 3, 5, 8, 12, 20], "cover": [-0.9, -0.6, -0.3, 0, 0.5, 1, 2, 5], "lm": [0, 0.5, 1, 2, 3, 5, 8, 12, 20],
        "tR": [0, 0.25, 0.5, 1, 2, 4], "tL": [0, 0.25, 0.5, 1, 2, 4], "tB": [0, 0.5, 1, 2, 4, 8], "stairs": [-0.5, 0, 1, 3, 10]}
KEYS = ["crowd", "lm", "cover", "tR", "tL", "tB"] + (["stairs"] if r == 4 else [])


def score(p):
    th = theta_of(**p)
    u = 0; ins = 0
    for g, x, y in zip(G, D, Y):
        a = argmins(g.q(th, x["legs"]), 1e-6)
        u += a == [y]; ins += y in a
    return u + 0.5 * (ins - u), u, ins


p = {k: 0.0 for k in KEYS}
best = score(p)
t = time.time()
for rnd in range(4):
    changed = False
    for k in KEYS:
        for v in GRID[k]:
            if v == p[k]:
                continue
            q = dict(p, **{k: v})
            s = score(q)
            if s[0] > best[0]:
                best, p, changed = s, q, True
    print(f"R{r} {cond} vòng {rnd}: {p}  điểm {best[0]:.1f}  duy nhất đúng {best[1]}/{len(D)}  trong argmin {best[2]}/{len(D)}  ({time.time()-t:.0f}s)", flush=True)
    if not changed:
        break
open(f"cache/p2_cd_r{r}_{cond}{os.environ.get('TAG', '')}.txt", "w").write(json.dumps({"p": p, "uni": best[1], "ins": best[2], "n": len(D)}))
