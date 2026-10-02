import sys, pickle
sys.path.insert(0, "src")
import numpy as np
from collections import defaultdict
from common import *

scenes = load_split("validation")[2]
cache = pickle.load(open(CACHE / "world_validation_dev.pkl", "rb"))
err = defaultdict(list)
for s in scenes:
    info = cache[s["image"]]["info"]
    pp = np.array(list(info["xy"].values()))
    lm = {tuple(l["rc"]) for l in s["landmarks"]}
    for n in s["nodes"]:
        d = np.hypot(pp[:, 0] - n["xy"][0], pp[:, 1] - n["xy"][1])
        j = d.argmin()
        if d[j] < 25:
            kind = "robot" if tuple(n["rc"]) == tuple(s["robot"]["rc"]) else ("landmark" if tuple(n["rc"]) in lm else "plain")
            err[(s["style"], kind)].append((pp[j, 0] - n["xy"][0], pp[j, 1] - n["xy"][1]))
for k in sorted(err):
    e = np.array(err[k]); d = np.hypot(e[:, 0], e[:, 1])
    print(k, "n", len(e), "mean |d| %.1f  p95 %.1f  max %.1f | bias dx %.1f dy %.1f | >6px: %d" % (d.mean(), np.percentile(d, 95), d.max(), e[:, 0].mean(), e[:, 1].mean(), (d > 6).sum()))
