"""Độ đúng bộ phân loại giao lộ theo kiểu vẽ (dùng vị trí đúng từ scenes.json)."""
import sys
sys.path.insert(0, "src")
import numpy as np
from collections import Counter
from common import *
from cvfeat import *
from mlp import MLP

m = MLP.load(OUT / "models_dev" / "node.npz")
scenes = load_split("validation")[2]
tot = Counter(); err = Counter(); conf = Counter()
for s in scenes:
    im = Img(read_rgb(DATA / "validation" / s["image"]))
    xy = {tuple(n["rc"]): n["xy"] for n in s["nodes"]}
    u = grid_unit(list(xy), list(xy.values()))
    lm = {tuple(l["rc"]): l["type"] for l in s["landmarks"]}
    rcs = list(lm) + [tuple(s["robot"]["rc"])]
    X = np.array([im.node_crop(xy[rc][0], xy[rc][1], u) for rc in rcs])
    P = m.predict_proba(X)[0].argmax(1)
    for rc, p in zip(rcs, P):
        y = 5 + PLACES.index(lm[rc]) if rc in lm else 1 + ACTIONS.index(s["robot"]["heading"])
        k = (s["style"], "robot" if y < 5 else "landmark")
        tot[k] += 1; err[k] += p != y
        if p != y: conf[(s["style"], NODE_CLASSES[y], NODE_CLASSES[p])] += 1
for k in sorted(tot): print(k, f"{err[k]}/{tot[k]} sai ({1 - err[k] / tot[k]:.4f})")
print(conf.most_common(25))
