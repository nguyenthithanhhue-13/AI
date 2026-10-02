import sys, pickle
sys.path.insert(0, "src")
import numpy as np, cv2
from collections import Counter
from common import *
from cvfeat import *
from mlp import MLP

al = MLP.load(OUT / "models_dev" / "align.npz")
scenes = load_split("validation")[2]
dets = pickle.load(open(CACHE / "det_validation_dev.pkl", "rb"))
c = Counter(); tiles = []
for s in scenes:
    if s["style"] != "print": continue
    im = Img(read_rgb(DATA / "validation" / s["image"]))
    xy = {tuple(n["rc"]): n["xy"] for n in s["nodes"]}
    u = grid_unit(list(xy), list(xy.values()))
    pp = np.array([(p[0], p[1]) for p in dets[s["image"]]["nodes"] if p[2] > 0.5])
    for l in s["landmarks"]:
        x, y = xy[tuple(l["rc"])]
        d = np.hypot(pp[:, 0] - x, pp[:, 1] - y); j = d.argmin()
        if d[j] > 14: c["undetected"] += 1; continue
        a, b = pp[j]
        d0 = (a - x, b - y)
        for _ in range(3):
            A = al.predict_proba(im.node_crop(a, b, u)[3024:][None])
            a -= (A[0].argmax() - ALIGN_BINS) * ALIGN_STEP * u; b -= (A[1].argmax() - ALIGN_BINS) * ALIGN_STEP * u
        bad = np.hypot(a - x, b - y) > 0.03 * u
        c[(l["type"], bool(bad))] += 1
        if bad:
            print(l["type"], "unit %.0f" % u, "det err (%.1f, %.1f) -> after align (%.1f, %.1f)" % (d0[0], d0[1], a - x, b - y), s["degradation"])
            if len(tiles) < 12: tiles.append(im.node_crop(x, y, u)[3024:].reshape(56, 72))
print(sorted(c.items(), key=str))
if tiles: cv2.imwrite("scratch/vis_printbad.png", cv2.resize(np.hstack(tiles), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
