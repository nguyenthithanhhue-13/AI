import sys
sys.path.insert(0, "src")
import numpy as np, cv2
from collections import Counter
from common import *
from cvfeat import *
from mlp import MLP

m = MLP.load(OUT / "models_dev" / "swatch.npz")
scenes = load_split("validation")[2]
tot = Counter(); err = Counter(); conf = Counter(); tiles = []; offs = []
for s in scenes:
    im = Img(read_rgb(DATA / "validation" / s["image"]))
    centers = [((l["swatch"][0] + l["swatch"][2]) / 2, (l["swatch"][1] + l["swatch"][3]) / 2) for l in s["legend"]]
    off = im.label_offset(centers)
    true_off = np.median([l["label"][0] - c[0] for l, c in zip(s["legend"], centers)])
    offs.append(off - true_off)
    X = np.array([im.swatch_crop(c[0], c[1], off) for c in centers])
    PK = m.predict_proba(X)[0].argmax(1)
    for l, p, x in zip(s["legend"], PK, X):
        k = l["kind"][6:] if l["kind"].startswith("place:") else l["kind"]
        y = LEGEND_KINDS.index(k)
        tot[s["style"]] += 1; err[s["style"]] += p != y
        if p != y:
            conf[(s["style"], k, LEGEND_KINDS[p])] += 1
            if len(tiles) < 40: tiles.append(x[2592:].reshape(14, 110))
print({k: f"{err[k]}/{tot[k]}" for k in tot})
offs = np.array(offs)
print("offset error: p5/p50/p95", np.percentile(offs, [5, 50, 95]), "|err|>4:", (np.abs(offs) > 4).sum(), "of", len(offs))
print(conf.most_common(30))
cv2.imwrite("scratch/vis_labels.png", cv2.resize(np.vstack(tiles), None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
