import sys
sys.path.insert(0, "src")
import numpy as np, cv2
from common import *
from cvfeat import *

scenes = load_split("validation")[2]
tiles = {}
rng = np.random.default_rng(0)
for s in scenes[:120]:
    im = Img(read_rgb(DATA / "validation" / s["image"]))
    centers = [((l["swatch"][0] + l["swatch"][2]) / 2, (l["swatch"][1] + l["swatch"][3]) / 2) for l in s["legend"]]
    off = im.label_offset(centers)
    for l, c in zip(s["legend"], centers):
        if l["kind"] in ("normal", "crowded", "place:canteen", "robot", "place:lab"):
            tiles.setdefault(l["kind"], [])
            if len(tiles[l["kind"]]) < 9:
                tiles[l["kind"]].append(im.label_patch(c[0] + off, c[1]))
out = np.vstack([np.vstack(v) for v in tiles.values()])
cv2.imwrite("scratch/vis_labels.png", cv2.resize(out, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
