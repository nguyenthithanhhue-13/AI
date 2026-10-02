import sys, pickle
sys.path.insert(0, "src")
import numpy as np
from common import *
from cvfeat import grid_unit

scenes = load_split("validation")[2]
cache = pickle.load(open(CACHE / "world_validation_dev.pkl", "rb"))
dets = pickle.load(open(CACHE / "det_validation_dev.pkl", "rb"))
r = []
for si, s in enumerate(scenes):
    info = cache[s["image"]]["info"]
    xy = {tuple(n["rc"]): n["xy"] for n in s["nodes"]}
    gu = grid_unit(list(xy), list(xy.values()))
    r.append(info["unit"] / gu)
    wb = s["weather_box"]; wc = ((wb[0] + wb[2]) / 2, (wb[1] + wb[3]) / 2)
    wp = sorted(dets[s["image"]]["weather"], key=lambda t: -t[2])
    if si in (77, 88, 101, 104, 116, 132, 174, 206, 211, 229, 296):
        print(si, s["style"], "gt weather center", [round(v) for v in wc], "peaks", [(round(p[0]), round(p[1]), round(p[2], 2)) for p in wp[:4]])
r = np.array(r)
print("unit ratio: p1 %.3f p50 %.3f p99 %.3f; |err|>5%%: %d" % (np.percentile(r, 1), np.median(r), np.percentile(r, 99), (np.abs(r - 1) > 0.05).sum()))
