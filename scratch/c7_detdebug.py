import os, sys, time
sys.path.insert(0, "src")
import numpy as np, cv2
from common import *
from mlp import MLP
from cv_infer import dense_detect, peaks, legend_swatches
from cvfeat import Img, read_rgb

det = MLP.load(OUT / "models_dev" / "det.npz")
scenes = load_split("validation")[2]
for i in map(int, sys.argv[1:]):
    s = scenes[i]
    rgb = read_rgb(DATA / "validation" / s["image"])
    im = Img(rgb)
    t = time.time()
    heat, step = dense_detect(im, det)
    print(i, s["image"], s["style"], rgb.shape, "detect %.2fs" % (time.time() - t), "heat", heat.shape, "class mass", heat.reshape(-1, 6).mean(0).round(3))
    nodes = peaks(heat[..., 1] + heat[..., 2] + heat[..., 3], step, 0.5, 3)
    sw_all = peaks(heat[..., 4], step, 0.5, 2)
    sw = legend_swatches(sw_all)
    print("  nodes", len(nodes), "gt", len(s["nodes"]), "| swatches raw", len(sw_all), "kept", len(sw), "gt", len(s["legend"]))
    vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    for x, y, sc in nodes: cv2.circle(vis, (int(x), int(y)), 9, (255, 0, 255), 2)
    for x, y, sc in sw_all: cv2.rectangle(vis, (int(x) - 8, int(y) - 8), (int(x) + 8, int(y) + 8), (0, 160, 0), 2)
    for x, y, sc in sw: cv2.rectangle(vis, (int(x) - 12, int(y) - 12), (int(x) + 12, int(y) + 12), (0, 255, 255), 2)
    cv2.imwrite(f"scratch/det_{i}.png", vis)
