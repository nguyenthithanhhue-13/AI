import sys, json
sys.path.insert(0, "src")
from collections import Counter
from common import *

for sp in ["train", "validation"]:
    scenes = load_split(sp)[2]
    c = Counter(); ex = {}
    for i, s in enumerate(scenes):
        W, H = s["width"], s["height"]
        x0 = min(l["swatch"][0] for l in s["legend"]) / W; x1 = max(l["label"][2] for l in s["legend"]) / W
        y0 = min(l["swatch"][1] for l in s["legend"]) / H; y1 = max(l["swatch"][3] for l in s["legend"]) / H
        nx0 = min(n["xy"][0] for n in s["nodes"]) / W; nx1 = max(n["xy"][0] for n in s["nodes"]) / W
        ny0 = min(n["xy"][1] for n in s["nodes"]) / H; ny1 = max(n["xy"][1] for n in s["nodes"]) / H
        pos = "right" if x0 > nx1 else ("left" if x1 < nx0 else ("bottom" if y0 > ny1 else ("top" if y1 < ny0 else "overlap")))
        # số cột của chú giải
        cols = len({round(l["swatch"][0] / 20) for l in s["legend"]})
        wb = s["weather_box"]
        wpos = ("L" if wb[0] / W < 0.3 else ("R" if wb[0] / W > 0.6 else "M")) + ("T" if wb[1] / H < 0.3 else ("B" if wb[1] / H > 0.6 else "M"))
        k = (pos, cols, wpos)
        c[k] += 1
        ex.setdefault(k, (i, s["image"], s["style"], (W, H)))
    print(sp)
    for k, v in sorted(c.items(), key=lambda t: -t[1]):
        print("  ", k, v, ex[k])
    print("  rows per legend:", Counter(len(s["legend"]) for s in scenes))
