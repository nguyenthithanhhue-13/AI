"""Màu của ký hiệu địa điểm trên bản đồ có khớp với mẫu trong chú giải của chính ảnh đó không? (theo kiểu vẽ)"""
import sys
sys.path.insert(0, "src")
import numpy as np
from collections import Counter, defaultdict
from common import *
from cvfeat import read_rgb, grid_unit


def ring(img, x, y, r0, r1):
    H, W = img.shape[:2]
    ys, xs = np.mgrid[int(y - r1):int(y + r1) + 1, int(x - r1):int(x + r1) + 1]
    d = np.hypot(xs - x, ys - y)
    m = (d >= r0) & (d <= r1) & (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
    return np.median(img[ys[m], xs[m]].astype(float), axis=0)


for split in ["validation"]:
    scenes = load_split(split)[2]
    stat = defaultdict(Counter)
    fixed = defaultdict(lambda: defaultdict(list))
    for s in scenes[:200]:
        img = read_rgb(DATA / split / s["image"])
        xy = {tuple(n["rc"]): n["xy"] for n in s["nodes"]}
        u = grid_unit(list(xy), list(xy.values()))
        leg = {}
        for l in s["legend"]:
            if l["kind"].startswith("place:"):
                b = l["swatch"]
                leg[l["kind"][6:]] = ring(img, (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, 9, 12)
        for lm in s["landmarks"]:
            x, y = xy[tuple(lm["rc"])]
            c = ring(img, x, y, 0.115 * u, 0.15 * u)
            fixed[s["style"]][lm["type"]].append(c)
            d = {t: np.abs(c - v).sum() for t, v in leg.items()}
            best = min(d, key=d.get)
            srt = sorted(d.values())
            stat[s["style"]]["ok" if best == lm["type"] else "wrong"] += 1
            stat[s["style"]]["margin<25"] += (srt[1] - srt[0]) < 25
    for st, c in stat.items():
        print(st, dict(c))
        # màu có cố định theo loại trên toàn bộ dữ liệu không?
        print("   std màu theo loại:", {t: int(np.std(np.array(v), axis=0).mean()) for t, v in fixed[st].items()})
