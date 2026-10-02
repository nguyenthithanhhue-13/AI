"""Vì sao tập giao lộ dự đoán khác thật? In khác biệt và vẽ ảnh cho vài cảnh."""
import sys, pickle
sys.path.insert(0, "src")
import numpy as np, cv2
from collections import Counter
from common import *
from cvfeat import read_rgb

scenes = load_split("validation")[2]
cache = pickle.load(open(CACHE / "world_validation_dev.pkl", "rb"))
kinds = Counter(); shown = 0
want = set(map(int, sys.argv[1:]))
for si, s in enumerate(scenes):
    it = cache[s["image"]]
    if it["world"] is None:
        kinds["error"] += 1
        if kinds["error"] <= 2: print(si, it["error"][-600:])
        continue
    pxy = it["info"]["xy"]
    g = {tuple(n["rc"]): n["xy"] for n in s["nodes"]}
    gp = np.array(list(g.values())); pp = np.array(list(pxy.values()))
    d = np.hypot(gp[:, None, 0] - pp[None, :, 0], gp[:, None, 1] - pp[None, :, 1])
    missed = int((d.min(1) > 12).sum()); extra = int((d.min(0) > 12).sum())
    # so rc sau khi khớp theo vị trí: lấy độ lệch chỉ số phổ biến nhất
    off = Counter()
    grc = list(g.keys()); prc = list(pxy.keys())
    for i in range(len(grc)):
        j = int(np.argmin(d[i]))
        if d[i, j] <= 12: off[(prc[j][0] - grc[i][0], prc[j][1] - grc[i][1])] += 1
    consistent = len(off) == 1
    key = ("missed" if missed else "") + ("+extra" if extra else "") + ("" if consistent else "+rc_inconsistent")
    kinds[key or "ok"] += 1
    if (key and shown < 0) or si in want:
        shown += 1
        print(si, s["image"], s["style"], "missed", missed, "extra", extra, "offsets", dict(off), s["degradation"], "grid", s["grid"])
        vis = cv2.cvtColor(read_rgb(DATA / "validation" / s["image"]), cv2.COLOR_RGB2BGR)
        for rc, (x, y) in pxy.items():
            cv2.circle(vis, (int(x), int(y)), 10, (255, 0, 255), 2)
            cv2.putText(vis, f"{rc[0]},{rc[1]}", (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        for rc, (x, y) in g.items():
            cv2.circle(vis, (int(x), int(y)), 4, (0, 200, 0), -1)
        cv2.imwrite(f"scratch/nodes_{si}.png", vis)
print(kinds)
