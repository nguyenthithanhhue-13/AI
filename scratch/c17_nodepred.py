"""Bộ phân loại giao lộ: vị trí đúng / vị trí bộ dò / sau khi căn tâm. Theo kiểu vẽ và loại giao lộ."""
import sys, pickle
sys.path.insert(0, "src")
import numpy as np
from collections import Counter
from common import *
from cvfeat import *
from mlp import MLP

m = MLP.load(OUT / "models_dev" / "node.npz")
al = MLP.load(OUT / "models_dev" / "align.npz")
scenes = load_split("validation")[2]
cache = pickle.load(open(CACHE / "world_validation_dev.pkl", "rb"))
dets = pickle.load(open(CACHE / "det_validation_dev.pkl", "rb"))
c = Counter(); res = Counter()
for s in scenes:
    im = Img(read_rgb(DATA / "validation" / s["image"]))
    xy = {tuple(n["rc"]): n["xy"] for n in s["nodes"]}
    u = grid_unit(list(xy), list(xy.values()))
    pp = np.array([(p[0], p[1]) for p in dets[s["image"]]["nodes"] if p[2] > 0.5])
    lm = {tuple(l["rc"]): l["type"] for l in s["landmarks"]}
    rob = tuple(s["robot"]["rc"])
    items = []
    for rc, (x, y) in xy.items():
        d = np.hypot(pp[:, 0] - x, pp[:, 1] - y); j = d.argmin()
        if d[j] > 12: continue
        yv = 1 + ACTIONS.index(s["robot"]["heading"]) if rc == rob else (5 + PLACES.index(lm[rc]) if rc in lm else 0)
        items.append((x, y, pp[j, 0], pp[j, 1], yv))
    if not items: continue
    it = np.array(items)
    ax, ay = it[:, 2].copy(), it[:, 3].copy()
    for _ in range(2):
        A = al.predict_proba(np.array([im.node_crop(a, b, u)[3024:] for a, b in zip(ax, ay)]))
        ax -= (A[0].argmax(1) - ALIGN_BINS) * ALIGN_STEP * u; ay -= (A[1].argmax(1) - ALIGN_BINS) * ALIGN_STEP * u
    # biến thể: thử lưới 7x7 vị trí quanh đỉnh, chọn chỗ mô hình căn tâm tin nhất rằng "độ lệch = 0", rồi căn tiếp
    gx_, gy_ = it[:, 2].copy(), it[:, 3].copy()
    nonplain = np.nonzero(it[:, 4] > 0)[0]
    offs = [(dx * 0.03 * u, dy * 0.03 * u) for dx in range(-3, 4) for dy in range(-3, 4)]
    for i in nonplain:
        X = np.array([im.node_crop(it[i, 2] + dx, it[i, 3] + dy, u)[3024:] for dx, dy in offs])
        A = al.predict_proba(X)
        k = int(np.argmax(A[0][:, ALIGN_BINS] * A[1][:, ALIGN_BINS]))
        gx_[i] += offs[k][0]; gy_[i] += offs[k][1]
    for _ in range(2):
        A = al.predict_proba(np.array([im.node_crop(a, b, u)[3024:] for a, b in zip(gx_, gy_)]))
        gx_ -= (A[0].argmax(1) - ALIGN_BINS) * ALIGN_STEP * u; gy_ -= (A[1].argmax(1) - ALIGN_BINS) * ALIGN_STEP * u
    for name, X_, Y_ in (("gt", it[:, 0], it[:, 1]), ("det", ax, ay), ("aligned", ax, ay)):
        if name == "aligned":      # chọn giữa vị trí bộ dò và vị trí đã căn theo độ tin cậy của bộ phân loại
            Pa = m.predict_proba(np.array([im.node_crop(a, b, u) for a, b in zip(ax, ay)]))[0]
            Pd = m.predict_proba(np.array([im.node_crop(a, b, u) for a, b in zip(it[:, 2], it[:, 3])]))[0]
            use_a = Pa.max(1) >= Pd.max(1)
            P = np.where(use_a, Pa.argmax(1), Pd.argmax(1))
        else:
            P = m.predict_proba(np.array([im.node_crop(a, b, u) for a, b in zip(X_, Y_)]))[0].argmax(1)
        for p, yv, gx, gy, a, b in zip(P, it[:, 4], it[:, 0], it[:, 1], X_, Y_):
            g = "plain" if yv == 0 else ("robot" if yv < 5 else "landmark")
            c[(s["style"], g, name)] += p != yv
            res[(s["style"], g, name)] += np.hypot(a - gx, b - gy) > 0.03 * u
            c[(s["style"], g, "n")] += 1
for st in ("classic", "night", "print", "sketch"):
    for g in ("plain", "landmark", "robot"):
        n = c[(st, g, "n")] // 3
        print(f"{st:8s} {g:9s} n={n:5d}  sai: gt {c[(st, g, 'gt')]:3d}  det {c[(st, g, 'det')]:3d}  aligned {c[(st, g, 'aligned')]:3d}   | lệch tâm >3% unit: det {res[(st, g, 'det')]:4d} aligned {res[(st, g, 'aligned')]:4d}")
