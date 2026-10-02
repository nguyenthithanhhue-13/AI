"""Khai thác "âm bản khó" cho bộ dò: chạy bộ dò hiện tại trên ảnh có nhãn, gom những chỗ nó báo nhầm
(và những chỗ nó bỏ sót) rồi thêm vào dữ liệu huấn luyện.

    python src/cv_hardneg.py train dev      # -> cache/cvhard_train.npz
"""
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

from common import *

_M = {}


def _init(mode):
    from mlp import MLP
    _M["det"] = MLP.load(OUT / f"models_{mode}" / "det.npz")


def mine(job):
    from cv_infer import dense_detect, peaks
    from cvfeat import Img, read_rgb, degrade
    split, s, idx, aug = job
    rng = np.random.default_rng(idx * 13 + 5 + (99991 if aug else 0))
    rgb = read_rgb(DATA / split / s["image"])
    if aug:
        rgb = degrade(rgb, rng)
    im = Img(rgb)
    heat, step = dense_detect(im, _M["det"])
    pos = [(n["xy"][0], n["xy"][1]) for n in s["nodes"]]
    lm = {tuple(l["rc"]) for l in s["landmarks"]}
    cls = [3 if tuple(n["rc"]) == tuple(s["robot"]["rc"]) else (2 if tuple(n["rc"]) in lm else 1) for n in s["nodes"]]
    for l in s["legend"]:
        b = l["swatch"]
        pos.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)); cls.append(4)
    wb = s["weather_box"]
    pos.append(((wb[0] + wb[2]) / 2, (wb[1] + wb[3]) / 2)); cls.append(5)
    P = np.array(pos)
    X, Y = [], []
    found = np.zeros(len(pos), bool)
    for score, ok_cls, rad in ((heat[..., 1] + heat[..., 2] + heat[..., 3], (1, 2, 3), 3), (heat[..., 4], (4,), 2), (heat[..., 5], (5,), 4)):
        for (x, y, sc) in peaks(score, step, thr=0.25, rad=rad):
            d = np.hypot(P[:, 0] - x, P[:, 1] - y)
            j = int(np.argmin(d))
            if d[j] <= 9 and cls[j] in ok_cls:
                found[j] |= sc > 0.5
            elif d[j] > 12:
                X.append(im.det_patch(x, y)); Y.append(0)          # báo nhầm
            elif cls[j] not in ok_cls:
                X.append(im.det_patch(P[j, 0], P[j, 1])); Y.append(cls[j])   # nhầm lớp -> thêm dương bản đúng lớp
    for j in np.nonzero(~found)[0]:                                 # bỏ sót -> thêm dương bản
        for _ in range(2):
            dx, dy = rng.integers(-2, 3, 2)
            X.append(im.det_patch(P[j, 0] + dx, P[j, 1] + dy)); Y.append(cls[j])
    return (np.array(X, np.uint8).reshape(-1, 3456), np.array(Y, np.int64), int((~found).sum()), sum(1 for y in Y if y == 0))


if __name__ == "__main__":
    split, mode = sys.argv[1], sys.argv[2]
    scenes = load_split(split)[2]
    jobs = [(split, s, i, False) for i, s in enumerate(scenes)]
    if split == "train":
        jobs += [(split, s, i, True) for i, s in enumerate(scenes) if i % 2 == 0]
    t = time.time()
    with Pool(11, initializer=_init, initargs=(mode,)) as p:
        res = p.map(mine, jobs, chunksize=4)
    X = np.concatenate([r[0] for r in res]); Y = np.concatenate([r[1] for r in res])
    print(f"{split}: {len(jobs)} ảnh, {time.time() - t:.0f}s; bỏ sót {sum(r[2] for r in res)}, báo nhầm {sum(r[3] for r in res)}; thêm {len(Y)} mẫu")
    path = CACHE / f"cvhard_{split}.npz"
    if path.exists() and "--append" in sys.argv:
        old = np.load(path)
        X = np.concatenate([old["det_x"], X]); Y = np.concatenate([old["det_y"], Y])
    np.savez(path, det_x=X, det_y=Y)
