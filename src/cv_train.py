"""Huấn luyện 5 mô hình CV (MLP numpy) từ cache do cv_data.py tạo.

    python src/cv_train.py dev      # học trên train, đo trên validation  -> outputs/models_dev/
    python src/cv_train.py final    # học trên train + validation          -> outputs/models_final/
"""
import sys
import time

import numpy as np

from common import *
from mlp import MLP

SPECS = {   # tên: (khóa dữ liệu, lớp ẩn, đầu ra, số epoch, lr)
    "swatch": ("sw", [256], [19, 3, 3], 20, 1e-3),
    "weather": ("we", [128], [2], 25, 1e-3),
    "det": ("det", [256, 64], [6], 8, 1e-3),
    "align": ("al", [512, 128], [13, 13], 16, 1e-3),
    "node": ("node", [512, 128], [15], 16, 1e-3),
    "edge": ("edge", [384, 128], [5, 2, 3], 10, 1e-3),
}


def load(split, key):
    if key != "det":
        d = np.load(CACHE / f"cvdata_{split}.npz")
        return d[key + "_x"], d[key + "_y"]
    d = np.load(CACHE / f"cvdet_{split}.npz")
    X, Y = d["det_x"], d["det_y"]
    hp = CACHE / f"cvhard_{split}.npz"
    if hp.exists():          # âm bản khó (cv_hardneg.py): lặp 2 lần để tăng trọng số
        h = np.load(hp)
        print(f"   + {len(h['det_y'])} mẫu khó ({split})")
        X = np.concatenate([X, h["det_x"], h["det_x"]]); Y = np.concatenate([Y, h["det_y"], h["det_y"]])
    return X, Y


def main(mode, only=None):
    out = OUT / f"models_{mode}"
    out.mkdir(exist_ok=True)
    total = 0
    for name, (key, hidden, heads, epochs, lr) in SPECS.items():
        if only and name not in only:
            continue
        X, Y = load("train", key)
        Xv, Yv = load("validation", key)
        if mode == "final":
            X, Y = np.concatenate([X, Xv]), np.concatenate([Y, Yv])
        t = time.time()
        print(f"== {name}: {X.shape} -> hidden {hidden} heads {heads}", flush=True)
        m = MLP(X.shape[1], hidden, heads, seed=1)
        m.fit(X, Y, epochs=epochs, lr=lr, batch=512 if len(X) > 20000 else 128, Xval=Xv, Yval=Yv, name=name)
        m.save(out / f"{name}.npz")
        total += m.n_params
        print(f"   {name}: {m.n_params:,} tham số, {time.time() - t:.0f}s, val acc {m.accuracy(Xv, Yv)}", flush=True)
    print(f"tổng tham số các mô hình vừa huấn luyện: {total:,}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or None)
