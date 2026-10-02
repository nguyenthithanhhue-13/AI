"""Thử riêng bộ dò + khớp lưới trên validation (chưa cần mô hình giao lộ/đoạn đường)."""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
from multiprocessing import Pool
from common import *

_M = {}


def init():
    from mlp import MLP
    _M["det"] = MLP.load(OUT / "models_dev" / "det.npz")


def run(job):
    from cv_infer import dense_detect, peaks, legend_swatches, fit_lattice
    from cvfeat import Img, read_rgb
    split, s = job
    t = time.time()
    im = Img(read_rgb(DATA / split / s["image"]))
    heat, step = dense_detect(im, _M["det"])
    t1 = time.time() - t
    nodes = peaks(heat[..., 1] + heat[..., 2] + heat[..., 3], step, 0.5, 3)
    sw_all = peaks(heat[..., 4], step, 0.5, 2)
    sw = legend_swatches(sw_all)
    wpk = peaks(heat[..., 5], step, 0.2, 4)
    res = {"img": s["image"], "t": t1, "style": s["style"]}
    # chú giải
    gsw = np.array([((l["swatch"][0] + l["swatch"][2]) / 2, (l["swatch"][1] + l["swatch"][3]) / 2) for l in s["legend"]])
    psw = np.array([(p[0], p[1]) for p in sw]) if sw else np.zeros((0, 2))
    d = np.hypot(gsw[:, None, 0] - psw[None, :, 0], gsw[:, None, 1] - psw[None, :, 1]) if len(psw) else np.full((len(gsw), 1), 99.)
    res["sw_missed"] = int((d.min(1) > 12).sum()); res["sw_extra"] = int((d.min(0) > 12).sum()) if len(psw) else 0
    if sw:
        sx = np.array([p[0] for p in sw]); sy = np.array([p[1] for p in sw])
        box = (sx.min() - 40, sy.min() - 34, sx.max() + 60, sy.max() + 34)
        nodes = [p for p in nodes if not (box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3])]
    wb = s["weather_box"]; wc = ((wb[0] + wb[2]) / 2, (wb[1] + wb[3]) / 2)
    if wpk:
        wx, wy, _ = max(wpk, key=lambda t: t[2])
        res["w_err"] = float(np.hypot(wx - wc[0], wy - wc[1]))
        nodes = [p for p in nodes if np.hypot(p[0] - wx, p[1] - wy) > 45]
    else:
        res["w_err"] = 999.0
    g = np.array([n["xy"] for n in s["nodes"]]); grc = [tuple(n["rc"]) for n in s["nodes"]]
    p = np.array([(q[0], q[1]) for q in nodes])
    d = np.hypot(g[:, None, 0] - p[None, :, 0], g[:, None, 1] - p[None, :, 1])
    res["n_missed"] = int((d.min(1) > 10).sum()); res["n_extra"] = int((d.min(0) > 10).sum())
    res["pos_err"] = float(np.mean(d.min(1)[d.min(1) <= 10]))
    rows, cols, theta, sp = fit_lattice(p)
    res["theta_err"] = abs(theta - s["degradation"]["rotation_deg"])
    # so khớp rc (sau khi trừ gốc)
    r0 = min(r for r, c in grc); c0 = min(c for r, c in grc)
    gmap = {(r - r0, c - c0) for r, c in grc}
    pmap = {(int(r), int(c)) for r, c in zip(rows, cols)}
    res["rc_ok"] = gmap == pmap
    return res


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    scenes = load_split("validation")[2][:n]
    t = time.time()
    with Pool(8, initializer=init) as pool:
        R = pool.map(run, [("validation", s) for s in scenes])
    print("total", time.time() - t, "s; per-image detect", np.mean([r["t"] for r in R]))
    for k in ("sw_missed", "sw_extra", "n_missed", "n_extra"):
        print(k, "scenes with >0:", sum(r[k] > 0 for r in R), "total", sum(r[k] for r in R))
    print("weather err >20px:", sum(r["w_err"] > 20 for r in R), "mean node pos err", np.mean([r["pos_err"] for r in R]), "theta err max", max(r["theta_err"] for r in R))
    print("rc exact:", sum(r["rc_ok"] for r in R), "/", len(R))
    for r in R:
        if not r["rc_ok"] or r["sw_missed"] or r["sw_extra"]:
            print("  ", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
