"""Chạy CV trên cả một split (song song theo ảnh), lưu cache theo ảnh:
    cache/det_{split}_{mode}.pkl     kết quả bộ dò (bước chậm nhất, ~6 giây/ảnh/nhân CPU)
    cache/world_{split}_{mode}.pkl   world cuối cùng

    python src/run_cv.py validation dev
"""
import os
import pickle
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from common import *

_M = {}


def _init(mode):
    from cv_infer import Models
    _M["m"] = Models(OUT / f"models_{mode}")


def _run(job):
    from cv_infer import analyze, detect
    from cvfeat import read_rgb, Img
    split, image, det = job
    try:
        rgb = read_rgb(DATA / split / image)
        if det is None:
            det = detect(Img(rgb), _M["m"])
        world, info = analyze(rgb, _M["m"], det)
        return image, world, info, None, det
    except Exception:      # ảnh lỗi không được làm hỏng cả lượt chạy
        import traceback
        return image, None, None, traceback.format_exc(), det


def scene_images(split):
    rows = load_json(DATA / split / "observations.json")
    return [rows[i]["image"] for i in range(0, len(rows), 10)]


def run(split, mode, procs=None):
    procs = procs or int(os.environ.get("PROCS", "11"))     # máy thiếu bộ nhớ ảo: PROCS=6
    images = scene_images(split)
    det_path = CACHE / f"det_{split}_{mode}.pkl"
    dets = pickle.load(open(det_path, "rb")) if det_path.exists() else {}
    t = time.time()
    with Pool(procs, initializer=_init, initargs=(mode,)) as p:
        res = p.map(_run, [(split, im, dets.get(im)) for im in images], chunksize=4)
    out = {im: {"world": w, "info": info, "error": err} for im, w, info, err, _ in res}
    new_dets = {im: d for im, _, _, _, d in res if d is not None}
    if len(new_dets) > len(dets):
        pickle.dump(new_dets, open(det_path, "wb"))
    nerr = sum(1 for v in out.values() if v["error"])
    print(f"{split}: {len(images)} ảnh trong {time.time() - t:.0f}s, lỗi: {nerr}")
    for im, v in out.items():
        if v["error"]:
            print(im, v["error"][-500:])
            break
    with open(CACHE / f"world_{split}_{mode}.pkl", "wb") as f:
        pickle.dump(out, f)
    return out


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
