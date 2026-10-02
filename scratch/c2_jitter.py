"""Độ lệch của giao lộ so với lưới lý tưởng (khớp affine theo hàng/cột)."""
import sys
sys.path.insert(0, "src")
import numpy as np
from common import *

for sp in ["train", "validation"]:
    scenes = load_split(sp)[2]
    res = []; rel = []; sx = []; sy = []; ang = []; swp = []
    for s in scenes:
        rc = np.array([n["rc"] for n in s["nodes"]], float); xy = np.array([n["xy"] for n in s["nodes"]], float)
        A = np.c_[rc[:, 1], rc[:, 0], np.ones(len(rc))]
        coef, *_ = np.linalg.lstsq(A, xy, rcond=None)
        r = xy - A @ coef
        dx = np.hypot(*coef[0]); dy = np.hypot(*coef[1])
        res.append(np.abs(r).max()); rel.append(np.abs(r).max() / min(dx, dy)); sx.append(dx); sy.append(dy)
        ang.append(np.degrees(np.arctan2(coef[0][1], coef[0][0])) - s["degradation"]["rotation_deg"])
        swp.append((s["legend"][0]["swatch"][2] - s["legend"][0]["swatch"][0]) / dx)
    res, rel = np.array(res), np.array(rel)
    print(sp, "max residual px: mean %.1f p99 %.1f max %.1f | relative to spacing: mean %.3f p99 %.3f max %.3f" % (res.mean(), np.percentile(res, 99), res.max(), rel.mean(), np.percentile(rel, 99), rel.max()))
    print("   spacing x %.0f-%.0f y %.0f-%.0f; ratio dx/dy %.2f-%.2f; angle-rot diff %.2f..%.2f" % (min(sx), max(sx), min(sy), max(sy), min(np.array(sx) / np.array(sy)), max(np.array(sx) / np.array(sy)), min(ang), max(ang)))
