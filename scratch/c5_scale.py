"""Kích thước ký hiệu địa điểm/robot có tỉ lệ với khoảng cách lưới không? Đo bán kính vòng tròn (kiểu sketch/night)."""
import sys
sys.path.insert(0, "src")
import numpy as np, cv2
from common import *
from cvfeat import read_rgb

scenes = load_split("train")[2]
rows = []
for s in scenes[:400]:
    if s["style"] not in ("sketch", "night", "classic"):
        continue
    rc = np.array([n["rc"] for n in s["nodes"]], float); xy = np.array([n["xy"] for n in s["nodes"]], float)
    A = np.c_[rc[:, 1], rc[:, 0], np.ones(len(rc))]
    coef, *_ = np.linalg.lstsq(A, xy, rcond=None)
    sx, sy = np.hypot(*coef[0]), np.hypot(*coef[1])
    img = read_rgb(DATA / "train" / s["image"])
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    pos = {tuple(n["rc"]): n["xy"] for n in s["nodes"]}
    x, y = pos[tuple(s["robot"]["rc"])]
    x0, y0 = int(x) - 45, int(y) - 45
    crop = g[max(y0, 0):y0 + 90, max(x0, 0):x0 + 90]
    c = cv2.HoughCircles(cv2.GaussianBlur(crop, (0, 0), 1.5), cv2.HOUGH_GRADIENT, 1, 50, param1=80, param2=18, minRadius=8, maxRadius=40)
    if c is not None:
        sw = s["legend"][-1]["swatch"]
        rows.append((s["style"], sx, sy, float(c[0, 0, 2]), s["width"], s["height"], sw[2] - sw[0], sw[3] - sw[1]))
for st in ("sketch", "night", "classic"):
    r = np.array([x[1:] for x in rows if x[0] == st])
    mn = np.minimum(r[:, 0], r[:, 1])
    print(st, len(r), "corr(radius, min spacing) %.2f" % np.corrcoef(mn, r[:, 2])[0, 1], "ratio r/min_sp: mean %.3f std %.3f" % ((r[:, 2] / mn).mean(), (r[:, 2] / mn).std()),
          "| corr(radius, swatch h) %.2f" % np.corrcoef(r[:, 6], r[:, 2])[0, 1], "ratio r/swatch_h mean %.3f std %.3f" % ((r[:, 2] / r[:, 6]).mean(), (r[:, 2] / r[:, 6]).std()),
          "| radius range %.0f-%.0f" % (r[:, 2].min(), r[:, 2].max()))
