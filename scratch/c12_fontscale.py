import sys
sys.path.insert(0, "src")
import numpy as np
from common import *

scenes = load_split("train")[2]
rows = []
for s in scenes:
    L = s["legend"]
    # chiều cao chữ: lấy dòng "robot" (chuỗi cố định 2 biến thể) -> dùng bề rộng nhãn / số ký tự làm thước đo cỡ chữ
    sw = np.array([l["swatch"] for l in L]); lb = np.array([l["label"] for l in L])
    cy = (sw[:, 1] + sw[:, 3]) / 2
    dy = np.diff(np.sort(cy)); pitch = np.median(dy[dy > 20]) if (dy > 20).any() else np.nan
    cw = np.median([(l["label"][2] - l["label"][0]) / len(l["text"]) for l in L])
    rows.append((s["style"], pitch, sw[0, 3] - sw[0, 1], sw[0, 2] - sw[0, 0], cw, np.median(lb[:, 3] - lb[:, 1]), np.median(lb[:, 0] - sw[:, 2])))
for st in ("classic", "night", "print", "sketch"):
    r = np.array([x[1:] for x in rows if x[0] == st], float)
    r = r[~np.isnan(r[:, 0])]
    print(st, "pitch %.0f-%.0f" % (r[:, 0].min(), r[:, 0].max()), "| char width %.1f-%.1f" % (r[:, 3].min(), r[:, 3].max()),
          "| corr(pitch, charw) %.2f" % np.corrcoef(r[:, 0], r[:, 3])[0, 1], "corr(swatch_h, charw) %.2f" % np.corrcoef(r[:, 1], r[:, 3])[0, 1],
          "corr(swatch_w, charw) %.2f" % np.corrcoef(r[:, 2], r[:, 3])[0, 1],
          "| charw/swatch_h std/mean %.3f" % (np.std(r[:, 3] / r[:, 1]) / np.mean(r[:, 3] / r[:, 1])),
          "charw/pitch %.3f" % (np.std(r[:, 3] / r[:, 0]) / np.mean(r[:, 3] / r[:, 0])), "gap label-swatch %.1f-%.1f" % (r[:, 5].min(), r[:, 5].max()))
