"""Phân loại biểu tượng thời tiết KHÔ (nắng / nắng nhẹ / nhiều mây) từ vùng weather_box; nhãn từ chú giải (khi ảnh ghi rõ)."""
import sys, pickle, numpy as np
sys.path.insert(0, "src")
from common import *
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
LAB = {"Thời tiết: nắng": 0, "Thời tiết: nắng nhẹ": 1, "Thời tiết: nhiều mây": 2}
def feat(path, box):
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    im = Image.open(path).convert("RGB").crop((x0, y0, x1, y1)).resize((20, 20))
    a = np.asarray(im).astype(float) / 255
    g = a.mean(2)
    hsv = np.asarray(im.convert("HSV")).astype(float) / 255
    hh = np.histogram(hsv[..., 0], bins=12, range=(0, 1), weights=hsv[..., 1])[0]
    return np.concatenate([g.ravel(), hh / (hh.sum() + 1e-9), [a[..., 0].mean() - a[..., 2].mean(), hsv[..., 1].mean()]])
X, y, F = [], [], {}
for sp in ("train", "validation"):
    for s in load_split(sp)[2]:
        if s["weather"] != "dry": continue
        f = feat(DATA / sp / s["image"], s["weather_box"]); F[(sp, s["scene_id"])] = (f, s["style"])
        t = [l["text"] for l in s["legend"] if l["kind"] == "weather"]
        if t and t[0] in LAB: X.append(f); y.append(LAB[t[0]])
X, y = np.array(X), np.array(y)
clf = LogisticRegression(C=3, max_iter=5000)
print("số ảnh có nhãn", len(y), np.bincount(y), "độ đúng kiểm định chéo", cross_val_score(clf, X, y, cv=5).mean().round(3))
clf.fit(X, y)
pred = {k: int(clf.predict([f])[0]) for k, (f, st) in F.items()}
pickle.dump(pred, open(CACHE / "p2_wicon3.pkl", "wb"))
import collections; print(collections.Counter(pred.values()))
