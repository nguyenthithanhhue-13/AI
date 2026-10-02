import sys, pickle
sys.path.insert(0, "src")
import numpy as np, cv2
from common import *
from cvfeat import *
from cv_infer import *

models = Models(OUT / "models_dev")
scenes = load_split("validation")[2]
dets = pickle.load(open(CACHE / "det_validation_dev.pkl", "rb"))
for si in map(int, sys.argv[1:]):
    s = scenes[si]
    rgb = read_rgb(DATA / "validation" / s["image"]); im = Img(rgb)
    det = dets[s["image"]]
    cand = []
    for p in sorted(det["nodes"] + det["swatches"], key=lambda t: -t[2]):
        if all(abs(p[0] - q[0]) > 14 or abs(p[1] - q[1]) > 14 for q in cand): cand.append(p)
    PK = models.swatch.predict_proba(np.array([im.swatch_crop(p[0], p[1]) for p in cand]))[0]
    gsw = np.array([((l["swatch"][0] + l["swatch"][2]) / 2, (l["swatch"][1] + l["swatch"][3]) / 2) for l in s["legend"]])
    vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    fp = fn = 0
    for p, pk in zip(cand, PK):
        is_leg = np.min(np.hypot(gsw[:, 0] - p[0], gsw[:, 1] - p[1])) < 12
        pred = pk[K_NONE] < 0.5
        fp += pred and not is_leg; fn += is_leg and not pred
        col = (0, 200, 0) if pred else (200, 200, 200)
        cv2.circle(vis, (int(p[0]), int(p[1])), 10, col, 2)
        if pred != is_leg:
            cv2.putText(vis, LEGEND_KINDS[pk.argmax()][:5] + " %.2f" % (1 - pk[K_NONE]), (int(p[0]) + 10, int(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    rows, _ = find_legend_rows(im, models, det)
    for p in rows: cv2.rectangle(vis, (int(p[0]) - 14, int(p[1]) - 14), (int(p[0]) + 14, int(p[1]) + 14), (0, 255, 255), 2)
    print(si, s["style"], "cand", len(cand), "false legend", fp, "missed legend", fn, "group", len(rows), "gt", len(gsw))
    cv2.imwrite(f"scratch/leg_{si}.png", vis)
