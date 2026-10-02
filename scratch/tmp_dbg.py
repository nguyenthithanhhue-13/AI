import sys, pickle
sys.path.insert(0, "src")
import numpy as np
from common import *
from cvfeat import *
from cv_infer import *
scenes = load_split("validation")[2]
dets = pickle.load(open(CACHE / "det_validation_dev.pkl", "rb"))
models = Models(OUT / "models_dev")
s = scenes[int(sys.argv[1])]
world, info = analyze(read_rgb(DATA / "validation" / s["image"]), models, dets[s["image"]])
print("pred nodes", len(info["xy"]), "gt", len(s["nodes"]), "theta", info["theta"], "unit", info["unit"])
print("robot", world["robot"], world["heading"], "gt", s["robot"])
gx = np.array([n["xy"] for n in s["nodes"]]); px = np.array(list(info["xy"].values()))
d = np.hypot(gx[:, None, 0] - px[None, :, 0], gx[:, None, 1] - px[None, :, 1]).min(1)
print("missed gt nodes:", [(n["rc"], n["xy"]) for n, dd in zip(s["nodes"], d) if dd > 12][:20])
print({k: info[k] for k in ("legend_how", "n_swatches", "p_rain_icon", "rain_text", "use_color")})
