"""Soi lỗi từng khâu CV trên validation từ cache world."""
import sys, pickle
sys.path.insert(0, "src")
import numpy as np
from collections import Counter
from common import *
from eval_cv import shift_world

scenes = load_split("validation")[2]
cache = pickle.load(open(CACHE / "world_validation_dev.pkl", "rb"))
what = sys.argv[1]
c = Counter()
for si, s in enumerate(scenes):
    it = cache[s["image"]]; info = it["info"]; w = shift_world(it["world"]); g = shift_world(world_from_scene(s))
    if what == "legend":
        gl = {v: k for k, v in s["road_look"].items()}
        c[(info["legend_how"], info["legend_agree"], gl == info["look2status"])] += 1
        if gl != info["look2status"]:
            print(si, s["image"], s["style"], "how", info["legend_how"], "conf %.2f" % info["legend_conf"], "agree", info["legend_agree"],
                  "n_sw", info["n_swatches"], "gt", len(s["legend"]), "pred", info["look2status"], "true", gl)
    if what == "rain":
        ok = g["rain"] == w["rain"]
        c[(s["style"], ok)] += 1
        c[("icon_ok", (info["p_rain_icon"] > 0.5) == g["rain"])] += 1
        c[("text", info["rain_text"], g["rain"])] += 1
        if not ok:
            print(si, s["image"], s["style"], "true rain", g["rain"], "icon p %.2f" % info["p_rain_icon"], "text", info["rain_text"],
                  [l["text"] for l in s["legend"] if l["kind"] == "weather"], s["degradation"])
    if what == "lm":
        if g["landmarks"] != w["landmarks"]:
            gm = {p: t for t, v in g["landmarks"].items() for p in v}; pm = {p: t for t, v in w["landmarks"].items() for p in v}
            diff = [(p, gm.get(p), pm.get(p)) for p in set(gm) | set(pm) if gm.get(p) != pm.get(p)]
            c[s["style"]] += 1
            for p, a, b in diff: c[("miss" if b is None else ("extra" if a is None else "confuse"), s["style"])] += 1
            if c[s["style"]] <= 4:
                print(si, s["image"], s["style"], "use_color", info["use_color"], "legend types ok", info["legend_types"] == set(g["landmarks"]), diff[:6])
    if what == "edges":
        ge = {(a, d): v for a, dd in g["adj"].items() for d, v in dd.items() if d in (1, 3)}
        pe = {(a, d): v for a, dd in w["adj"].items() for d, v in dd.items() if d in (1, 3)}
        gl = {v: k for k, v in s["road_look"].items()}
        if gl != info["look2status"]: continue
        for k in set(ge) | set(pe):
            if k not in pe: c[("missing", s["style"], ge[k][0])] += 1
            elif k not in ge: c[("spurious", s["style"], pe[k][0])] += 1
            elif ge[k][0] != pe[k][0]: c[("status", s["style"], ge[k][0], pe[k][0])] += 1
            elif ge[k][1] != pe[k][1]: c[("stairs", s["style"], ge[k][1])] += 1
print(sorted(c.items(), key=lambda t: -t[1])[:40])
