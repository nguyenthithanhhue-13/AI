"""Chỉ số 'sức khỏe' giống check_submission nhưng trên validation (để có mốc so sánh)."""
import sys, pickle
sys.path.insert(0, "src")
from collections import Counter
from common import *
from pipeline import parse_split
from strategy import q_values, INF

rows = load_json(DATA / "validation" / "observations.json")
scenes = load_split("validation")[2]
for cvm, nlpm in [("dev", "cv5"), ("dev", "trainonly"), ("final", "final")]:
    worlds = pickle.load(open(CACHE / f"world_validation_{cvm}.pkl", "rb"))
    ms = parse_split("validation", nlpm)
    st = Counter()
    for si, s in enumerate(scenes):
        w = worlds[s["image"]]["world"]; m = ms[si]
        st["goal not on map"] += m["goal"] not in w["landmarks"]
        st["goal not on map but NLP goal correct"] += m["goal"] not in w["landmarks"] and m["goal"] == s["mission"]["goal"]
        st["via not on map"] += bool(m["via"]) and m["via"] not in w["landmarks"]
        st["no goal mention found (conf=0)"] += m["conf"] == 0.0
        if m["goal"] in w["landmarks"]:
            mm = dict(m)
            if mm.get("via") and mm["via"] not in w["landmarks"]: mm["via"], mm["via_ref"] = None, None
            st["no path R0"] += min(q_values(w, mm, 0)) == INF
    print(cvm, nlpm, dict(st))
