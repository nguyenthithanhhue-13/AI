"""Phân tích LỖI end-to-end trên validation theo nhóm cảnh (mỗi robot): có điểm ghé / đích 2 bản / mô tả bản đồ / đêm / mưa /
gấp / dễ vỡ. Lưu dự đoán vào cache/p2_valpred.json."""
import sys, json, collections, os
sys.path.insert(0, "src")
from common import *
from pipeline_p2 import predict_split
if os.path.exists("cache/p2_valpred.json") and "re" not in sys.argv:
    preds = json.load(open("cache/p2_valpred.json"))
else:
    rows, preds, stat = predict_split("validation", "v1final", "trainonly")
    json.dump(preds, open("cache/p2_valpred.json", "w"))
rows, labels, scenes = load_split("validation")
lab = labels if isinstance(labels, list) else [labels[r["id"]] for r in rows]
G = {}
for s in scenes:
    m = s["mission"]; L = collections.Counter(l["type"] for l in s["landmarks"])
    k = (m.get("goal_ref") or {}).get("kind")
    G[s["scene_id"]] = {"ghé": bool(m["via"]), "đích 2 bản không ref": L[m["goal"]] > 1 and not m.get("goal_ref"),
                        "mô tả bản đồ": k in ("anchor_near", "north_most", "south_most", "west_most", "east_most"),
                        "đêm": s["style"] == "night", "mưa": s["weather"] == "rain", "gấp": m["urgent"], "dễ vỡ": m["fragile"]}
c = collections.Counter()
for i, row in enumerate(rows):
    sid = row["id"].rsplit("-", 1)[0]; r = int(row["id"].rsplit("R", 1)[1])
    ok = preds[i] == lab[i]
    c[r, "all", "n"] += 1; c[r, "all", "ok"] += ok
    for g, v in G[sid].items():
        c[r, g + "=" + str(int(v)), "n"] += 1; c[r, g + "=" + str(int(v)), "ok"] += ok
keys = ["all"] + [g + "=" + b for g in G[next(iter(G))] for b in "01"]
print("nhóm".ljust(26) + " ".join(f"R{r:<5d}" for r in range(10)))
for k in keys:
    print(k.ljust(22) + f"n={c[0, k, 'n']:3d} " + " ".join(f"{c[r, k, 'ok'] / max(1, c[r, k, 'n']):.3f}" for r in range(10)))
