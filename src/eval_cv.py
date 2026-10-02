"""Đo từng thành phần CV trên split có scenes.json, và điểm cuối khi CV + mission ĐÚNG (tách lỗi CV khỏi lỗi NLP).

    python src/eval_cv.py validation dev [limit]
"""
import os
import pickle
import sys
from collections import Counter

from common import *
from strategy import predict


def shift_world(w, origin=None):
    """Đưa (hàng, cột) về gốc 0 để so sánh (chiến thuật chỉ phụ thuộc vị trí tương đối)."""
    nodes = set(w["adj"]) | {w["robot"]} | {p for v in w["landmarks"].values() for p in v}
    r0, c0 = origin if origin is not None else (min(p[0] for p in nodes), min(p[1] for p in nodes))
    f = lambda p: (p[0] - r0, p[1] - c0)
    return {"adj": {f(a): dict(v) for a, v in w["adj"].items()}, "landmarks": {k: sorted(f(p) for p in v) for k, v in w["landmarks"].items()},
            "robot": f(w["robot"]), "heading": w["heading"], "rain": w["rain"]}


def main(split, mode, limit=None):
    rows, labels, scenes = load_split(split)
    if limit:
        scenes = scenes[:limit]
    cache = pickle.load(open(CACHE / f"world_{split}_{mode}{'_' + str(limit) if limit else ''}.pkl", "rb"))
    c = Counter()
    preds = []
    bad_scenes = []
    for si, s in enumerate(scenes):
        g = shift_world(world_from_scene(s))
        m = mission_from_scene(s)
        item = cache[s["image"]]
        c["scenes"] += 1
        if item["world"] is None:
            c["error"] += 1
            preds += [0] * 10
            continue
        info = item["info"]
        gn = {tuple(n["rc"]) for n in s["nodes"]}
        r0 = min(p[0] for p in gn); c0 = min(p[1] for p in gn)
        # gốc của lưới dự đoán: khớp theo VỊ TRÍ pixel với giao lộ thật (lấy độ lệch chỉ số phổ biến nhất),
        # để một điểm báo nhầm ở rìa không làm lệch toàn bộ phép so sánh
        gxy = {tuple(n["rc"]): n["xy"] for n in s["nodes"]}
        votes = Counter()
        for prc, (px, py) in info["xy"].items():
            grc = min(gxy, key=lambda k: (gxy[k][0] - px) ** 2 + (gxy[k][1] - py) ** 2)
            if (gxy[grc][0] - px) ** 2 + (gxy[grc][1] - py) ** 2 < 144:
                votes[(prc[0] - (grc[0] - r0), prc[1] - (grc[1] - c0))] += 1
        origin = votes.most_common(1)[0][0] if votes else None
        w = shift_world(item["world"], origin)
        gn = {(p[0] - r0, p[1] - c0) for p in gn}
        pr0, pc0 = origin if origin is not None else (0, 0)
        pn = {(p[0] - pr0, p[1] - pc0) for p in info["xy"]}
        ok_nodes = pn == gn
        c["nodes_exact"] += ok_nodes
        # cạnh
        ge = {(a, d): v for a, dd in g["adj"].items() for d, v in dd.items() if d in (1, 3)}
        pe = {(a, d): v for a, dd in w["adj"].items() for d, v in dd.items() if d in (1, 3)}
        gback = {(a, d): g["adj"][(a[0] + DRC[d][0], a[1] + DRC[d][1])][OPP[d]] for (a, d) in ge}
        pback = {(a, d): w["adj"][(a[0] + DRC[d][0], a[1] + DRC[d][1])][OPP[d]] for (a, d) in pe}
        e_ok = True
        for k in set(ge) | set(pe):
            c["edges"] += 1
            if k not in pe:
                c["edge_missing"] += 1; e_ok = False; continue
            if k not in ge:
                c["edge_spurious"] += 1; e_ok = False; continue
            st = ge[k][0] == pe[k][0]; sr = ge[k][1] == pe[k][1]
            ow = (ge[k][2], gback[k][2]) == (pe[k][2], pback[k][2])
            c["edge_status_err"] += not st; c["edge_stairs_err"] += not sr; c["edge_oneway_err"] += not ow
            e_ok &= st and sr and ow
        c["edges_exact"] += e_ok
        lm_ok = g["landmarks"] == w["landmarks"]
        c["landmarks_exact"] += lm_ok
        c["robot_pos"] += g["robot"] == w["robot"]
        c["heading"] += g["heading"] == w["heading"]
        c["rain"] += g["rain"] == w["rain"]
        gl = {v: k for k, v in s["road_look"].items()}   # look -> status
        c["legend_map"] += gl == info["look2status"]
        full = e_ok and lm_ok and g["robot"] == w["robot"] and g["heading"] == w["heading"] and g["rain"] == w["rain"]
        c["world_exact"] += full
        p = [predict(w, m, r) for r in range(10)]
        preds += p
        y = labels[si * 10:(si + 1) * 10]
        if p != y:
            bad_scenes.append((si, s["image"], s["style"], sum(a != b for a, b in zip(p, y)),
                               dict(nodes=ok_nodes, edges=e_ok, lm=lm_ok, robot=g["robot"] == w["robot"], head=g["heading"] == w["heading"],
                                    rain=g["rain"] == w["rain"], legend=gl == info["look2status"]), s["degradation"]))
    n = c["scenes"]
    print(f"== {split} ({mode}), {n} cảnh")
    for k in ("nodes_exact", "edges_exact", "landmarks_exact", "robot_pos", "heading", "rain", "legend_map", "world_exact"):
        print(f"   {k:16s} {c[k] / n:.4f}  ({n - c[k]} cảnh sai)")
    print(f"   cạnh: tổng {c['edges']}, thiếu {c['edge_missing']}, thừa {c['edge_spurious']}, sai trạng thái {c['edge_status_err']}, "
          f"sai bậc thang {c['edge_stairs_err']}, sai một chiều {c['edge_oneway_err']}  (lỗi CV: {c['error']})")
    score, per = macro_accuracy(rows[:len(preds)], labels[:len(preds)], preds)
    print(f"   ĐIỂM (CV dự đoán + mission đúng): macro acc {score:.4f}  theo robot {[round(v, 3) for v in per.values()]}")
    for b in bad_scenes[:int(os.environ.get("SHOW_BAD", "15"))]:
        print("   sai:", b[0], b[1], b[2], "robot sai:", b[3], "khâu sai:", [k for k, v in b[4].items() if not v], b[5])
    return bad_scenes


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
