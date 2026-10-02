"""Chẩn đoán một mô hình: sai ở điều kiện nào, khoảng cách chi phí của nhãn so với tối ưu."""
import sys, json
sys.path.insert(0, "src")
from collections import Counter
from common import *
from strategy import *

rows, labels, scenes = load_split("train")
W = [world_from_scene(s) for s in scenes]
M = [mission_from_scene(s) for s in scenes]
Y = [labels[i * 10:(i + 1) * 10] for i in range(len(scenes))]


def diag(r, P, cond=lambda i: True, order="SRLB", show=0):
    n = ok = 0
    gaps = Counter(); by = Counter(); tot = Counter(); tie_rel = Counter()
    shown = 0
    for i in range(len(Y)):
        if not cond(i): continue
        w, m = W[i], M[i]
        q = path_q(w, legs_for(w, m), P, r == 4)
        p = pick(q, w["heading"], order)
        y = Y[i][r]
        n += 1; ok += p == y
        b = min(q)
        key = ("via" if m["via"] else "novia", "dup" if len(w["landmarks"][m["goal"]]) > 1 and not m["goal_ref"] else "single")
        tot[key] += 1; by[key] += p == y
        if p != y:
            gaps[round(q[y] - b, 2)] += 1
            if abs(q[y] - b) < 1e-6:
                ties = [d for d in range(4) if q[d] <= b + 1e-6]
                tie_rel[("".join(sorted(REL[w["heading"]][d] for d in ties)), REL[w["heading"]][y])] += 1
            if shown < show:
                shown += 1
                print("  scene", i, scenes[i]["image"], "robot", w["robot"], ACTIONS[w["heading"]], "label", ACTIONS[y], "pred", ACTIONS[p],
                      "q", [round(x, 2) for x in q], "goal", m["goal"], w["landmarks"][m["goal"]], "ref", m["goal_ref"], "via", m["via"], w["landmarks"].get(m["via"]))
    print(f"robot {r} P={P}: acc {ok / n:.4f} (n={n})")
    print("   by cond:", {k: f"{by[k]}/{tot[k]}" for k in tot})
    print("   miss gaps:", sorted(gaps.items()))
    print("   miss within ties:", sorted(tie_rel.items()))


if __name__ == "__main__":
    diag(4, {"crowd": 1}, show=6)
    diag(5, {"turn": 3, "turnB": 100}, show=4)
    diag(8, {"turnR": 0, "turnL": 3, "turnB": 6}, show=4)
    diag(7, {"turn": 0.5, "turnB": 8, "crowd": 3}, cond=lambda i: M[i]["fragile"], show=4)
