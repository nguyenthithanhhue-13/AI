"""Mỗi loại lỗi đọc sai làm từng robot mất bao nhiêu điểm? (validation, bản đồ đúng, cố tình làm sai một trường)

Dùng để suy ra robot yếu nhất trên bảng xếp hạng đang yếu vì trường nào.
"""
import sys
sys.path.insert(0, "src")
import numpy as np
from common import *
from strategy import predict

rows, labels, scenes = load_split("validation")
W = [world_from_scene(s) for s in scenes]; M = [mission_from_scene(s) for s in scenes]
Y = np.array(labels).reshape(-1, 10)


def acc(mut_m=None, mut_w=None, cond=None):
    ok = np.zeros(10); n = 0
    for w, m, y in zip(W, M, Y):
        if cond and not cond(w, m): continue
        m2 = mut_m(dict(m)) if mut_m else m
        w2 = mut_w(dict(w)) if mut_w else w
        ok += np.array([predict(w2, m2, r) for r in range(10)]) == y; n += 1
    return ok / n, n


def flip(key):
    def f(m): m[key] = not m[key]; return m
    return f


def drop_via(m): m["via"], m["via_ref"] = None, None; return m
def drop_gref(m): m["goal_ref"] = None; return m
def flip_rain(w): w["rain"] = not w["rain"]; return w


print("Điểm từng robot khi một trường bị đọc SAI ở mọi cảnh có liên quan (1.000 = không ảnh hưởng):")
print(f"{'lỗi':46s} n   " + " ".join(f"R{r}   " for r in range(10)))
for name, mm, mw, cond in [
    ("bỏ sót 'gấp' (thật là gấp)", flip("urgent"), None, lambda w, m: m["urgent"]),
    ("báo nhầm 'gấp' (thật không gấp)", flip("urgent"), None, lambda w, m: not m["urgent"]),
    ("bỏ sót 'dễ vỡ' (thật là dễ vỡ)", flip("fragile"), None, lambda w, m: m["fragile"]),
    ("báo nhầm 'dễ vỡ' (thật không dễ vỡ)", flip("fragile"), None, lambda w, m: not m["fragile"]),
    ("sai thời tiết", None, flip_rain, None),
    ("bỏ sót điểm ghé", drop_via, None, lambda w, m: m["via"] is not None),
    ("bỏ sót tham chiếu của đích", drop_gref, None, lambda w, m: m["goal_ref"] is not None),
]:
    a, n = acc(mm, mw, cond)
    print(f"{name:46s} {n:3d} " + " ".join(f"{v:.3f}" for v in a))
