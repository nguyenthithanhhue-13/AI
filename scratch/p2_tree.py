"""CÂY ĐIỀU KIỆN tự động cho một robot: ở mỗi nút thử mọi cách chia (điều kiện nhị phân + loại nơi giao), mỗi nhánh học
tham số chi phí riêng (tối ưu từng tọa độ). Chọn cách chia theo PHẦN TRAIN GIỮ LẠI (20%), không theo validation; validation chỉ
để báo cáo. Độ sâu tối đa 2, nhánh tối thiểu 120 cảnh train.
    python scratch/p2_tree.py <robot>"""
import sys, json, time, re, collections, numpy as np
sys.path.insert(0, "scratch"); sys.path.insert(0, "src")
from p2_fast import *
from mapref import unaccent as _un

r = int(sys.argv[1])
NM = lambda x: x["s"]["style"] == "night"
BIN = {"mưa": lambda x: x["w"]["rain"], "đêm": NM, "gấp": lambda x: x["m"]["urgent"], "dễ vỡ": lambda x: x["m"]["fragile"],
       "ghé": lambda x: bool(x["m"]["via"]),
       "đích bản đồ": lambda x: bool(x["m"]["goal_ref"]) and x["m"]["goal_ref"]["kind"] in ("anchor_near", "north_most", "south_most", "west_most", "east_most"),
       "lưới>=7 hàng": lambda x: x["s"]["grid"]["rows"] >= 7, "mũi dọc": lambda x: x["w"]["heading"] in (0, 1)}
GRID = {"crowd": [0, 0.5, 1, 2, 3, 5, 8], "cover": [-0.6, -0.3, 0, 0.5, 1, 2], "lm": [0, 0.5, 1, 2, 3, 5, 8],
        "tR": [0, 0.25, 0.5, 1], "tL": [0, 0.25, 0.5, 1], "tB": [0, 0.5, 1, 2, 4, 8], "stairs": [0, 1, 3]}
KEYS = ["crowd", "lm", "cover", "tR", "tL", "tB"] + (["stairs"] if r == 4 else [])
ALL = load("train"); VA = load("validation")
rng = np.random.default_rng(0); idx = rng.permutation(len(ALL))
FIT = [ALL[i] for i in idx[:1600]]; HOLD = [ALL[i] for i in idx[1600:]]
SGC = {}


def sg(x):
    k = id(x)
    if k not in SGC:
        SGC[k] = SG(x["w"], r == 4, lm_excl(x))
    return SGC[k]


def pred(x, p):
    a = argmins(sg(x).q(theta_of(**p), x["legs"]), 1e-6)
    return min(a, key=lambda d: "SRLB".index(REL[x["w"]["heading"]][d])) if a else None


def fit(D, n=350):
    D = D[:n]

    def sc(p):
        th = theta_of(**p); u = i = 0
        for x in D:
            a = argmins(sg(x).q(th, x["legs"]), 1e-6)
            u += a == [x["y"][r]]; i += x["y"][r] in a
        return u + 0.5 * (i - u)
    p = {k: 0.0 for k in KEYS}; b = sc(p)
    for _ in range(2):
        ch = False
        for k in KEYS:
            for v in GRID[k]:
                if v != p[k]:
                    q = dict(p, **{k: v}); s = sc(q)
                    if s > b:
                        b, p, ch = s, q, True
        if not ch:
            break
    return p


def splitters():
    for n, f in BIN.items():
        yield n, (lambda f: lambda x: str(bool(f(x))))(f)
    yield "loại đích", lambda x: x["m"]["goal"]
    yield "nhóm đích3", lambda x: {"library": "a", "lecture": "a", "office": "a", "canteen": "b", "dorm": "b"}.get(x["m"]["goal"], "c")


def build(fit_set, hold_set, depth, used):
    """Trả về cây: ("leaf", p) hoặc ("split", tên, hàm, {nhánh: cây})."""
    p = fit(fit_set)
    base = sum(pred(x, p) == x["y"][r] for x in hold_set)
    node = ("leaf", p)
    if depth == 0 or len(fit_set) < 240:
        return node, base
    best = (base, None)
    for name, f in splitters():
        if name in used:
            continue
        groups = collections.defaultdict(list)
        for x in fit_set:
            groups[f(x)].append(x)
        if any(len(g) < 120 for g in groups.values()) or len(groups) < 2:
            continue
        P = {k: fit(g) for k, g in groups.items()}
        s = sum(pred(x, P.get(f(x), p)) == x["y"][r] for x in hold_set)
        if s > best[0] + 2:
            best = (s, (name, f, groups))
    if best[1] is None:
        return node, base
    name, f, groups = best[1]
    hold_g = collections.defaultdict(list)
    for x in hold_set:
        hold_g[f(x)].append(x)
    kids = {}; tot = 0
    for k, g in groups.items():
        kids[k], s = build(g, hold_g.get(k, []), depth - 1, used | {name})
        tot += s
    return ("split", name, f, kids), tot


def apply(tree, x):
    if tree[0] == "leaf":
        return tree[1]
    _, name, f, kids = tree
    k = f(x)
    return apply(kids[k], x) if k in kids else apply(next(iter(kids.values())), x)


def show(tree, ind=""):
    if tree[0] == "leaf":
        return ind + str({k: v for k, v in tree[1].items() if v}) + "\n"
    s = ""
    for k, t in tree[3].items():
        s += f"{ind}[{tree[1]} = {k}]\n" + show(t, ind + "   ")
    return s


def to_json(tree):
    if tree[0] == "leaf":
        return {"leaf": tree[1]}
    return {"split": tree[1], "kids": {k: to_json(t) for k, t in tree[3].items()}}


t0 = time.time()
tree, hs = build(FIT, HOLD, 2, set())
va = sum(pred(x, apply(tree, x)) == x["y"][r] for x in VA) / len(VA)
print(f"R{r}: giữ lại {hs}/{len(HOLD)} = {hs/len(HOLD):.3f} | validation {va:.3f}  ({time.time()-t0:.0f}s)")
print(show(tree))
json.dump(to_json(tree), open(f"cache/p2_tree_r{r}.json", "w"), ensure_ascii=False)
