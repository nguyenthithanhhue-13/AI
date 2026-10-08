"""Mô hình chi phí theo ĐIỀU KIỆN: mỗi robot, mỗi nhóm (kiểu vẽ / thời tiết...) một bộ tham số, dò từng tọa độ trên train
nhóm đó, đo trên validation cùng nhóm. So với một bộ tham số chung.
    python scratch/p2_cd3.py <robot> <khóa nhóm: style|weather|none>"""
import sys, json, time, numpy as np, collections
sys.path.insert(0, "scratch")
from p2_fast import *

r = int(sys.argv[1]); key = sys.argv[2]
import pickle
_WI3 = pickle.load(open("cache/p2_wicon3.pkl", "rb")) if key in ("w3", "nd_w3") else {}


def W3(x):
    if x["s"]["weather"] == "rain":
        return "mưa"
    k = [v for (sp, sid), v in _WI3.items() if sid == x["s"]["scene_id"]]
    return ["nắng", "nắng nhẹ", "nhiều mây"][k[0]] if k else "khô?"
KF = {"style": lambda x: x["s"]["style"], "weather": lambda x: x["s"]["weather"], "none": lambda x: "all",
      "style_w": lambda x: x["s"]["style"] + "/" + x["s"]["weather"],
      "nd": lambda x: "đêm" if x["s"]["style"] == "night" else "ngày",
      "nd_w": lambda x: ("đêm" if x["s"]["style"] == "night" else "ngày") + "/" + x["s"]["weather"],
      "nd_u": lambda x: ("đêm" if x["s"]["style"] == "night" else "ngày") + "/gấp=" + str(x["m"]["urgent"]),
      "nd_f": lambda x: ("đêm" if x["s"]["style"] == "night" else "ngày") + "/dễ vỡ=" + str(x["m"]["fragile"]),
      "nd_w_u": lambda x: ("đêm" if x["s"]["style"] == "night" else "ngày") + "/" + x["s"]["weather"] + "/gấp=" + str(x["m"]["urgent"]),
      "nd_w_f": lambda x: ("đêm" if x["s"]["style"] == "night" else "ngày") + "/" + x["s"]["weather"] + "/dễ vỡ=" + str(x["m"]["fragile"]),
      "nd_u_f": lambda x: ("đêm" if x["s"]["style"] == "night" else "ngày") + "/gấp=" + str(x["m"]["urgent"]) + "/dễ vỡ=" + str(x["m"]["fragile"]),
      "nd_v": lambda x: ("đêm" if x["s"]["style"] == "night" else "ngày") + "/ghé=" + str(bool(x["m"]["via"])),
      "nd_v_f": lambda x: ("đêm" if x["s"]["style"] == "night" else "ngày") + "/ghé=" + str(bool(x["m"]["via"])) + "/dễ vỡ=" + str(x["m"]["fragile"]),
      "nd_v_u": lambda x: ("đêm" if x["s"]["style"] == "night" else "ngày") + "/ghé=" + str(bool(x["m"]["via"])) + "/gấp=" + str(x["m"]["urgent"]),
      "nd_v_w": lambda x: ("đêm" if x["s"]["style"] == "night" else "ngày") + "/ghé=" + str(bool(x["m"]["via"])) + "/" + x["s"]["weather"],
      "v": lambda x: "ghé=" + str(bool(x["m"]["via"])),
      "gt": lambda x: x["m"]["goal"], "gt_w": lambda x: x["m"]["goal"] + "/" + x["s"]["weather"],
      "gt_nd": lambda x: x["m"]["goal"] + "/" + ("đêm" if x["s"]["style"] == "night" else "ngày"),
      "vt": lambda x: str(x["m"]["via"]),
      "v_gt": lambda x: "ghé=True" if x["m"]["via"] else "ghé=False/" + x["m"]["goal"],
      "v_f": lambda x: "ghé=" + str(bool(x["m"]["via"])) + "/dễ vỡ=" + str(x["m"]["fragile"]),
      "tt": lambda x: str(x["m"]["via"] or x["m"]["goal"]),
      "tt_w": lambda x: str(x["m"]["via"] or x["m"]["goal"]) + "/" + x["s"]["weather"],
      "tt_nd": lambda x: str(x["m"]["via"] or x["m"]["goal"]) + "/" + ("đêm" if x["s"]["style"] == "night" else "ngày"),
      "tt_f": lambda x: str(x["m"]["via"] or x["m"]["goal"]) + "/dễ vỡ=" + str(x["m"]["fragile"]),
      "tt_u": lambda x: str(x["m"]["via"] or x["m"]["goal"]) + "/gấp=" + str(x["m"]["urgent"]),
      "tt_v": lambda x: str(x["m"]["via"] or x["m"]["goal"]) + "/ghé=" + str(bool(x["m"]["via"])),
      "gt_f": lambda x: x["m"]["goal"] + "/dễ vỡ=" + str(x["m"]["fragile"]),
      "gt_u": lambda x: x["m"]["goal"] + "/gấp=" + str(x["m"]["urgent"]),
      "gt_v": lambda x: x["m"]["goal"] + "/ghé=" + str(bool(x["m"]["via"])),
      "w3": lambda x: W3(x),
      "nd_w3": lambda x: ("đêm" if x["s"]["style"] == "night" else "ngày") + "/" + W3(x),
      "w": lambda x: x["s"]["weather"], "u": lambda x: str(x["m"]["urgent"]), "f": lambda x: str(x["m"]["fragile"])}[key]
TR = load("train"); VA = load("validation")
GRID = {"crowd": [0, 0.5, 1, 2, 3, 5, 8, 12], "cover": [-0.9, -0.6, -0.3, 0, 0.5, 1, 2], "lm": [0, 0.5, 1, 2, 3, 5, 8, 12],
        "normal": [0, 0.5, 1, 2, 3, 5, 8, 12, 20], "tR": [0, 0.25, 0.5, 1, 2], "tL": [0, 0.25, 0.5, 1, 2], "tB": [0, 0.5, 1, 2, 4, 8], "tR0": [0, 0.25, 0.5, 1, 2, 4], "tL0": [0, 0.25, 0.5, 1, 2, 4], "tB0": [0, 0.5, 1, 2, 4, 8, 20], "stairs": [-0.5, 0, 1, 3, 10]}
KEYS = ["crowd", "lm", "cover", "tR", "tL", "tB"] + (["tR0", "tL0", "tB0"] if __import__("os").environ.get("FIRST") else []) + (["stairs"] if r == 4 else []) + (["normal"] if __import__("os").environ.get("NORMAL") else [])


def fit(D):
    G = [SG(x["w"], r == 4, lm_excl(x)) for x in D]

    def sc(p):
        th = theta_of(**p); u = 0; i = 0
        for g, x in zip(G, D):
            a = argmins(g.q(th, x["legs"]), 1e-6)
            u += a == [x["y"][r]]; i += x["y"][r] in a
        return u + 0.5 * (i - u)
    p = {k: 0.0 for k in KEYS}
    for k in ("tR0", "tL0", "tB0"):
        if k in p:
            p[k] = None if False else 0.0
    b = sc(p)
    for _ in range(3):
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


def evaluate(p, D, tie="SRLB"):
    th = theta_of(**p); ok = 0
    for x in D:
        g = SG(x["w"], r == 4, lm_excl(x))
        a = argmins(g.q(th, x["legs"]), 1e-6)
        h = x["w"]["heading"]
        pred = min(a, key=lambda d: tie.index(REL[h][d])) if a else 0
        ok += pred == x["y"][r]
    return ok


gtr = collections.defaultdict(list); gva = collections.defaultdict(list)
for x in TR + (VA if __import__('os').environ.get('FITVAL') else []): gtr[KF(x)].append(x)
for x in VA: gva[KF(x)].append(x)
tot_tr = tot_va = 0; P = {}
t = time.time()
for k in sorted(gtr):
    gk = gtr[k]; gk = (gk[-len(gva.get(k, [])):] + gk[:int(__import__('os').environ.get('NFIT', '700'))]) if __import__('os').environ.get('FITVAL') else gk[:int(__import__('os').environ.get('NFIT', '700'))]
    p = fit(gk); P[k] = p
    a, b = evaluate(p, gtr[k]), evaluate(p, gva.get(k, []))
    tot_tr += a; tot_va += b
    print(f"R{r} {key}={k}: train {a}/{len(gtr[k])}  validation {b}/{len(gva.get(k, []))}  {p}  ({time.time()-t:.0f}s)", flush=True)
print(f"R{r} {key}: TỔNG train {tot_tr/len(TR):.4f}  validation {tot_va/len(VA):.4f}")
json.dump({str(k): v for k, v in P.items()}, open(f"cache/p2_cd3_r{r}_{key}{'_tv' if __import__('os').environ.get('FITVAL') else ''}.json", "w"))
