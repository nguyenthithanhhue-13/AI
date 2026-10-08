"""TỰ DÒ ĐIỀU KIỆN ẨN cho một robot: thử từng cách chia cảnh thành 2 nhóm (thông tin trên ảnh / trong câu), học tham số chi
phí riêng từng nhóm (tối ưu từng tọa độ, gọn), đo validation. In xếp hạng theo validation.
    python scratch/p2_cond.py <robot> [n_fit]"""
import sys, json, time, collections, re, numpy as np
sys.path.insert(0, "scratch")
from p2_fast import *

r = int(sys.argv[1]); NF = int(sys.argv[2]) if len(sys.argv) > 2 else 350
sys.path.insert(0, "src")
from mapref import unaccent as _un
UN = lambda x: _un(x["m"]["text"])
GOALT = lambda x: next((t for t, v in x["w"]["landmarks"].items() if tuple(x["legs"][-1][0]) in v), None)
SPLITS = {
    "không chia": lambda x: 0, "mưa": lambda x: x["w"]["rain"], "đêm": lambda x: x["s"]["style"] == "night",
    "gấp": lambda x: x["m"]["urgent"], "dễ vỡ": lambda x: x["m"]["fragile"], "có điểm ghé": lambda x: bool(x["m"]["via"]),
    "đích mô tả bản đồ": lambda x: bool(x["m"]["goal_ref"]) and x["m"]["goal_ref"]["kind"] in ("anchor_near", "north_most", "south_most", "west_most", "east_most"),
    "đích 2 bản": lambda x: len(x["legs"][-1]) > 1 or len(x["w"]["landmarks"].get(GOALT(x) or "", [])) > 1,
    "mũi dọc": lambda x: x["w"]["heading"] in (0, 1), "mũi lên/trái": lambda x: x["w"]["heading"] in (0, 2),
    "kiểu classic/print": lambda x: x["s"]["style"] in ("classic", "print"),
    "chú giải hoán đổi": lambda x: any(k != v for k, v in x["s"]["road_look"].items()),
    "ảnh xoay": lambda x: abs(x["s"]["degradation"]["rotation_deg"]) > 0.5,
    "nén jpeg": lambda x: x["s"]["degradation"]["jpeg_quality"] is not None,
    "lưới >= 7 hàng": lambda x: x["s"]["grid"]["rows"] >= 7, "lưới >= 7 cột": lambda x: x["s"]["grid"]["cols"] >= 7,
    "nhiều địa điểm": lambda x: len(x["s"]["landmarks"]) >= 11,
    "đích sách vở (TV/GĐ/HC)": lambda x: GOALT(x) in ("library", "lecture", "office"),
    "đích ăn/ở (CA/KTX)": lambda x: GOALT(x) in ("canteen", "dorm"),
    "đích y tế/TN": lambda x: GOALT(x) in ("clinic", "lab"),
    "đích cổng/xe/TT": lambda x: GOALT(x) in ("gate", "parking", "sports"),
    "robot ở mép": lambda x: x["w"]["robot"][0] == 0 or x["w"]["robot"][1] == 0,
    "hàng giấy tờ": lambda x: bool(re.search(r"\b(tai lieu|ho so|giay|sach|phong bi|con dau|giao trinh|bien ten|so ghi)\b", UN(x))),
    "hàng đồ ăn": lambda x: bool(re.search(r"\b(khay com|nuoc|thuc pham|banh|com hop|do an)\b", UN(x))),
    "hàng y tế": lambda x: bool(re.search(r"\b(thuoc|khau trang|bang gac|so cuu)\b", UN(x))),
    "hàng thiết bị": lambda x: bool(re.search(r"\b(bong|vot|micro|may chieu|linh kien|dung cu|bo dam|the)\b", UN(x))),
    "hàng hóa chất/mẫu": lambda x: bool(re.search(r"\b(hoa chat|mau vat)\b", UN(x))),
    "ghé loại sách vở": lambda x: x["m"]["via"] in ("library", "lecture", "office"),
    "ghé loại ăn/ở": lambda x: x["m"]["via"] in ("canteen", "dorm"),
    "điểm đến kế sách vở": lambda x: (x["m"]["via"] or x["m"]["goal"]) in ("library", "lecture", "office"),
    "điểm đến kế ăn/ở": lambda x: (x["m"]["via"] or x["m"]["goal"]) in ("canteen", "dorm"),
}
ONLY = sys.argv[3].split(",") if len(sys.argv) > 3 else None
if ONLY:
    SPLITS = {k: v for k, v in SPLITS.items() if k in ONLY or k == "không chia"}
GRID = {"crowd": [0, 0.5, 1, 2, 3, 5, 8], "cover": [-0.6, -0.3, 0, 0.5, 1], "lm": [0, 0.5, 1, 2, 3, 5, 8],
        "tR": [0, 0.25, 0.5, 1], "tL": [0, 0.25, 0.5, 1], "tB": [0, 0.5, 1, 2, 4, 8], "stairs": [0, 1, 3]}
KEYS = ["crowd", "lm", "cover", "tR", "tL", "tB"] + (["stairs"] if r == 4 else [])
TR = load("train"); VA = load("validation")
import os
if os.environ.get("NOVIA"):
    TR = [x for x in TR if not x["m"]["via"]]; VA = [x for x in VA if not x["m"]["via"]]
GTR = {id(x): SG(x["w"], r == 4, lm_excl(x)) for x in TR[:1600]}
GVA = {id(x): SG(x["w"], r == 4, lm_excl(x)) for x in VA}


def pred(x, g, p):
    a = argmins(g.q(theta_of(**p), x["legs"]), 1e-6)
    return min(a, key=lambda d: "SRLB".index(REL[x["w"]["heading"]][d])) if a else None


def fit(D):
    def sc(p):
        th = theta_of(**p); u = 0; i = 0
        for x in D:
            a = argmins(GTR[id(x)].q(th, x["legs"]), 1e-6)
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


res = []
t0 = time.time()
TRf = TR[:1600]
for name, f in SPLITS.items():
    P = {}
    for v in (False, True):
        D = [x for x in TRf if bool(f(x)) == v][:NF]
        P[v] = fit(D) if D else {k: 0.0 for k in KEYS}
    ok = sum(pred(x, GVA[id(x)], P[bool(f(x))]) == x["y"][r] for x in VA)
    res.append((ok / len(VA), name, P))
    print(f"R{r} chia theo [{name}]: validation {ok / len(VA):.3f}  ({time.time() - t0:.0f}s)", flush=True)
res.sort(key=lambda z: -z[0])
print("XẾP HẠNG:", [(n, round(a, 3)) for a, n, _ in res[:8]])
json.dump([(a, n, {str(k): v for k, v in P.items()}) for a, n, P in res], open(f"cache/p2_cond_r{r}.json", "w"), ensure_ascii=False)
