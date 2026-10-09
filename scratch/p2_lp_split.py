"""DÒ YẾU TỐ ẨN bằng tính TÁCH ĐƯỢC TUYẾN TÍNH: với mỗi nhóm điều kiện (KEY) của một robot, LP biên cực đại (scratch/p2_lp.py) cho số
cảnh train KHÔNG khớp; thử chia nhóm theo từng thuộc tính cảnh -> số cảnh không khớp sau khi chia (LP riêng từng nửa). So với chia
NGẪU NHIÊN cùng tỉ lệ (trung bình 3 lần). Thuộc tính tốt = giảm mạnh hơn hẳn ngẫu nhiên. Chỉ dùng train.
    python scratch/p2_lp_split.py <robot> <key> [min_violations=8]"""
import sys, collections, numpy as np
R_, KEY = sys.argv[1], sys.argv[2]; MINV = int(sys.argv[3]) if len(sys.argv) > 3 else 8
sys.argv = ["x", R_, KEY]
g = {"__name__": "x"}
exec(open("scratch/p2_lp.py", encoding="utf-8").read().split('\nif __name__ == "__main__":\n')[0], g)
r = int(R_)
TR = g["prep"]("train")
KM = ("anchor_near", "north_most", "south_most", "west_most", "east_most")
ATTR = {
    "kiểu vẽ": lambda x: x["s"]["style"],
    "mưa": lambda x: x["w"]["rain"],
    "gấp": lambda x: x["m"]["urgent"], "dễ vỡ": lambda x: x["m"]["fragile"], "có ghé": lambda x: bool(x["m"]["via"]),
    "đích mô tả bản đồ": lambda x: (x["m"]["goal_ref"] or {}).get("kind") in KM,
    "đích có tham chiếu": lambda x: bool(x["m"]["goal_ref"]),
    "kiểu tham chiếu": lambda x: (x["m"]["goal_ref"] or {}).get("kind", "-"),
    "đích 2 bản": lambda x: len(x["legs"][-1]) > 1,
    "loại đích": lambda x: x["m"]["goal"],
    "loại ghé": lambda x: str(x["m"]["via"]),
    "hướng mũi": lambda x: x["w"]["heading"],
    "chú giải hoán đổi": lambda x: any(k != v for k, v in x["s"]["road_look"].items()),
    "lưới >= 7 hàng": lambda x: x["s"]["grid"]["rows"] >= 7, "lưới >= 8 cột": lambda x: x["s"]["grid"]["cols"] >= 8,
    "nhiều địa điểm (>=11)": lambda x: len(x["s"]["landmarks"]) >= 11,
    "ảnh xoay": lambda x: abs(x["s"]["degradation"]["rotation_deg"]) > 0.5,
    "jpeg": lambda x: x["s"]["degradation"]["jpeg_quality"] is not None,
    "câu không dấu": lambda x: x["m"]["text"].isascii(),
    "câu có đính chính": lambda x: any(k in g["unaccent"](x["m"]["text"]) for k in ("nham", "doi lai", "thay vao do", "dung ra", "huy don")),
    "robot ở mép": lambda x: x["w"]["robot"][0] in (0, x["s"]["grid"]["rows"] - 1) or x["w"]["robot"][1] in (0, x["s"]["grid"]["cols"] - 1),
    "đích ở mép": lambda x: any(p[0] in (0, x["s"]["grid"]["rows"] - 1) or p[1] in (0, x["s"]["grid"]["cols"] - 1) for p in x["legs"][-1]),
    "đích xa (Manhattan>=6)": lambda x: min(abs(p[0] - x["w"]["robot"][0]) + abs(p[1] - x["w"]["robot"][1]) for p in x["legs"][-1]) >= 6,
    "đích phía trên robot": lambda x: min(p[0] for p in x["legs"][-1]) < x["w"]["robot"][0],
    "đích bên phải robot": lambda x: max(p[1] for p in x["legs"][-1]) > x["w"]["robot"][1],
}
sys.path.insert(0, "src")
from mapref import unaccent
g["unaccent"] = unaccent


def viol(items):
    if len(items) < 3:
        return 0
    w = g["solve"](items, np.r_[1.0, np.zeros(21)])
    return sum(g["predict"](mv, w) != x["y"][r] for x, mv in items)


groups = collections.defaultdict(list)
for it in TR:
    groups[g["gkey"](it[0])].append(it)
rng = np.random.default_rng(0)
tot = collections.Counter(); base_tot = 0; rnd_tot = collections.Counter()
for k in sorted(groups, key=str):
    items = groups[k]; v0 = viol(items)
    if v0 < MINV:
        continue
    base_tot += v0
    line = []
    for name, f in ATTR.items():
        part = collections.defaultdict(list)
        for it in items:
            part[f(it[0])].append(it)
        if len(part) < 2:
            tot[name] += v0; rnd_tot[name] += v0
            continue
        v = sum(viol(p) for p in part.values())
        # chia ngẫu nhiên cùng cỡ các phần
        sizes = [len(p) for p in part.values()]
        vr = []
        for _ in range(2):
            perm = rng.permutation(len(items)); s = 0; vv = 0
            for sz in sizes:
                vv += viol([items[i] for i in perm[s:s + sz]]); s += sz
            vr.append(vv)
        tot[name] += v; rnd_tot[name] += np.mean(vr)
        line.append((v - np.mean(vr), name, v, np.mean(vr)))
    line.sort()
    print(f"nhóm {k} (n={len(items)}): không khớp {v0} | tốt nhất: " + "; ".join(f"{n} {v} (ngẫu nhiên {vr:.0f})" for _, n, v, vr in line[:4]), flush=True)
print(f"\nR{r} key={KEY}: tổng không khớp {base_tot}. Sau khi chia theo (so với ngẫu nhiên):")
for name in sorted(tot, key=lambda n: tot[n] - rnd_tot[n]):
    print(f"   {name:24s} {tot[name]:4d}  (ngẫu nhiên {rnd_tot[name]:.0f}, lợi {rnd_tot[name] - tot[name]:+.0f})")
