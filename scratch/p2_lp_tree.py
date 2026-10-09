"""Yếu tố ẩn: trong một nhóm điều kiện, LP biên cực đại (scratch/p2_lp.py) -> cảnh KHÔNG khớp; cây quyết định nông đoán "không khớp"
từ nhiều đặc trưng hình học / cảnh -> đặc trưng quan trọng. Chỉ train.
    python scratch/p2_lp_tree.py <robot> <key> "<nhóm>" """
import sys, collections, numpy as np
R_, KEY, G = sys.argv[1], sys.argv[2], sys.argv[3]
sys.argv = ["x", R_, KEY]
g = {"__name__": "x"}
exec(open("scratch/p2_lp.py", encoding="utf-8").read().split('\nif __name__ == "__main__":\n')[0], g)
from sklearn.tree import DecisionTreeClassifier, export_text
r = int(R_)
TR = g["prep"]("train")
items = [it for it in TR if str(g["gkey"](it[0])) == G]
w = g["solve"](items, np.r_[1.0, np.zeros(g["NW"] - 1)])
REL = g["REL"]; DRC = g["DRC"]
names = ["số đoạn ngắn nhất", "số bước ngắn nhất (hòa)", "số bước đi được", "bước thẳng đi được", "quay đầu đi được",
         "kề robot có đường đông", "kề robot có mái che", "đích cùng hàng", "đích cùng cột", "|Δhàng|", "|Δcột|", "robot ở mép",
         "số địa điểm", "đích 2 bản", "mưa", "hướng mũi", "chênh chi phí 2 bước tốt nhất", "đích ở hướng mũi", "cột lưới", "hàng lưới",
         "số đường đông", "số đường mái", "số cạnh đóng", "một chiều"]
X, Y = [], []
for x, mv in items:
    wd = x["w"]; s = wd["robot"]; h = wd["heading"]
    hops = {d: F[:, 0].min() for d, F in mv}
    sh = min(hops.values()) if hops else 0
    cost = sorted((F @ w).min() for d, F in mv)
    gl = x["legs"][-1][0]
    adj = wd["adj"].get(s, {})
    st = [i[0] for i in adj.values()]
    allst = [i[0] for dd in wd["adj"].values() for i in dd.values()]
    fwd = (gl[0] - s[0], gl[1] - s[1]); hd = DRC[h]
    X.append([sh, sum(v == sh for v in hops.values()), len(mv), any(REL[h][d] == "S" for d, _ in mv), any(REL[h][d] == "B" for d, _ in mv),
              "crowded" in st, "covered" in st, gl[0] == s[0], gl[1] == s[1], abs(gl[0] - s[0]), abs(gl[1] - s[1]),
              s[0] in (0, x["s"]["grid"]["rows"] - 1) or s[1] in (0, x["s"]["grid"]["cols"] - 1), len(x["s"]["landmarks"]),
              len(x["legs"][-1]) > 1, wd["rain"], h, (cost[1] - cost[0]) if len(cost) > 1 else 9, fwd[0] * hd[0] + fwd[1] * hd[1] > 0,
              x["s"]["grid"]["cols"], x["s"]["grid"]["rows"], allst.count("crowded") / 2, allst.count("covered") / 2,
              allst.count("closed") / 2, sum(1 for dd in wd["adj"].values() for i in dd.values() if not i[2])])
    Y.append(g["predict"](mv, w) != x["y"][r])
X = np.array(X, float); Y = np.array(Y)
print(f"nhóm {G}: {len(Y)} cảnh, không khớp {Y.sum()}")
t = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, class_weight="balanced", random_state=0).fit(X, Y)
print(export_text(t, feature_names=names))
imp = sorted(zip(t.feature_importances_, names), reverse=True)[:6]
print("quan trọng:", [(n, round(v, 2)) for v, n in imp])
# so sánh trung bình từng đặc trưng giữa khớp / không khớp
for j, n in enumerate(names):
    a, b = X[~Y, j].mean(), X[Y, j].mean()
    if abs(a - b) > 0.25 * (X[:, j].std() + 1e-9):
        print(f"   {n:28s} khớp {a:6.2f} | không khớp {b:6.2f}")
