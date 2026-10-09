"""LP biên cực đại theo từng KHOẢNG CÁCH (số đoạn ngắn nhất từ robot tới điểm đến) trong một nhóm điều kiện -> trọng số (chia cho trọng
số số đoạn) đổi theo khoảng cách thế nào. python scratch/p2_lp_dist.py <robot> <key> "<nhóm>" """
import sys, collections, numpy as np
R_, KEY, G = sys.argv[1], sys.argv[2], sys.argv[3]
sys.argv = ["x", R_, KEY]
g = {"__name__": "x"}
exec(open("scratch/p2_lp.py", encoding="utf-8").read().split('\nif __name__ == "__main__":\n')[0], g)
r = int(R_)
TR = g["prep"]("train")
items = [it for it in TR if str(g["gkey"](it[0])) == G]
N = ["hop", "đông", "mái", "lm", "bậc", "R", "L", "B", "kmái"] + ["lib", "dor", "spo", "cli", "can", "par", "lec", "lab", "off", "gat"] + ["R0", "L0", "B0"]
def short(x, mv):
    return int(min(F[:, 0].min() for d, F in mv))
bins = collections.defaultdict(list)
for it in items:
    s = short(*it); bins[min(s, 12) // 2 * 2].append(it)
for b in sorted(bins):
    its = bins[b]
    w = g["solve"](its, np.r_[1.0, np.zeros(21)])
    v = sum(g["predict"](mv, w) != x["y"][r] for x, mv in its)
    e = w[0] + w[8]   # đoạn không mái = hop + kmái
    print(f"số đoạn ngắn nhất {b}-{b + 1}: n={len(its):3d} không khớp {v:2d} | " + " ".join(f"{N[k]}={w[k] / w[0]:.2f}" for k in range(22) if w[k] > 1e-6))
