"""QUY HOẠCH TUYẾN TÍNH BIÊN CỰC ĐẠI (kiểu SVM mềm) cho trọng số chi phí, từng nhóm điều kiện, trên mặt Pareto có loại địa điểm
(cache/p2_paretotu_*): biến w >= 0 (19 đặc trưng đường + 3 phạt rẽ bước đầu), mỗi cảnh chọn đường "nhãn" p_i = đường rẻ nhất của bước
thật theo w hiện tại; ràng buộc (f_q - f_p) . w >= 1 - s_i với mọi đường q của bước khác; cực tiểu sum s_i + lam * sum w.
Lặp chọn lại p_i vài vòng. Đo train / validation (thông tin đúng).
    python scratch/p2_lp.py <robot> <key: chữ n w u f v g> [lam]"""
import sys, pickle, collections, time, numpy as np
from scipy.optimize import linprog
sys.path.insert(0, "scratch")
from p2_lib import *
r = int(sys.argv[1]); KEY = sys.argv[2]; LAM = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3
leg = int(r == 4); RELS = "SRLB"; NW = 22


def prep(split):
    D = load(split)
    P = pickle.load(open(f"cache/p2_paretotu_{split}_{leg}.pkl", "rb"))
    out = []
    for x, fr in zip(D, P):
        mv = []
        for d, (rel, F) in sorted(fr.items()):
            first = np.zeros(3)
            if rel != "S":
                first["RLB".index(rel)] = 1
            mv.append((d, np.hstack([F.astype(float), np.tile(first, (len(F), 1))])))
        out.append((x, mv))
    return out


def gkey(x):
    s = x["s"]; m = x["m"]; k = []
    for ch in KEY:
        k.append({"n": s["style"] == "night", "w": x["w"]["rain"], "u": m["urgent"], "f": m["fragile"], "v": bool(m["via"]),
                  "g": m["goal"]}[ch])
    return tuple(k)


def predict(mv, w):
    best = None
    for d, F in mv:
        c = (F @ w).min()
        if best is None or c < best[0] - 1e-9:
            best = (c, d)
    return best[1] if best else None


def solve(items, w0):
    w = w0.copy()
    for it in range(4):
        rows, rhs, si = [], [], []
        for i, (x, mv) in enumerate(items):
            y = x["y"][r]
            lab = [F for d, F in mv if d == y]
            if not lab:
                continue
            p = lab[0][np.argmin(lab[0] @ w)]
            for d, F in mv:
                if d == y:
                    continue
                for q in F:
                    rows.append(-(q - p)); rhs.append(-1.0); si.append(i)       # -(q-p).w - s_i <= -1
        n = len(items); A = np.zeros((len(rows), NW + n))
        A[:, :NW] = np.array(rows)
        A[np.arange(len(rows)), NW + np.array(si)] = -1
        c = np.hstack([np.full(NW, LAM), np.ones(n)])
        bounds = [(0, None)] * (NW + n)
        bounds[0] = (0.01, None)                     # số đoạn luôn có trọng số dương
        res = linprog(c, A_ub=A, b_ub=np.array(rhs), bounds=bounds, method="highs")
        if res.status != 0:
            break
        w = res.x[:NW]
    return w


if __name__ == "__main__":
    t0 = time.time()
    TR = prep("train"); VA = prep("validation")
    gt = collections.defaultdict(list); gv = collections.defaultdict(list)
    for it in TR:
        gt[gkey(it[0])].append(it)
    for it in VA:
        gv[gkey(it[0])].append(it)
    tt = tv = 0
    for g in sorted(gt, key=str):
        w = solve(gt[g], np.r_[1.0, np.zeros(NW - 1)])
        a = sum(predict(mv, w) == x["y"][r] for x, mv in gt[g]); av = sum(predict(mv, w) == x["y"][r] for x, mv in gv.get(g, []))
        tt += a; tv += av
        print(f"  {g}: train {a}/{len(gt[g])} val {av}/{len(gv.get(g, []))}", flush=True)
    print(f"R{r} LP biên cực đại key={KEY} lam={LAM}: train {tt / len(TR):.3f} validation {tv / len(VA):.3f} ({time.time() - t0:.0f}s)")
