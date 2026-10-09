"""Thử thứ tự PHÁ HÒA tuyệt đối (LÊN/XUỐNG/TRÁI/PHẢI) cho mô hình siêu tuyến tính đã học (không học lại): với các ngưỡng hòa eps,
chọn thứ tự (24 tương đối + 24 tuyệt đối) tốt nhất trên train, đo validation.
    python scratch/p2_hyper_abs.py <robot> [tag ""|goal]"""
import sys, os, itertools, numpy as np, torch
r = int(sys.argv[1]); HT = sys.argv[2] if len(sys.argv) > 2 else ""
os.environ["COND"] = {"goal": "goal", "basic": "basic"}.get(HT, "full")
sys.argv = ["x", str(r)]
h = {"__name__": "x"}
exec(open("scratch/p2_hyper.py", encoding="utf-8").read().split('\nif __name__ == "__main__":\n')[0], h)
TR = h["prep"]("train"); VA = h["prep"]("validation")
nets = []
for sd in torch.load(f"cache/p2_hyper{HT}_r{r}.pt"):
    m = h["Hyper"](TR["C"].shape[1]); m.load_state_dict(sd); nets.append(m.eval())


def qmat(B):
    with torch.no_grad():
        q = (sum(n.q(B, None) for n in nets) / len(nets)).numpy()
    Q = np.full((B["n"], 4), np.inf); Q[B["msc"].numpy(), B["mdir"].numpy()] = q
    R = np.full((B["n"], 4), -1); R[B["msc"].numpy(), B["mdir"].numpy()] = B["mrel"].numpy()
    return Q, R


def pred(Q, R, order, eps):
    best = Q.min(1, keepdims=True)
    cand = Q <= best + eps
    if order.startswith("A"):
        rank = np.array([order[1:].index(str(d)) for d in range(4)])[None, :].repeat(len(Q), 0)
    else:
        rank = np.where(R >= 0, np.array([order.index(c) for c in "SRLB"])[np.maximum(R, 0)], 9)
    return np.where(cand, rank, 99).argmin(1)


QT, RT = qmat(TR); QV, RV = qmat(VA); yt = TR["y"].numpy(); yv = VA["y"].numpy()
orders = ["".join(p) for p in itertools.permutations("SRLB")] + ["A" + "".join(p) for p in itertools.permutations("0123")]
print(f"R{r} [{HT or 'full'}] gốc (argmin): train {(QT.argmin(1) == yt).mean():.3f} val {(QV.argmin(1) == yv).mean():.3f}")
for eps in (1e-6, 0.05, 0.1, 0.2, 0.4):
    res = sorted(((pred(QT, RT, o, eps) == yt).mean(), o) for o in orders)[::-1][:3]
    print(f"   eps={eps}: " + " | ".join(f"{o} train {a:.3f} val {(pred(QV, RV, o, eps) == yv).mean():.3f}" for a, o in res))
