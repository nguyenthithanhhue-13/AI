"""Độ NHẠY của trọng số mô hình siêu tuyến tính (bản đầy đủ, chỉ học train) với từng điều kiện: lật cờ (mưa, đêm, gấp, dễ vỡ, ghé,
đích mô tả bản đồ) hoặc đổi loại đích / loại ghé sang loại khác, trên các cảnh train -> trung bình |thay đổi quyết định| (tỉ lệ cảnh
mà bước đi thay đổi). Điều kiện nào gần 0 = robot không dùng.
    python scratch/p2_hyper_sens.py <robot>"""
import sys, os, numpy as np, torch
r = int(sys.argv[1])
os.environ["COND"] = "full"
sys.argv = ["x", str(r)]
h = {"__name__": "x"}
exec(open("scratch/p2_hyper.py", encoding="utf-8").read().split('\nif __name__ == "__main__":\n')[0], h)
TR = h["prep"]("train")
nets = []
for sd in torch.load(f"cache/p2_hyper_r{r}.pt"):
    m = h["Hyper"](TR["C"].shape[1]); m.load_state_dict(sd); nets.append(m.eval())


def pred(B):
    with torch.no_grad():
        q = (sum(n.q(B, None) for n in nets) / len(nets)).numpy()
    Q = np.full((B["n"], 4), np.inf); Q[B["msc"].numpy(), B["mdir"].numpy()] = q
    return Q.argmin(1)


base = pred(TR)
C0 = TR["C"].clone()
out = []
for j, name in enumerate(["mưa", "đêm", "gấp", "dễ vỡ", "có ghé", "đích bản đồ"]):
    C = C0.clone(); C[:, j] = 1 - C[:, j]
    out.append((name, (pred(dict(TR, C=C)) != base).mean()))
rng = np.random.default_rng(0)
for lo, name in ((6, "loại đích"), (16, "loại ghé")):
    C = C0.clone()
    for i in range(len(C)):
        cur = C[i, lo:lo + 10]
        if cur.sum() == 0:
            continue
        k = int(cur.argmax()); k2 = int(rng.choice([t for t in range(10) if t != k]))
        C[i, lo:lo + 10] = 0; C[i, lo + k2] = 1
    out.append((name, (pred(dict(TR, C=C)) != base).mean()))
print(f"R{r}: " + "  ".join(f"{n}={v:.3f}" for n, v in out))
