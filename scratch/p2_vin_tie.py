"""PHÁ HÒA cho bộ tìm đường học được: các bước có log-xác suất (gộp nhóm mạng) cách bước tốt nhất < delta được coi là hòa,
chọn theo thứ tự hướng riêng của robot (tương đối S/R/L/B hoặc tuyệt đối LÊN/XUỐNG/TRÁI/PHẢI). delta + thứ tự học trên train,
đo validation (thông tin đúng). Cần PYTHONPATH=F:/pylibs.
    python scratch/p2_vin_tie.py <robot> <tag: '' | big | abs> [vpre]"""
import sys, glob, itertools, os
r = int(sys.argv[1]); TAGS = sys.argv[2].split(",") if len(sys.argv) > 2 else [""]
os.environ["VPRE"] = sys.argv[3] if len(sys.argv) > 3 else ""
sys.argv = [sys.argv[0], str(r), "1", "128" if "big" in TAGS else "64"]
import numpy as np, torch
exec(open("scratch/p2_vin2.py", encoding="utf-8").read().split("t0 = time.time()")[0])
sys.path.insert(0, "src"); import common
nets = []
for tag in TAGS:
    hid = 128 if tag == "big" else 64
    for f in sorted(glob.glob(f"cache/p2_vin2{tag}_r{r}_s[0-9].pt")):
        HID = hid
        n = Net().to(dev); n.load_state_dict(torch.load(f)); nets.append(n.eval())
print("số mạng", len(nets))


def logits(D):
    out = []
    with torch.no_grad():
        for b in batches(D, 150, False):
            out.append(sum(torch.log_softmax(n(b, hard=True), 1) for n in nets).cpu())
    return torch.cat(out).numpy()


Ltr, Lva = logits(TR), logits(VA)
ytr, yva = TR["y"].numpy(), VA["y"].numpy()
htr = (TR["start"].numpy() % 4); hva = (VA["start"].numpy() % 4)
REL = [[common.rel_turn(h, d) for d in range(4)] for h in range(4)]


def apply(L, h, delta, kind, order):
    out = L.argmax(1).copy()
    for i in range(len(L)):
        top = L[i].max()
        cand = [d for d in range(4) if L[i, d] >= top - delta and L[i, d] > -50]
        if len(cand) > 1:
            key = (lambda d: order.index(REL[h[i]][d])) if kind == "rel" else (lambda d: order.index(d))
            out[i] = min(cand, key=key)
    return out


base_tr = (Ltr.argmax(1) == ytr).mean(); base_va = (Lva.argmax(1) == yva).mean()
print(f"R{r} gốc: train {base_tr:.3f} validation {base_va:.3f}")
res = []
for delta in (0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0):
    for kind, alph in (("rel", "SRLB"), ("abs", [0, 1, 2, 3])):
        for order in itertools.permutations(alph):
            order = list(order)
            a = (apply(Ltr, htr, delta, kind, order) == ytr).mean()
            res.append((a, delta, kind, order))
res.sort(key=lambda z: -z[0])
for a, delta, kind, order in res[:5]:
    v = (apply(Lva, hva, delta, kind, order) == yva).mean()
    print(f"   delta={delta} {kind} {''.join(map(str, order))}: train {a:.3f} validation {v:.3f}")
# độ chắc vs đúng / sai trên validation
m = np.sort(Lva, 1); marg = m[:, -1] - m[:, -2]; ok = Lva.argmax(1) == yva
for lo, hi in ((0, .1), (.1, .5), (.5, 1), (1, 2), (2, 99)):
    s = (marg >= lo) & (marg < hi)
    print(f"   biên [{lo},{hi}): n={s.sum():3d} đúng {ok[s].mean() if s.any() else 0:.3f}")
