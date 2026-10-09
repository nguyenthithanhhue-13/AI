"""R9 (tham lam, chỉ nhìn một bước): điểm bước d = w(C) · đặc trưng bước; w = MLP(điều kiện cảnh) (tuyến tính theo đặc trưng).
Đặc trưng bước: khoảng cách từ giao lộ kế tới điểm đến gần nhất (Manhattan, Euclid lưới, số đoạn BFS, Euclid PIXEL / bước lưới TB),
các khoảng cách đó TRỪ khoảng cách tại vị trí hiện tại, [đông, mái che, thường, bậc thang, vào địa điểm, vào từng loại (10)],
rẽ S/R/L/B. Điểm đến = điểm ghé nếu có, ngược lại nơi giao. Học softmax CE trên train, đo validation (thông tin đúng).
    python scratch/p2_greedy.py [robot=9] [steps] [hid] [seeds] [final]"""
import sys, os, math, collections, itertools, time, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "scratch")
from p2_lib import *

r = int(sys.argv[1]) if len(sys.argv) > 1 else 9; STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
HID = int(sys.argv[3]) if len(sys.argv) > 3 else 64; SEEDS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
FINAL = len(sys.argv) > 5 and sys.argv[5] == "final"
PL = ["library", "dorm", "sports", "clinic", "canteen", "parking", "lecture", "lab", "office", "gate"]
COND = os.environ.get("COND", "full")
torch.set_num_threads(6)


def bfs(w, tg, legged):
    D = {t: 0 for t in tg}; q = collections.deque(tg)
    radj = collections.defaultdict(list)
    for n, dd in w["adj"].items():
        for d, info in dd.items():
            if edge_ok(info, legged):
                radj[(n[0] + DRC[d][0], n[1] + DRC[d][1])].append(n)
    while q:
        n = q.popleft()
        for m in radj.get(n, []):
            if m not in D:
                D[m] = D[n] + 1; q.append(m)
    return D


def feats(w, legs, xy, legged=False):
    """{d: vector} cho các bước đi được."""
    s = w["robot"]; tg = legs[0]; h = w["heading"]
    tgs = {p for L in legs for p in L}
    lmt = {p: PL.index(t) for t, v in w["landmarks"].items() for p in v if p not in tgs}
    V = bfs(w, tg, legged)
    sp = np.mean([math.dist(xy[a], xy[b]) for a in xy for b in [(a[0], a[1] + 1)] if b in xy] or [1.0])
    def dists(n):
        man = min(abs(n[0] - t[0]) + abs(n[1] - t[1]) for t in tg)
        euc = min(math.hypot(n[0] - t[0], n[1] - t[1]) for t in tg)
        px = min(math.dist(xy[n], xy[t]) for t in tg) / sp if n in xy and all(t in xy for t in tg) else euc
        return [man, euc, min(V.get(n, 30), 30), px]
    d0 = dists(s)
    out = {}
    for d, info in w["adj"].get(s, {}).items():
        if not edge_ok(info, legged):
            continue
        n2 = (s[0] + DRC[d][0], s[1] + DRC[d][1])
        st = info[0]; rel = REL[h][d]
        dn = dists(n2)
        out[d] = np.array(dn + [a - b for a, b in zip(dn, d0)] +
                          [st == "crowded", st == "covered", st == "normal", bool(info[1]), n2 in lmt] + [lmt.get(n2) == k for k in range(10)] +
                          [rel == x for x in "SRLB"], np.float32)
    return out


def cond(x):
    m = x["m"]; s = x["s"]
    mapg = bool(m["goal_ref"]) and m["goal_ref"]["kind"] in ("anchor_near", "north_most", "south_most", "west_most", "east_most")
    base = [x["w"]["rain"], s["style"] == "night", m["urgent"], m["fragile"], bool(m["via"]), mapg]
    if COND == "basic":
        return np.array(base, np.float32)
    if COND == "goal":
        return np.array(base + [m["goal"] == t for t in PL], np.float32)
    return np.array(base + [m["goal"] == t for t in PL] + [m["via"] == t for t in PL], np.float32)


def prep(split):
    D = load(split)
    X = np.zeros((len(D), 4, 27), np.float32); M = np.zeros((len(D), 4), bool); C = []; Y = []
    for i, x in enumerate(D):
        xy = {tuple(n["rc"]): tuple(n["xy"]) for n in x["s"]["nodes"]}
        for d, v in feats(x["w"], x["legs"], xy, r == 4).items():
            X[i, d] = v; M[i, d] = True
        C.append(cond(x)); Y.append(x["y"][r])
    return dict(X=torch.tensor(X), M=torch.tensor(M), C=torch.tensor(np.stack(C)), y=torch.tensor(Y))


class G(nn.Module):
    def __init__(self, nc, nf=27):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(nc, HID), nn.ReLU(), nn.Linear(HID, HID), nn.ReLU(), nn.Linear(HID, nf))

    def forward(self, B):
        w = self.f(B["C"])                                   # [n, nf]
        s = -(B["X"] * w[:, None, :]).sum(-1)                # điểm thấp = tốt -> logit = -điểm
        return torch.where(B["M"], s, torch.full_like(s, -1e4))


def train_one(seed, B):
    torch.manual_seed(seed)
    net = G(B["C"].shape[1]); opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(STEPS):
        loss = nn.functional.cross_entropy(net(B), B["y"]); opt.zero_grad(); loss.backward(); opt.step()
    return net


if __name__ == "__main__":
    t0 = time.time()
    TR = prep("train"); VA = prep("validation")
    DATA = {k: torch.cat([TR[k], VA[k]]) for k in TR} if FINAL else TR
    nets = [train_one(s, DATA) for s in range(SEEDS)]
    acc = lambda B: (sum(torch.log_softmax(n(B), 1) for n in nets).argmax(1) == B["y"]).float().mean().item()
    tag = os.environ.get("GTAG", "")
    torch.save([n.state_dict() for n in nets], f"cache/p2_greedy{tag}_r{r}{'_final' if FINAL else ''}.pt")
    print(f"R{r} greedy{tag} cond={COND} steps={STEPS} hid={HID} seeds={SEEDS} final={FINAL}: train {acc(TR):.3f} validation {acc(VA):.3f} ({time.time() - t0:.0f}s)")
