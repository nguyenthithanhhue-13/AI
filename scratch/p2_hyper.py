"""Mô hình SIÊU TUYẾN TÍNH trên mặt Pareto (TYPED=1 UNC=1 scratch/p2_pareto.py -> cache/p2_paretotu_*):
trọng số đặc trưng đường w = softplus(MLP(điều kiện cảnh)) (22 số: 19 đặc trưng đường + 3 phạt rẽ bước đầu); chi phí đường =
w · đặc trưng (TUYẾN TÍNH theo đường, như bộ sinh); bước = argmin qua các đường Pareto. Học bằng gradient (softmin), đo validation.
Điều kiện: [mưa, đêm, gấp, dễ vỡ, có ghé, đích mô tả bản đồ, loại đích (10), loại ghé (10)].
    python scratch/p2_hyper.py <robot> [steps] [hidden] [seeds] [final]"""
import sys, os, pickle, itertools, time, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "scratch")
from p2_lib import *

r = int(sys.argv[1]); STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1500; HID = int(sys.argv[3]) if len(sys.argv) > 3 else 64
SEEDS = int(sys.argv[4]) if len(sys.argv) > 4 else 3; FINAL = len(sys.argv) > 5 and sys.argv[5] == "final"
leg = int(r == 4); RELS = "SRLB"
PL = ["library", "dorm", "sports", "clinic", "canteen", "parking", "lecture", "lab", "office", "gate"]
NFP = 19
torch.set_num_threads(6)


COND = os.environ.get("COND", "full")      # full (26) | basic (6 cờ) | goal (6 cờ + loại đích) | 4 (mưa, đêm, gấp, dễ vỡ)


def cond(x):
    m = x["m"]; s = x["s"]
    mapg = bool(m["goal_ref"]) and m["goal_ref"]["kind"] in ("anchor_near", "north_most", "south_most", "west_most", "east_most")
    base = [x["w"]["rain"], s["style"] == "night", m["urgent"], m["fragile"], bool(m["via"]), mapg]
    if COND == "4":
        return np.array(base[:4], np.float32)
    if COND == "basic":
        return np.array(base, np.float32)
    if COND == "goal":
        return np.array(base + [m["goal"] == t for t in PL], np.float32)
    return np.array(base + [m["goal"] == t for t in PL] + [m["via"] == t for t in PL], np.float32)


def prep(split):
    D = load(split)
    P = pickle.load(open(f"cache/p2_paretotu_{split}_{leg}.pkl", "rb"))
    F, pm, msc, mdir, mrel, C, Y = [], [], [], [], [], [], []
    mi = 0
    for i, (x, fr) in enumerate(zip(D, P)):
        C.append(cond(x)); Y.append(x["y"][r])
        for d, (rel, Fm) in sorted(fr.items()):
            F.append(Fm); pm += [mi] * len(Fm); msc.append(i); mdir.append(d); mrel.append(RELS.index(rel)); mi += 1
    return dict(F=torch.tensor(np.concatenate(F), dtype=torch.float32), pm=torch.tensor(pm), msc=torch.tensor(msc),
                mdir=torch.tensor(mdir), mrel=torch.tensor(mrel), C=torch.tensor(np.stack(C)), y=torch.tensor(Y), n=len(D), nm=mi)


class Hyper(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(nc, HID), nn.ReLU(), nn.Linear(HID, HID), nn.ReLU(), nn.Linear(HID, NFP + 3))
        with torch.no_grad():
            self.f[-1].bias.fill_(-2.0); self.f[-1].bias[0] = 1.0
        self.t = nn.Parameter(torch.tensor(0.0))

    def q(self, B, tau):
        w = nn.functional.softplus(self.f(B["C"]))                       # [n, 22]
        c = (B["F"] * w[B["msc"][B["pm"]], :NFP]).sum(1)                 # chi phí từng đường
        if tau is None:
            qm = torch.full((B["nm"],), float("inf")).scatter_reduce(0, B["pm"], c, "amin")
        else:
            m = torch.full((B["nm"],), float("inf")).scatter_reduce(0, B["pm"], c, "amin")
            e = torch.zeros(B["nm"]).index_add(0, B["pm"], torch.exp(-(c - m[B["pm"]]) / tau))
            qm = m - tau * torch.log(e)
        first = torch.zeros(B["nm"], 3)
        sel = B["mrel"] > 0
        first[sel, B["mrel"][sel] - 1] = 1
        return qm + (first * w[B["msc"], NFP:]).sum(1)

    def logits(self, B, tau):
        q = self.q(B, tau)
        L = torch.full((B["n"], 4), -1e4)
        L[B["msc"], B["mdir"]] = -q / (0.2 * torch.exp(self.t))
        return L


def hard_pred(nets, B, order="SRLB"):
    with torch.no_grad():
        q = sum(n.q(B, None) for n in nets) / len(nets)
    q = q + torch.tensor([order.index(c) for c in RELS], dtype=torch.float32)[B["mrel"]] * 1e-5
    L = torch.full((B["n"], 4), float("inf")); L[B["msc"], B["mdir"]] = q
    return L.argmin(1)


def train_one(seed, B):
    torch.manual_seed(seed)
    net = Hyper(B["C"].shape[1]); opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=float(os.environ.get("WD", "0")))
    for st in range(STEPS):
        tau = max(0.05, 1.0 * (0.05 / 1.0) ** (st / (0.7 * STEPS)))
        loss = nn.functional.cross_entropy(net.logits(B, tau), B["y"])
        opt.zero_grad(); loss.backward(); opt.step()
    return net


def cat(A, Bb):
    off = A["nm"]; offs = A["n"]
    return dict(F=torch.cat([A["F"], Bb["F"]]), pm=torch.cat([A["pm"], Bb["pm"] + off]), msc=torch.cat([A["msc"], Bb["msc"] + offs]),
                mdir=torch.cat([A["mdir"], Bb["mdir"]]), mrel=torch.cat([A["mrel"], Bb["mrel"]]), C=torch.cat([A["C"], Bb["C"]]),
                y=torch.cat([A["y"], Bb["y"]]), n=A["n"] + Bb["n"], nm=A["nm"] + Bb["nm"])


if __name__ == "__main__":
    t0 = time.time()
    TR = prep("train"); VA = prep("validation")
    DATA = cat(TR, VA) if FINAL else TR
    nets = [train_one(s, DATA) for s in range(SEEDS)]
    orders = ["".join(p) for p in itertools.permutations(RELS)]
    acc = {o: (hard_pred(nets, DATA, o) == DATA["y"]).float().mean().item() for o in orders}
    o = max(orders, key=lambda k: acc[k])
    tr = (hard_pred(nets, TR, o) == TR["y"]).float().mean().item(); va = (hard_pred(nets, VA, o) == VA["y"]).float().mean().item()
    tag = os.environ.get("HTAG", "")
    torch.save([n.state_dict() for n in nets], f"cache/p2_hyper{tag}_r{r}{'_final' if FINAL else ''}.pt")
    print(f"R{r} hyper{tag} steps={STEPS} hid={HID} seeds={SEEDS} final={FINAL} order={o}: train {tr:.3f} validation {va:.3f} ({time.time() - t0:.0f}s)", flush=True)
