"""BỘ TÌM ĐƯỜNG KHẢ VI: học hàm chi phí của một robot bằng mạng (chi phí bước chuyển = MLP(đặc trưng bước, điều kiện cảnh)),
lặp giá trị "mềm" trên (giao lộ, hướng) tới đích (2 chặng nếu có điểm ghé), xác suất bước đầu = softmax(-Q / tau).
Học trên train, đo validation (thông tin đúng). Cần PYTHONPATH=F:/pylibs (torch CUDA).
    python scratch/p2_vin.py <robot> [epochs] [hidden]"""
import sys, time, numpy as np, torch, torch.nn as nn

r = int(sys.argv[1]); EP = int(sys.argv[2]) if len(sys.argv) > 2 else 60; HID = int(sys.argv[3]) if len(sys.argv) > 3 else 64
dev = "cuda"
torch.manual_seed(0)
BIG = 60.0; K = 48; TAU = 0.15


def load(split):
    z = np.load(f"cache/p2_vin_{split}.npz")
    leg = r == 4
    return dict(nxt=torch.tensor(z["nxtl" if leg else "nxt"].astype(np.int64)), E=torch.tensor(z["EL" if leg else "E"].astype(np.float32)),
                goal=torch.tensor(z["goal"]), via=torch.tensor(z["via"]), start=torch.tensor(z["start"].astype(np.int64)),
                C=torch.tensor(z["C"]), y=torch.tensor(z["y"][:, r].astype(np.int64)))


TR = load("train"); VA = load("validation")
NF = TR["E"].shape[-1]; NC = TR["C"].shape[-1]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(NF + NC, HID), nn.ReLU(), nn.Linear(HID, HID), nn.ReLU(), nn.Linear(HID, 1))
        self.out_t = nn.Parameter(torch.tensor(0.0))

    def cost(self, E, C):
        B = E.shape[0]
        x = torch.cat([E, C[:, None, None, :].expand(B, 325, 4, NC)], -1)
        return nn.functional.softplus(self.f(x).squeeze(-1)) + 0.05          # [B, 325, 4]

    def forward(self, b, hard=False):
        c = self.cost(b["E"], b["C"])
        B = c.shape[0]
        nxt = b["nxt"]

        def smin(q):
            return q.amin(-1) if hard else -TAU * torch.logsumexp(-q / TAU, -1)

        def leg(term):
            V = term.clone()
            for _ in range(K):
                q = c + V.gather(1, nxt.view(B, -1)).view(B, 325, 4)
                V = torch.minimum(term, smin(q))
                V = torch.cat([V[:, :324], torch.full((B, 1), 1e3, device=V.device)], 1)
            return V
        gs = b["goal"].repeat_interleave(4, 1)
        term = torch.where(gs, torch.zeros(B, 324, device=c.device), torch.full((B, 324), BIG, device=c.device))
        term = torch.cat([term, torch.full((B, 1), 1e3, device=c.device)], 1)
        V = leg(term)
        hv = b["via"].any(1)
        if hv.any():
            vs = b["via"].repeat_interleave(4, 1)
            t1 = torch.where(vs, V[:, :324], torch.full((B, 324), BIG, device=c.device))
            t1 = torch.cat([t1, torch.full((B, 1), 1e3, device=c.device)], 1)
            V1 = leg(t1)
            V = torch.where(hv[:, None], V1, V)
        st = b["start"]
        ns = nxt[torch.arange(B), st]                                    # [B, 4]
        q = c[torch.arange(B), st] + V.gather(1, ns)
        q = torch.where(ns == 324, torch.full_like(q, 1e3), q)
        return -q / (0.3 * torch.exp(self.out_t))


def batches(D, bs, shuffle=True):
    n = len(D["y"]); idx = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        j = idx[i:i + bs]
        yield {k: v[j].to(dev) for k, v in D.items()}


net = Net().to(dev)
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EP)


def evaluate(D):
    net.eval(); ok = 0
    with torch.no_grad():
        for b in batches(D, 150, False):
            ok += (net(b, hard=True).argmax(1) == b["y"]).sum().item()
    net.train()
    return ok / len(D["y"])


t0 = time.time()
best = (0, None)
for ep in range(EP):
    tot = 0
    for b in batches(TR, 125):
        loss = nn.functional.cross_entropy(net(b), b["y"])
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
    sched.step()
    if ep % 5 == 4 or ep == EP - 1:
        tr, va = evaluate(TR), evaluate(VA)
        if va > best[0]:
            best = (va, ep)
            torch.save(net.state_dict(), f"cache/p2_vin_r{r}.pt")
        print(f"R{r} ep {ep}: loss {tot:.2f} train {tr:.3f} validation {va:.3f}  ({time.time()-t0:.0f}s)", flush=True)
print(f"R{r} TỐT NHẤT validation {best[0]:.3f} (ep {best[1]})")
