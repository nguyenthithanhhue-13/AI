"""GỘP mô hình siêu tuyến tính (p2_hyper) + bộ tìm đường học được (p2_vin2) trên validation (thông tin đúng):
log-xác suất từng bên (chuẩn hóa softmax) cộng có trọng số alpha. In độ chính xác từng bên và khi gộp.
    python scratch/p2_hv_ens.py <robot> <vin tags, vd ",big"> [hyper tag]"""
import sys, glob, os, numpy as np, torch
r = int(sys.argv[1]); VT = sys.argv[2].split(",") if len(sys.argv) > 2 else [""]; HT = sys.argv[3] if len(sys.argv) > 3 else ""
argv0 = sys.argv[0]
# --- bộ tìm đường học được
sys.argv = [argv0, str(r), "1", "64"]
g = {}
exec(open("scratch/p2_vin2.py", encoding="utf-8").read().split("t0 = time.time()")[0], g)
vnets = []
for tag in VT:
    g["HID"] = 128 if tag == "big" else 64
    for f in sorted(glob.glob(f"cache/p2_vin2{tag}_r{r}_s[0-9].pt")):
        n = g["Net"]().to(g["dev"]); n.load_state_dict(torch.load(f)); vnets.append(n.eval())
VA = g["VA"]
with torch.no_grad():
    lv = []
    for b in g["batches"](VA, 150, False):
        lv.append(sum(torch.log_softmax(n(b, hard=True), 1) for n in vnets).cpu() / len(vnets))
    LV = torch.cat(lv).numpy()
yv = VA["y"].numpy()
# --- siêu tuyến tính (nhiều tag, cách nhau bởi "+": "" = đầy đủ, goal, basic)
LHs = []
for ht in HT.split("+"):
    sys.argv = [argv0, str(r)]
    os.environ["COND"] = {"basic": "basic", "goal": "goal"}.get(ht, "full")
    h = {"__name__": "x"}
    exec(open("scratch/p2_hyper.py", encoding="utf-8").read().split('if __name__ == "__main__":')[0], h)
    HV = h["prep"]("validation")
    hn = []
    for sd in torch.load(f"cache/p2_hyper{ht}_r{r}.pt"):
        m = h["Hyper"](HV["C"].shape[1]); m.load_state_dict(sd); hn.append(m.eval())
    with torch.no_grad():
        L = sum(torch.log_softmax(m.logits(HV, None), 1) for m in hn).numpy() / len(hn)
    LHs.append(np.where(L < -1e3, -1e4, L))
    print(f"   hyper[{ht or 'full'}] {(LHs[-1].argmax(1) == yv).mean():.3f}")
LH = sum(LHs) / len(LHs)
assert (HV["y"].numpy() == yv).all()
acc = lambda L: (L.argmax(1) == yv).mean()
print(f"R{r} [{HT}]: vin {acc(LV):.3f} | hyper gộp {acc(LH):.3f} | " + " ".join(f"a={a}: {acc(a * LH + (1 - a) * LV):.3f}" for a in (0.3, 0.5, 0.67, 0.8)))
