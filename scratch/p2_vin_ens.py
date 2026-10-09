"""So nhóm mạng: nhỏ (hid 64) / lớn (hid 128) / gộp cả hai, trên validation (thông tin đúng).
    python scratch/p2_vin_ens.py <robot>"""
import sys, glob, torch
sys.argv = [sys.argv[0], sys.argv[1], "1", "64", "1"]
src = open("scratch/p2_vin2.py", encoding="utf-8").read().split("t0 = time.time()\nDATA")[0]
r = int(sys.argv[1])
exec(src)


def nets_of(tag, hid):
    global HID
    out = []
    for f in sorted(glob.glob(f"cache/p2_vin2{tag}_r{r}_s?.pt")):
        HID = hid
        n = Net().to(dev)
        n.f = nn.Sequential(nn.Linear(NF + NC + 1, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1)).to(dev)
        n.load_state_dict(torch.load(f, map_location=dev)); out.append(n)
    return out


small, big = nets_of("", 64), nets_of("big", 128)
for name, ns in (("nhỏ", small), ("lớn", big), ("gộp", small + big)):
    if ns:
        print(f"R{r} {name} ({len(ns)} mạng): validation {ens_eval(ns, VA):.3f}", flush=True)
