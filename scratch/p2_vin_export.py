"""Xuất trọng số bộ tìm đường học được (torch .pt) sang npz cho src/vin.py.
    python scratch/p2_vin_export.py trainonly|final 1 2 6 8 ...  -> outputs/vin_<mode>.npz"""
import sys, glob, numpy as np, torch
mode = sys.argv[1]; ROB = [int(a) for a in sys.argv[2:]]
out = {}
for r in ROB:
    pat = f"cache/p2_vin2_r{r}_s*{'_final' if mode == 'final' else ''}.pt"
    files = sorted(f for f in glob.glob(pat) if (mode == "final") == f.endswith("_final.pt"))
    for i, f in enumerate(files):
        sd = torch.load(f, map_location="cpu")
        out[f"r{r}_s{i}_w0"] = sd["f.0.weight"].numpy(); out[f"r{r}_s{i}_b0"] = sd["f.0.bias"].numpy()
        out[f"r{r}_s{i}_w1"] = sd["f.2.weight"].numpy(); out[f"r{r}_s{i}_b1"] = sd["f.2.bias"].numpy()
        out[f"r{r}_s{i}_w2"] = sd["f.4.weight"].numpy(); out[f"r{r}_s{i}_b2"] = sd["f.4.bias"].numpy()
        out[f"r{r}_s{i}_out_t"] = sd["out_t"].numpy()
    print("R", r, len(files), "mạng")
np.savez(f"outputs/vin_{mode}.npz", **out)
