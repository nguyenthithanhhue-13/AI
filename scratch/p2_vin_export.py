"""Xuất trọng số bộ tìm đường học được (torch .pt) sang npz cho src/vin.py, mỗi robot chọn nhóm mạng riêng.
    python scratch/p2_vin_export.py trainonly|final 1:,big 2: 4:,big 6: 8:big   (r:tag1,tag2 ; tag rỗng = mạng nhỏ hid 64)
    -> outputs/vin_<mode>.npz"""
import sys, glob, numpy as np, torch
mode = sys.argv[1]
out = {}
for spec in sys.argv[2:]:
    r, tags = spec.split(":")
    r = int(r); i = 0
    for tag in tags.split(","):
        pat = f"cache/p2_vin2{tag}_r{r}_s?{'_final' if mode == 'final' else ''}.pt"
        for f in sorted(glob.glob(pat)):
            sd = torch.load(f, map_location="cpu")
            out[f"r{r}_s{i}_w0"] = sd["f.0.weight"].numpy(); out[f"r{r}_s{i}_b0"] = sd["f.0.bias"].numpy()
            out[f"r{r}_s{i}_w1"] = sd["f.2.weight"].numpy(); out[f"r{r}_s{i}_b1"] = sd["f.2.bias"].numpy()
            out[f"r{r}_s{i}_w2"] = sd["f.4.weight"].numpy(); out[f"r{r}_s{i}_b2"] = sd["f.4.bias"].numpy()
            out[f"r{r}_s{i}_out_t"] = sd["out_t"].numpy()
            i += 1
    print("R", r, i, "mạng")
np.savez(f"outputs/vin_{mode}.npz", **out)
