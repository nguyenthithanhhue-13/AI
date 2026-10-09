"""Xuất mô hình siêu tuyến tính (torch, cache/p2_hyper<tag>_r<r>[_final].pt) sang outputs/hyper_<mode>.npz cho src/hyper.py.
    python scratch/p2_hyper_export.py trainonly|final r:cond:order[:tag] ...   ví dụ 3:full:SRLB 6:basic:SRLB:basic"""
import sys, numpy as np, torch
mode = sys.argv[1]
OUTTAG = __import__("os").environ.get("OUTTAG", "")     # outputs/hyper<OUTTAG>_<mode>.npz
arrs, meta = {}, {}
for spec in sys.argv[2:]:
    parts = spec.split(":")
    r, cond, order = int(parts[0]), parts[1], parts[2]
    tag = parts[3] if len(parts) > 3 else ""
    sds = torch.load(f"cache/p2_hyper{tag}_r{r}{'_final' if mode == 'final' else ''}.pt")
    for s, sd in enumerate(sds):
        for k, src in (("w0", "f.0.weight"), ("b0", "f.0.bias"), ("w1", "f.2.weight"), ("b1", "f.2.bias"), ("w2", "f.4.weight"), ("b2", "f.4.bias")):
            arrs[f"r{r}_s{s}_{k}"] = sd[src].numpy().astype(np.float64)
        arrs[f"r{r}_s{s}_t"] = sd["t"].numpy().astype(np.float64)
    meta[r] = (cond, order, len(sds))
np.savez(f"outputs/hyper{OUTTAG}_{mode}.npz", meta=np.array(meta, dtype=object), **arrs)
print("đã ghi", f"outputs/hyper{OUTTAG}_{mode}.npz", meta)
