"""Lỗi validation của mô hình siêu tuyến tính (gộp các bộ đã học) theo TỔ HỢP điều kiện (đêm, mưa, gấp, dễ vỡ, ghé) và số cảnh train
cùng tổ hợp -> lỗi có dồn vào tổ hợp hiếm không.
    python scratch/p2_hyper_grperr.py <robot> [tags +goal+sel]"""
import sys, os, collections, numpy as np, torch
r = int(sys.argv[1]); TAGS = (sys.argv[2] if len(sys.argv) > 2 else "+goal+sel").split("+")
SEL = {1: "rain+night+urg", 2: "rain+night+goal", 3: "rain+goal", 4: "rain+urg+frag+goal", 5: "night",
       6: "night+urg+frag+via+mapg+goal", 7: "rain+night+urg+frag+goal", 8: "night+urg+frag+via+goal"}
L = 0
for t in TAGS:
    os.environ["COND"] = SEL[r] if t == "sel" else {"goal": "goal"}.get(t, "full")
    sys.argv = ["x", str(r)]
    h = {"__name__": "x"}
    exec(open("scratch/p2_hyper.py", encoding="utf-8").read().split('\nif __name__ == "__main__":\n')[0], h)
    VA = h["prep"]("validation")
    for sd in torch.load(f"cache/p2_hyper{t}_r{r}.pt"):
        m = h["Hyper"](VA["C"].shape[1]); m.load_state_dict(sd)
        with torch.no_grad():
            L = L + torch.log_softmax(m.eval().logits(VA, None), 1).numpy()
pred = L.argmax(1); y = VA["y"].numpy()
key = lambda x: (int(x["s"]["style"] == "night"), int(x["w"]["rain"]), int(x["m"]["urgent"]), int(x["m"]["fragile"]), int(bool(x["m"]["via"])))
DT = h["load"]("train"); DV = h["load"]("validation")
nt = collections.Counter(key(x) for x in DT)
st = collections.defaultdict(lambda: [0, 0])
for x, p, yy in zip(DV, pred, y):
    st[key(x)][0] += 1; st[key(x)][1] += p == yy
print(f"R{r} validation {np.mean(pred == y):.3f}. (đêm, mưa, gấp, vỡ, ghé): n_train | val đúng/n")
bins = collections.defaultdict(lambda: [0, 0])
for k in sorted(st, key=lambda k: nt[k]):
    n, ok = st[k]
    b = "<40" if nt[k] < 40 else "40-80" if nt[k] < 80 else ">=80"
    bins[b][0] += n; bins[b][1] += ok
print("   theo cỡ nhóm train:", {b: f"{v[1]}/{v[0]} = {v[1] / max(1, v[0]):.3f}" for b, v in bins.items()})
