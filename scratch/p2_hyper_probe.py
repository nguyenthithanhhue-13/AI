"""MỔ mô hình siêu tuyến tính (bản cờ + loại đích, chỉ học train): trọng số theo từng loại đích × (mưa / đêm), chuẩn hóa theo trọng
số "số đoạn" -> đọc ra loại đích nào giống nhau.
    python scratch/p2_hyper_probe.py <robot> [tag=goal] [cờ cố định, vd urg=0,frag=0,via=0]"""
import sys, os, numpy as np, torch
r = int(sys.argv[1]); HT = sys.argv[2] if len(sys.argv) > 2 else "goal"
os.environ["COND"] = {"goal": "goal", "": "full"}.get(HT, HT)
sys.argv = ["x", str(r)]
h = {"__name__": "x"}
exec(open("scratch/p2_hyper.py", encoding="utf-8").read().split('\nif __name__ == "__main__":\n')[0], h)
nc = 16 if HT == "goal" else 26
nets = []
for sd in torch.load(f"cache/p2_hyper{HT}_r{r}.pt"):
    m = h["Hyper"](nc); m.load_state_dict(sd); nets.append(m.eval())
PL = h["PL"]
NAMES = ["hop", "crowd", "cover", "lm", "stairs", "tR", "tL", "tB", "unc"] + [p[:4] for p in PL] + ["R0", "L0", "B0"]
show = [0, 1, 2, 8, 3, 5, 6, 7] + list(range(9, 19)) + [19, 20, 21]
for rain in (0, 1):
    for night in (0, 1):
        print(f"--- mưa={rain} đêm={night} (chia cho trọng số số đoạn)")
        print("đích     | " + " ".join(f"{NAMES[k]:>5s}" for k in show))
        for gi, gt in enumerate(PL):
            c = [rain, night, 0, 0, 0, 0] + [int(k == gi) for k in range(10)] + ([0] * 10 if nc == 26 else [])
            with torch.no_grad():
                w = sum(torch.nn.functional.softplus(n.f(torch.tensor([c], dtype=torch.float32)))[0] for n in nets).numpy() / len(nets)
            print(f"{gt:8s} | " + " ".join(f"{w[k] / w[0]:5.2f}" for k in show))
