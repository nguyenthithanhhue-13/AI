"""Vẽ bản đồ một cảnh (thông tin đúng) dạng chữ: giao lộ, địa điểm, robot, trạng thái đoạn đường (= thường, # đông, ~ mái che,
x đóng, s bậc thang, > < ^ v một chiều), nhãn 10 robot. python scratch/p2_ascii.py <scene_id> ..."""
import sys
sys.path.insert(0, "src")
from common import *
AB = {"library": "TV", "dorm": "KT", "sports": "TT", "clinic": "YT", "canteen": "CT", "parking": "BX", "lecture": "GĐ", "lab": "TN",
      "office": "HC", "gate": "CG"}
for sid in sys.argv[1:]:
    sp = sid.split("-")[0]
    rows, labels, scenes = load_split(sp)
    i = [s["scene_id"] for s in scenes].index(sid); s = scenes[i]
    R, C = s["grid"]["rows"], s["grid"]["cols"]
    lm = {tuple(l["rc"]): AB[l["type"]] for l in s["landmarks"]}
    E = {}
    for e in s["edges"]:
        a, b = tuple(e["a"]), tuple(e["b"])
        ch = {"normal": "=", "crowded": "#", "covered": "~", "closed": "x", "missing": " "}.get(e["status"], "?")
        if e["stairs"]:
            ch = "s"
        if e["oneway_to"]:
            t = tuple(e["oneway_to"]); src = a if t == b else b
            ch = {(0, 1): ">", (0, -1): "<", (1, 0): "v", (-1, 0): "^"}[(t[0] - src[0], t[1] - src[1])]
        E[frozenset([a, b])] = ch
    rb = tuple(s["robot"]["rc"]); hd = s["robot"]["heading"]
    m = s["mission"]
    print(f"== {sid} | robot {rb} mũi {hd} | mưa={s['weather']} kiểu={s['style']} | {m['text'][:150]}")
    print(f"   goal={m['goal']} ref={m['goal_ref']} via={m['via']} gấp={m['urgent']} vỡ={m['fragile']} | nhãn (UDLR): " +
          " ".join(f"R{r}={'UDLR'[labels[i * 10 + r]]}" for r in range(10)))
    for r in range(R):
        line = ""; below = ""
        for c in range(C):
            n = (r, c)
            cell = "@@" if n == rb else lm.get(n, "+ ")
            line += cell
            h = E.get(frozenset([n, (r, c + 1)]), " ")
            line += (h * 3) if c < C - 1 else ""
            v = E.get(frozenset([n, (r + 1, c)]), " ")
            below += v + "    "
        print("   " + line)
        if r < R - 1:
            print("   " + below)
