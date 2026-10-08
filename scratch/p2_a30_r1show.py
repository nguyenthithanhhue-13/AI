import sys
sys.path.insert(0, "scratch")
from p2_lib import *
from p2_draw import draw
from p2_a20_dagperc import bfs_dist
D = load("train"); r = 1; k = 0
for i, x in enumerate(D):
    w = x["w"]; s = w["robot"]
    if len(x["legs"]) != 1: continue
    V = bfs_dist(w, x["legs"][0], False)
    y = x["y"][r]; n2 = (s[0] + DRC[y][0], s[1] + DRC[y][1])
    if V.get(n2, 99) < V.get(s, 99): continue
    k += 1
    if k > 4: break
    print(f"#{i} hướng mũi {ACTIONS[w['heading']]} mưa={w['rain']} gấp={x['m']['urgent']} dễ vỡ={x['m']['fragile']} R1={ACTIONS[y]}  10 robot {''.join(ACTIONS[v][0] for v in x['y'])}  hop từ robot={V.get(s)}")
    print("   ", x["m"]["text"][:200])
    print(draw(x))
