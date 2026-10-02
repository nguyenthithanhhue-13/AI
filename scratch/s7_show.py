"""In bản đồ dạng chữ của một cảnh train để soi bằng mắt."""
import sys
sys.path.insert(0, "src")
from common import *

rows, labels, scenes = load_split("train")


def show(i):
    s = scenes[i]
    w = world_from_scene(s)
    R, C = s["grid"]["rows"], s["grid"]["cols"]
    nodes = {tuple(n["rc"]) for n in s["nodes"]}
    lm = {tuple(l["rc"]): l["type"][:3] for l in s["landmarks"]}
    sym = {"normal": "n", "crowded": "C", "covered": "v", "closed": "X"}
    print(f"scene {i} robot {w['robot']} {ACTIONS[w['heading']]} rain={w['rain']} mission={s['mission']['text']}")
    print("labels", [ACTIONS[y] for y in labels[i * 10:(i + 1) * 10]])
    for r in range(R):
        line = ""; below = ""
        for c in range(C):
            p = (r, c)
            cell = " . " if p not in nodes else ("[R]" if p == w["robot"] else lm.get(p, " o "))
            line += f"{cell:3s}"
            e = w["adj"].get(p, {}).get(3)
            if e:
                t = sym[e[0]] + ("s" if e[1] else "") + ("" if e[2] else "<")
                back = w["adj"][(r, c + 1)][2]
                if not back[2]: t += ">"
                line += f"-{t:3s}-"
            else:
                line += "     "
            e = w["adj"].get(p, {}).get(1)
            if e:
                t = sym[e[0]] + ("s" if e[1] else "") + ("" if e[2] else "^")
                back = w["adj"][(r + 1, c)][0]
                if not back[2]: t += "v"
                below += f"{t:3s}     "
            else:
                below += "        "
        print(line); print(below)


for i in map(int, sys.argv[1:]):
    show(i)
