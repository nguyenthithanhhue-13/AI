"""In vài ca hòa của một robot: bản đồ ASCII, hướng mũi, đích, các bước hòa, nhãn của cả 10 robot."""
import sys, collections
sys.path.insert(0, "scratch")
from p2_lib import *
SYM = {"normal": "-", "crowded": "=", "covered": "~", "closed": " "}
def draw(w, s, marks):
    R = max(n["rc"][0] for n in s["nodes"]) + 1; C = max(n["rc"][1] for n in s["nodes"]) + 1
    nodes = {tuple(n["rc"]) for n in s["nodes"]}
    lines = []
    for i in range(R):
        a = ""; b = ""
        for j in range(C):
            ch = marks.get((i, j), "+" if (i, j) in nodes else " ")
            a += ch
            e = w["adj"].get((i, j), {}).get(3)
            a += (SYM[e[0]] if e[2] else ">") if e and e[0] != "closed" else (" " if not e else " ")
            if e and e[1]: a = a[:-1] + "s"
            e2 = w["adj"].get((i, j), {}).get(1)
            b += ("|" if e2[0] == "normal" else "H" if e2[0] == "crowded" else ":" if e2[0] == "covered" else " ") if e2 else " "
            if e2 and e2[1]: b = b[:-1] + "s"
            b += " "
        lines.append(a); lines.append(b)
    return "\n".join(lines)
if __name__ == "__main__":
    r = int(sys.argv[1]); K = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    single, multi = pickle.load(open(f"cache/p2_tiesplit_r{r}.pkl", "rb"))
    for x, a, y, reach in single[:K]:
        w = x["w"]; s = x["s"]
        common = sorted(set.intersection(*[reach[d] for d in a]))[0]
        marks = {w["robot"]: "R"}
        for k, c in enumerate(common): marks[c] = "V" if k < len(common) - 1 else "G"
        print(f"heading {ACTIONS[w['heading']]}  hòa {[ACTIONS[d] for d in a]}  nhãn R{r}={ACTIONS[y]}  rain={w['rain']} urgent={x['m']['urgent']} fragile={x['m']['fragile']}")
        print("  nhãn 10 robot:", [ACTIONS[v][0] for v in x["y"]])
        print(draw(w, s, marks)); print()
