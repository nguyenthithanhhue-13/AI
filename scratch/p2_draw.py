"""Vẽ bản đồ ASCII đầy đủ: giao lộ (RB robot, chữ hoa = địa điểm, chữ thường = đích / ghé), đoạn ngang 3 ký tự
(trạng thái -=~, chiều một chiều > <, bậc thang s), đoạn dọc (trạng thái |H:, chiều v ^, bậc thang s)."""
from p2_lib import *

ST_H = {"normal": "-", "crowded": "=", "covered": "~"}
ST_V = {"normal": "|", "crowded": "H", "covered": ":"}


def draw(x):
    w = x["w"]; s = x["s"]
    lm = {p: k[:2].upper() for k, v in w["landmarks"].items() for p in v}
    tg = {p for L in x["legs"] for p in L}
    R = max(n["rc"][0] for n in s["nodes"]) + 1; C = max(n["rc"][1] for n in s["nodes"]) + 1
    out = ["     " + "".join(f"{c:<5d}" for c in range(C))]
    for r in range(R):
        a = f"{r:3d}  "; b = "     "
        for c in range(C):
            p = (r, c)
            mk = "RB" if p == w["robot"] else (lm[p] if p in lm else ("++" if p in w["adj"] else "  "))
            if p in tg and p != w["robot"]:
                mk = mk.lower()
            a += mk
            e = w["adj"].get(p, {}).get(3)
            if e and e[0] != "closed":
                back = w["adj"].get((r, c + 1), {}).get(2)
                ow = ">" if (back and not back[2]) else ("<" if not e[2] else ST_H[e[0]])
                a += ST_H[e[0]] + ow + ("s" if e[1] else ST_H[e[0]])
            else:
                a += "   "
            e2 = w["adj"].get(p, {}).get(1)
            if e2 and e2[0] != "closed":
                back = w["adj"].get((r + 1, c), {}).get(0)
                ow = "v" if (back and not back[2]) else ("^" if not e2[2] else " ")
                b += ST_V[e2[0]] + ow + ("s" if e2[1] else " ") + "  "
            else:
                b += "     "
        out.append(a); out.append(b)
    return "\n".join(out)
