"""Độ đúng của mô hình hiện tại theo MÓN HÀNG nêu trong câu (từ khóa phổ biến trong train)."""
import sys, json, collections, re
sys.path.insert(0, "scratch"); sys.path.insert(0, "src")
from p2_fast import *
from strat_ml import group_key
from mapref import unaccent
ITEMS = ["tai lieu", "hop giay", "buu kien", "tui do", "thung hang", "goi hang", "phong bi", "ho so", "sach", "khay com", "nuoc",
         "thuoc", "khau trang", "bang gac", "linh kien", "hoa chat", "mau vat", "bong", "vot", "micro", "may chieu", "con dau", "the", "bien ten"]
r = int(sys.argv[1]); key = sys.argv[2]
P = json.load(open(f"cache/p2_cd3_r{r}_{key}.json", encoding="utf-8"))
c = collections.Counter()
for x in load("train"):
    w = x["w"]; p = P.get(group_key(key, x["s"]["style"] == "night", w["rain"], x["m"]["urgent"], x["m"]["fragile"], bool(x["m"]["via"])))
    a = argmins(SG(w, r == 4, lm_excl(x)).q(theta_of(**p), x["legs"]), 1e-6)
    pred = min(a, key=lambda d: "SRLB".index(REL[w["heading"]][d])) if a else None
    t = unaccent(x["m"]["text"])
    for it in ITEMS:
        if re.search(r"\b" + it + r"\b", t):
            for wc in ("mưa" if w["rain"] else "khô",):
                c[(it, wc, "n")] += 1; c[(it, wc, "ok")] += pred == x["y"][r]
for it in ITEMS:
    s = "  ".join(f"{wc}: {c[(it, wc, 'ok')]}/{c[(it, wc, 'n')]}={c[(it, wc, 'ok')]/max(1,c[(it, wc, 'n')]):.2f}" for wc in ("mưa", "khô") if c[(it, wc, 'n')])
    if s: print(f"R{r} {it:12s} {s}")
