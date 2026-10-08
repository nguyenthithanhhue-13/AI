"""MỨC ĐỘ khẩn cấp / dễ vỡ theo CỤM TỪ trong câu (train): độ đúng của cấu hình hiện tại cho từng cụm.
    python scratch/p2_a57_levels.py <robot> urgent|fragile"""
import sys, json, re, collections
sys.path.insert(0, "scratch"); sys.path.insert(0, "src")
from p2_fast import *
from strat_ml import StrategyML
from mapref import unaccent

r = int(sys.argv[1]); kind = sys.argv[2]
H = json.load(open("outputs/strategy_hybrid.json", encoding="utf-8"))
st = StrategyML({}, H)
URG = ["hoa toc", "can ngay trong \\d+ phut", "can ngay", "khan cap", "gap nhe", "dang can gap lam", "cang nhanh cang tot",
       "khong duoc cham tre", "uu tien so mot", "di ngay", "khong gap dau", "cu tu tu", "khong voi", "chieu nay giao cung duoc",
       "mai giao cung kip", "khong can voi", "tu tu"]
FRA = ["hang de vo", "do thuy tinh", "do gom", "de hong khi va dap", "nhe tay", "rat de vo", "mong manh", "ky va dap",
       "chac chan", "khong de vo", "khong vo duoc", "roi cung chang sao", "hang ben", "khong lo vo"]
PH = URG if kind == "urgent" else FRA
c = collections.Counter()
for x in load("train"):
    t = unaccent(x["m"]["text"])
    hit = [p for p in PH if re.search(p, t)]
    key = hit[0] if hit else "(không cụm nào)"
    cfg = H[str(r)]
    night = x["s"]["style"] == "night"
    d = getattr(st, "_cost_move")(x["w"], x["legs"], r, cfg, night, x["m"]["urgent"], x["m"]["fragile"]) if cfg["method"] == "cost" else None
    c[(key, x["m"][kind], "n")] += 1; c[(key, x["m"][kind], "ok")] += d == x["y"][r]
for (k, v, t), n in sorted(c.items()):
    if t == "n":
        print(f"R{r} {k:28s} nhãn {kind}={v!s:5s} n={n:4d}  đúng {c[(k, v, 'ok')] / n:.2f}")
