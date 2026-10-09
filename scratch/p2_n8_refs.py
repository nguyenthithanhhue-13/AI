"""Phép thử TỰ SOẠN cho THAM CHIẾU CHỌN BẢN (loại có 2 bản): nhiều cách nói tiếng Việt cho bắc / nam / tây / đông / gần / xa,
trong khung câu thường gặp, có dấu / không dấu. Đo: đích chốt đúng BẢN (vị trí). Bản đồ giả: đích 2 bản cách xa nhau, mốc 1 bản.
    python scratch/p2_n8_refs.py <file bộ đọc>"""
import sys, random, collections
sys.path.insert(0, "src")
from common import *
import nlp3
from strategy import resolve_targets
p = nlp3.MissionParser3.load(OUT / (sys.argv[1] if len(sys.argv) > 1 else "nlp3_trainonly.pkl"))
REF = {
    "north": ["{G} phía bắc", "{G} ở phía bắc", "{G} nằm phía bắc", "{G} phía trên", "{G} ở phía trên bản đồ", "{G} mạn trên",
              "{G} mạn bắc", "{G} ở góc trên", "{G} nằm ở nửa trên bản đồ", "{G} phía trên cùng", "{G} miền bắc", "{G} hướng bắc",
              "{G} bên trên", "{G} ở trên"],
    "south": ["{G} phía nam", "{G} ở phía dưới bản đồ", "{G} mạn dưới", "{G} nằm phía nam", "{G} ở nửa dưới bản đồ", "{G} hướng nam",
              "{G} bên dưới", "{G} ở dưới"],
    "west": ["{G} phía tây", "{G} bên trái", "{G} ở bên trái bản đồ", "{G} mạn trái", "{G} phía bên trái", "{G} hướng tây", "{G} ở nửa trái"],
    "east": ["{G} phía đông", "{G} bên phải", "{G} ở bên phải bản đồ", "{G} mạn phải", "{G} phía bên phải", "{G} hướng đông", "{G} ở nửa phải"],
    "near": ["{G} gần {A} hơn", "{G} nằm gần {A}", "{G} cạnh {A}", "{G} sát {A}", "{G} kế bên {A}", "{G} không xa {A}", "{G} gần {A}"],
    "far": ["{G} xa {A} hơn", "{G} nằm xa {A}", "{G} cách xa {A}", "{G} không gần {A}", "{G} ở xa {A}"],
}
NAMES = {"library": "thư viện", "dorm": "ký túc xá", "canteen": "căn tin", "clinic": "trạm y tế", "parking": "bãi xe",
         "lecture": "giảng đường", "lab": "phòng thí nghiệm", "office": "phòng hành chính", "gate": "cổng trường", "sports": "nhà thể thao"}
FR = ["Giao gói hàng tới {X} giúp mình.", "Có đơn giao hộp giấy ở {X}. Đừng nhầm với {D} nhé.", "Người nhận ở {X} đang chờ bưu kiện.",
      "Trước khi mang tập tài liệu tới {X}, nhớ ghé {V} lấy con dấu."]
random.seed(2)
types = list(NAMES)
c = collections.Counter(); bad = collections.Counter()
for kind, pats in REF.items():
    for pat in pats:
        for fr in FR:
            g, a, d, v = random.sample(types, 4)
            # bản đồ: đích 2 bản ở (0,0) và (6,6); mốc ở (0,1) -> bản gần mốc là (0,0)
            L = {g: [(0, 0), (6, 6)], a: [(0, 1)], d: [(3, 3)], v: [(5, 1)]}
            for t in types:
                L.setdefault(t, [(2, 4 + types.index(t) % 3)] if t not in (g, a, d, v) else L[t])
            want = {"north": (0, 0), "south": (6, 6), "west": (0, 0), "east": (6, 6), "near": (0, 0), "far": (6, 6)}[kind]
            x = pat.format(G=NAMES[g], A=NAMES[a])
            for txt in (fr.format(X=x, D=NAMES[d], V=NAMES[v]), nlp3._unaccent(fr.format(X=x, D=NAMES[d], V=NAMES[v]))):
                world = {"landmarks": L, "adj": {}, "robot": (4, 4), "heading": 0, "rain": False}
                r = nlp3.resolve3(p.parse(txt, set(L), L), L)
                got = resolve_targets(world, r["goal"], r["goal_ref"]) if r["goal"] == g else []
                ok = got == [want]
                c[kind, "n"] += 1; c[kind, "ok"] += ok
                if not ok and not txt.isascii():
                    bad[(kind, pat)] += 1
print({k: f"{c[k, 'ok']}/{c[k, 'n']}" for k in REF})
for (k, pat), n in sorted(bad.items(), key=lambda z: -z[1])[:25]:
    print(f"   SAI {k:6s} {pat}  x{n}")
