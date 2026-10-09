"""Phép thử TỰ SOẠN: mô tả qua bản đồ KHÔNG có "nhất" (hoặc "nhất" gõ hỏng nặng), end-to-end trên bản đồ validation.
    python scratch/p2_n11_nonhat.py"""
import sys, random, collections
sys.path.insert(0, "src")
from common import *
import nlp3, mapref
p = nlp3.MissionParser3.load(OUT / "nlp3_trainonly.pkl")
DIRS = {"north_most": ["bắc", "trên"], "south_most": ["nam", "dưới"], "west_most": ["tây", "trái"], "east_most": ["đông", "phải"]}
MOST = ["địa điểm ở phía {c} bản đồ", "chỗ nằm ở mép {s} bản đồ", "nơi ở rìa {c}", "địa điểm nằm tít phía {c}",
        "vị trí ở phía {c} ngoài rìa", "chỗ ở mạn {c} bản đồ", "nơi nằm về phía {c} nhất", "địa điểm ở phía {c} nhât"]
ANCH = ["chỗ ngay cạnh {a}", "địa điểm sát vách {a}", "nơi kế bên {a}", "địa điểm nằm gần {a}", "chỗ ở sát {a}",
        "vị trí liền kề {a}", "địa điểm giáp {a}", "nơi gần {a} nht", "chỗ nằm cạnh {a} nhứt"]
NAMES = {"library": ["thư viện"], "dorm": ["ký túc xá"], "canteen": ["căn tin", "nhà ăn"], "clinic": ["phòng y tế"],
         "parking": ["bãi gửi xe"], "lecture": ["giảng đường"], "lab": ["phòng thí nghiệm"], "office": ["phòng đào tạo"],
         "gate": ["cổng trường"], "sports": ["nhà thi đấu"]}
FR = ["Giao gói hàng đến {X} giúp mình. Không cần ghé {D}.", "Người nhận ở {X} đang chờ bưu kiện. Gấp nhé!",
      "Lấy chìa khóa ở {V} xong thì vận chuyển hộp giấy đến {X}. Cứ từ từ.", "Đích đến là {X}, hàng cần giao là tập tài liệu."]
random.seed(3)
c = collections.Counter(); bad = collections.Counter()
for s in load_split("validation")[2]:
    L = collections.defaultdict(list)
    for lm in s["landmarks"]:
        L[lm["type"]].append(tuple(lm["rc"]))
    L = dict(L); types = list(L); one = [t for t in types if len(L[t]) == 1]
    for _ in range(2):
        kind = random.choice(list(DIRS) + (["anchor_near"] * 2 if one else []))
        if kind == "anchor_near":
            a = random.choice(one); want = mapref.resolve(kind, L, a); pat = random.choice(ANCH); x = pat.format(a=random.choice(NAMES[a]))
        else:
            want = mapref.resolve(kind, L); pat = random.choice(MOST); x = pat.format(c=DIRS[kind][0], s=DIRS[kind][1])
        d, v = random.sample(types, 2)
        txt = random.choice(FR).format(X=x, D=random.choice(NAMES[d]), V=random.choice(NAMES[v]))
        for t in (txt, nlp3._unaccent(txt)):
            r = p.parse(t, set(L), L)
            ok = r.get("goal_ref") == ("pos", want); c["n"] += 1; c["ok"] += ok
            if not ok:
                bad[pat] += 1
                if bad[pat] == 1:
                    print("   SAI", kind, "|", t, "->", r.get("goal"), r.get("goal_ref"), r.get("mapref"))
print(f"đúng {c['ok']}/{c['n']} = {c['ok'] / c['n']:.3f}")
for pat, k in bad.most_common(10):
    print(f"  x{k:3d} {pat}")
