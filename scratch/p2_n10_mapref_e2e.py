"""Phép thử TỰ SOẠN vòng 3 (độc lập với p2_n9 dùng để chỉnh luật), đo END-TO-END: câu mô tả nơi giao qua bản đồ bằng cách nói
mới, chèn vào câu nhiều vế, chạy bộ đọc nlp3 trên BẢN ĐỒ THẬT của validation (thông tin đúng) -> đích chốt đúng VỊ TRÍ không.
    python scratch/p2_n10_mapref_e2e.py [file bộ đọc]"""
import sys, random, collections
sys.path.insert(0, "src")
from common import *
import nlp3, mapref
p = nlp3.MissionParser3.load(OUT / (sys.argv[1] if len(sys.argv) > 1 else "nlp3_trainonly.pkl"))

DIRS = {"north_most": ["bắc", "trên"], "south_most": ["nam", "dưới"], "west_most": ["tây", "trái"], "east_most": ["đông", "phải"]}
MOST = ["điểm nằm xa nhất ở hướng {c}", "địa điểm ở rìa {c} nhất", "nơi nằm tít về phía {c}", "chỗ ở cực {c}",
        "địa điểm xa về phía {c} hơn mọi nơi khác", "tòa nhà nằm sát mép {s} nhất", "nơi ở ngoài cùng phía {c} của bản đồ",
        "địa điểm sát cạnh {s} của bản đồ nhất", "địa điểm có vị trí xa nhất về phía {c}", "nơi gần mép {s} nhất",
        "công trình ở phía {c} cùng", "chỗ nằm ở mạn {c} xa nhất", "địa điểm ở phía {c} xa nhất trên bản đồ",
        "nơi nằm về hướng {c} nhiều nhất", "chỗ nằm ở tận cùng hướng {c}"]
ANCH = ["chỗ gần nhất với {a}", "địa điểm nằm sát {a} nhất", "nơi gần {a} hơn hết", "nơi cách {a} ngắn nhất",
        "địa điểm nằm gần nhất so với {a}", "địa điểm sát vách {a} nhất", "chỗ gần kề với {a} nhất",
        "tòa nhà gần {a} nhất trên bản đồ", "địa điểm lân cận gần nhất của {a}", "nơi ở ngay bên cạnh {a} nhất",
        "vị trí gần {a} nhất", "địa điểm nằm cạnh {a} gần nhất", "khu vực gần {a} nhất"]
NAMES = {"library": ["thư viện", "nơi mượn sách"], "dorm": ["ký túc xá", "KTX"], "canteen": ["căn tin", "nhà ăn"],
         "clinic": ["phòng y tế", "trạm y tế"], "parking": ["bãi gửi xe", "nhà xe"], "lecture": ["giảng đường", "phòng học lớn"],
         "lab": ["phòng thí nghiệm"], "office": ["phòng hành chính", "phòng đào tạo"], "gate": ["cổng trường", "cổng chính"],
         "sports": ["nhà thi đấu", "sân thể thao"]}
FR = ["Xin chào! Hàng dễ vỡ, đi cẩn thận. Giao gói hàng đến {X} giúp mình. Không cần ghé {D}. Cảm ơn!",
      "Robot ơi, người nhận ở {X} đang chờ bưu kiện. Đừng nhầm với {D} nhé. Gấp nhé!",
      "Nhắn robot: Lấy chìa khóa ở {V} xong thì vận chuyển hộp giấy đến {X}. Cứ từ từ, không vội.",
      "Yêu cầu mới: Không giao {D} nữa nhé, đổi lại: Đích đến là {X}, hàng cần giao là tập tài liệu. Cảm ơn nhiều.",
      "Chào robot, (tin trước ghi {D} là nhầm) Hàng cho {X}: túi đồ. Hôm qua đã giao ở {V} rồi."]
random.seed(7)
scenes = load_split("validation")[2]
c = collections.Counter(); bad = collections.Counter(); n = 0
for s in scenes:
    L = {t: [tuple(q) for q in v] for t, v in landmarks_of(s).items()} if "landmarks_of" in globals() else None
    if L is None:
        L = collections.defaultdict(list)
        for lm in s["landmarks"]:
            L[lm["type"]].append(tuple(lm["rc"]))
        L = dict(L)
    types = list(L)
    one = [t for t in types if len(L[t]) == 1]
    for _ in range(3):
        kind = random.choice(list(DIRS) + ["anchor_near"] * 2)
        if kind == "anchor_near":
            if not one:
                continue
            a = random.choice(one)
            want = mapref.resolve(kind, L, a); pat = random.choice(ANCH)
            x = pat.format(a=random.choice(NAMES[a]))
        else:
            want = mapref.resolve(kind, L); pat = random.choice(MOST)
            cc, ss = DIRS[kind]; x = pat.format(c=cc, s=ss)
        d, v = random.sample(types, 2)
        txt = random.choice(FR).format(X=x, D=random.choice(NAMES[d]), V=random.choice(NAMES[v]))
        for t in (txt, nlp3._unaccent(txt)):
            r = p.parse(t, set(L), L)
            ok = r.get("goal_ref") == ("pos", want)
            c[kind, "n"] += 1; c[kind, "ok"] += ok
            if not ok:
                bad[pat] += 1
                if bad[pat] <= 1:
                    print("   SAI", kind, "|", t, "->", r.get("goal"), r.get("goal_ref"), r.get("mapref"))
tot = sum(c[k, "ok"] for k in list(DIRS) + ["anchor_near"]); N = sum(c[k, "n"] for k in list(DIRS) + ["anchor_near"])
print({k: f"{c[k, 'ok']}/{c[k, 'n']}" for k in list(DIRS) + ["anchor_near"]}, f"tổng {tot}/{N} = {tot / N:.3f}")
for pat, k in bad.most_common(15):
    print(f"  x{k:3d} {pat}")
