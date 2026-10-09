"""Phép thử TỰ SOẠN vòng 2 cho bộ nhận diện mô tả qua bản đồ: các cách nói CHƯA có trong p2_n4 (trật tự "gần nhất với X",
"X ... nhất" đảo, đầu ngữ khác, "mạn / bờ / đầu" phía, câu không có "nhất"...). Không lấy gì từ test.
    python scratch/p2_n9_mapref2.py"""
import sys
sys.path.insert(0, "src")
from common import *
import mapref
mapref.VOCAB = mapref.build_vocab([s["mission"]["text"] for sp in ("train", "validation") for s in load_split(sp)[2]])

D = {"north_most": ("bắc", "trên"), "south_most": ("nam", "dưới"), "west_most": ("tây", "trái"), "east_most": ("đông", "phải")}
MOST_T = ["địa điểm nằm về phía {c} xa nhất", "nơi ở phía {c} xa nhất", "địa điểm ở mạn {c} nhất", "chỗ ở rìa phía {c} nhất",
          "địa điểm sát mép phía {c} nhất", "nơi gần mép {s} của bản đồ nhất", "địa điểm gần rìa {s} bản đồ nhất",
          "địa điểm nằm sát biên {s} nhất", "chỗ ở cực {c} của bản đồ", "điểm cực {c} trên bản đồ", "địa điểm nằm {c} nhất",
          "tòa nhà ở phía {c} ngoài cùng", "nơi xa nhất theo hướng {c}", "địa điểm xa nhất theo phía {c}",
          "địa điểm đi về phía {c} xa nhất", "chỗ nằm ở tận phía {c}", "nơi ở tận mép {s}", "địa điểm sát mép {s} nhất bản đồ",
          "phòng nằm ở phía {c} nhất", "dãy nhà ở phía {c} nhất", "nơi nằm ở hàng/cột ngoài cùng phía {c}",
          "địa điểm có vị trí {c} nhất", "chỗ nào ở xa nhất về phía {c}", "địa điểm xa xôi nhất về phía {c}",
          "nơi ở phía {s} cùng của bản đồ", "chỗ ở phần {s} cùng", "địa điểm nằm ở phía {c} hơn cả",
          "địa điểm nằm lệch về phía {c} nhiều nhất", "nơi nằm sâu nhất về phía {c}", "địa điểm ở phía {c} cùng"]
ANCH_T = ["địa điểm gần nhất với {a}", "nơi gần nhất so với {a}", "chỗ gần {a} nhất ngoài chính nó", "địa điểm ở gần {a} nhất",
          "nơi nằm kề sát {a} nhất", "địa điểm có khoảng cách tới {a} ngắn nhất", "chỗ cách {a} ít bước nhất",
          "nơi cách {a} gần nhất", "địa điểm sát cạnh {a} nhất", "địa điểm gần kề {a} nhất", "chỗ nằm sát sườn {a} nhất",
          "nơi láng giềng gần nhất của {a}", "địa điểm gần {a} hơn mọi nơi khác", "nơi gần nhất tính từ {a}",
          "địa điểm nằm gần {a} hơn tất cả", "tòa nhà gần nhất cạnh {a}", "điểm nằm sát {a} nhất", "chỗ ở kế {a} nhất",
          "địa điểm sát nách {a} nhất", "nơi gần {a} nhất trên bản đồ", "địa điểm láng giềng gần {a} nhất",
          "nơi có khoảng cách gần {a} nhất", "địa điểm cách {a} một quãng ngắn nhất", "vị trí gần nhất với {a}",
          "chỗ gần nhất bên cạnh {a}", "nơi ít xa {a} nhất", "khu vực nằm kế bên {a} nhất"]
ANCHORS = {"library": "thư viện", "canteen": "căn tin", "gate": "cổng trường", "clinic": "phòng y tế", "dorm": "ký túc xá"}
FRAMES = ["Giao gói hàng đến {} giúp mình.", "Có đơn giao hộp giấy ở {}.", "Người nhận ở {} đang chờ bưu kiện.",
          "Trước khi mang thùng hàng tới {}, nhớ ghé căn tin lấy khay cơm.", "Đích đến là {}, hàng cần giao là túi đồ.",
          "Hàng cho {}: bưu kiện."]

tot = ok = 0; miss = {}
def chk(text, kind, anc=None, tag=""):
    global tot, ok
    for t in (text, mapref.unaccent(text)):
        d = mapref.detect(t); tot += 1
        good = d is not None and d[0] == kind and (anc is None or (d[1] and mapref.unaccent(anc) in d[1]))
        ok += good
        if not good:
            miss.setdefault(tag, []).append((t, d))

for kind, (c, s) in D.items():
    for tp in MOST_T:
        for fr in FRAMES:
            chk(fr.format(tp.format(c=c, s=s)), kind, tag=tp)
for at, an in ANCHORS.items():
    for tp in ANCH_T:
        for fr in FRAMES:
            chk(fr.format(tp.format(a=an)), "anchor_near", an, tag=tp)
print(f"nhận đúng {ok}/{tot} = {ok / tot:.3f}")
for tag, L in sorted(miss.items(), key=lambda kv: -len(kv[1])):
    print(f"  {len(L):3d}  {tag:45s} | {L[0][0][:70]} -> {L[0][1]}")
