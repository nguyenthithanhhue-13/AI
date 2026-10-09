"""Phép thử TỰ SOẠN cho khung ĐIỂM GHÉ và câu ĐỐI CHIẾU (không lấy từ test). Bộ đọc chỉ học train (outputs/nlp3_trainonly.pkl).
Bản đồ giả: mỗi loại một bản. Đo đích / điểm ghé đọc đúng."""
import sys, itertools, random
sys.path.insert(0, "src")
from common import *
import nlp3
p = nlp3.MissionParser3.load(OUT / sys.argv[1] if len(sys.argv) > 1 else OUT / "nlp3_trainonly.pkl")
NAMES = {"library": "thư viện", "dorm": "ký túc xá", "canteen": "căn tin", "clinic": "trạm y tế", "parking": "bãi xe",
         "lecture": "giảng đường", "lab": "phòng thí nghiệm", "office": "phòng hành chính", "gate": "cổng trường", "sports": "nhà thể thao"}
VIA = ["Lấy hồ sơ ở {V} xong thì mang túi đồ đến {G}.", "Ghé {V} lấy đồ trước, rồi giao tới {G}.",
       "Trước tiên qua {V} nhận hàng, sau đó chuyển tới {G}.", "Giao gói hàng tới {G}. Nhưng phải tạt qua {V} trước đã.",
       "Chưa đi thẳng được: ghé {V} trước, rồi mang tới {G}.", "Đi qua {V} rồi mới tới {G} nhé.",
       "Trên đường tới {G}, nhớ ghé {V}.", "Mang hộp giấy tới {G}, nhưng trước đó ghé {V} một chút.",
       "Từ {V} lấy hàng rồi chuyển đến {G}.", "Dừng ở {V} trước, sau đó đến {G}.", "Hãy ghé {V} rồi hẵng tới {G}.",
       "Nhận bưu kiện tại {V}, sau đó giao cho người ở {G}."]
CON = ["Giao gói hàng tới {G}, không phải {D}.", "Giao tới {G} chứ không phải {D}.", "Không phải {D} mà là {G} nhé, giao gói hàng tới đó.",
       "Điểm giao là {G}, đừng giao {D}.", "Hàng cho {G} chứ không phải {D}: hộp giấy.", "Đích đến là {G}, không phải {D} như tin trước.",
       "{D} không phải điểm nhận. Giao túi đồ tới {G}.", "Bỏ qua {D}, không phải ở đó. Mang hàng tới {G}.",
       "Lúc nãy ghi nhầm {D}, thật ra là {G}. Giao hộp giấy tới đó.", "Giao hộp giấy tới {G} (không phải {D})."]
random.seed(0)
types = list(NAMES)
L = {t: [(i, i)] for i, t in enumerate(types)}
present = set(types)
res = {"điểm ghé": [0, 0], "đối chiếu": [0, 0]}; bad = []
for fr in VIA:
    for _ in range(6):
        g, v = random.sample(types, 2)
        for txt in (fr.format(G=NAMES[g], V=NAMES[v]), nlp3._unaccent(fr.format(G=NAMES[g], V=NAMES[v]))):
            r = nlp3.resolve3(p.parse(txt, present, L), L)
            ok = r["goal"] == g and r["via"] == v
            res["điểm ghé"][0] += ok; res["điểm ghé"][1] += 1
            if not ok and not txt.isascii(): bad.append((txt, r["goal"], r["via"]))
for fr in CON:
    for _ in range(6):
        g, d = random.sample(types, 2)
        for txt in (fr.format(G=NAMES[g], D=NAMES[d]), nlp3._unaccent(fr.format(G=NAMES[g], D=NAMES[d]))):
            r = nlp3.resolve3(p.parse(txt, present, L), L)
            ok = r["goal"] == g and not r["via"]
            res["đối chiếu"][0] += ok; res["đối chiếu"][1] += 1
            if not ok and not txt.isascii(): bad.append((txt, r["goal"], r["via"]))
print(res)
for b in bad[:30]: print("  SAI", b)
