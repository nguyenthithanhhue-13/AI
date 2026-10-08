"""Phép thử TÊN GỌI MỚI (tự soạn, không có trong train / validation / kho vòng 1): đo độ đúng loại đích khi đích là tên lạ,
trong khung câu tiếng Việt thường gặp, có địa điểm gây nhiễu và điểm ghé, có dấu / không dấu.
    python scratch/p2_n7_novel.py <file bộ đọc>"""
import sys, random, collections
sys.path.insert(0, "src")
from common import *
import nlp3
p = nlp3.MissionParser3.load(OUT / (sys.argv[1] if len(sys.argv) > 1 else "nlp3_trainonly_canon.pkl"))
NOVEL = {
    "library": ["trung tâm học liệu", "phòng tư liệu", "nhà sách của trường", "khu lưu trữ sách", "phòng mượn trả tài liệu"],
    "dorm": ["khu lưu trú sinh viên", "nhà nghỉ sinh viên", "tòa nội trú B", "khu ký túc", "dãy nhà ở tập thể"],
    "sports": ["sân vận động mini", "nhà tập thể dục", "khu thể dục thể thao", "sân chơi thể thao", "phòng thể hình"],
    "clinic": ["phòng y tế học đường", "điểm sơ cứu", "phòng chăm sóc sức khỏe", "khu khám bệnh", "trạm cứu thương"],
    "canteen": ["khu ẩm thực", "nhà ăn sinh viên", "quán cơm trường", "khu phục vụ ăn uống", "phòng ăn chung"],
    "parking": ["điểm đỗ phương tiện", "khu để xe máy", "bãi giữ xe đạp", "nhà để xe sinh viên", "khu đậu ô tô"],
    "lecture": ["tòa nhà học", "phòng học chung", "khu lên lớp", "giảng đường lớn A", "phòng hội thảo học"],
    "lab": ["phòng nghiên cứu", "phòng thực hành hóa", "xưởng thí nghiệm", "khu thực nghiệm", "phòng làm thí nghiệm vật lý"],
    "office": ["phòng công tác sinh viên", "khu hành chính tổng hợp", "phòng tiếp dân", "văn phòng nhà trường", "phòng quản lý đào tạo"],
    "gate": ["cổng chính trường", "lối ra vào", "cổng bảo vệ", "cổng số hai", "trạm kiểm soát ra vào"],
}
CAN = {"library": "thư viện", "dorm": "ký túc xá", "canteen": "căn tin", "clinic": "trạm y tế", "parking": "bãi xe",
       "lecture": "giảng đường", "lab": "phòng thí nghiệm", "office": "phòng hành chính", "gate": "cổng trường", "sports": "nhà thể thao"}
FR = ["Có đơn giao hộp giấy ở {G}. Đừng nhầm với {D} nhé.", "Nhờ robot chuyển gói hàng qua {G}. Không cần ghé {D}.",
      "Trước khi mang tập tài liệu tới {G}, nhớ ghé {V} lấy con dấu. Hôm qua đã giao ở {D} rồi.",
      "Người nhận ở {G} đang chờ bưu kiện. Bỏ qua {D}, không phải ở đó.", "Điểm giao: {G}. Hàng: túi đồ. {D} không phải điểm nhận.",
      "Hủy đơn giao {D}. Thay vào đó: Giao thùng hàng tới {G} giúp mình."]
random.seed(1)
types = list(NOVEL)
c = collections.Counter(); bad = collections.Counter()
for g in types:
    for alias in NOVEL[g]:
        for fr in FR:
            d, v = random.sample([t for t in types if t != g], 2)
            L = {t: [(i, i)] for i, t in enumerate(types)}
            for txt in (fr.format(G=alias, D=CAN[d], V=CAN[v]), nlp3._unaccent(fr.format(G=alias, D=CAN[d], V=CAN[v]))):
                r = nlp3.resolve3(p.parse(txt, set(types), L), L)
                ok = r["goal"] == g
                c[g, "n"] += 1; c[g, "ok"] += ok
                if not ok: bad[(alias, r["goal"])] += 1
tot = sum(c[g, "ok"] for g in types) / sum(c[g, "n"] for g in types)
print(f"đúng loại đích với tên mới: {tot:.3f}  " + " ".join(f"{g}={c[g,'ok']}/{c[g,'n']}" for g in types))
print("sai nhiều nhất:", bad.most_common(12))
