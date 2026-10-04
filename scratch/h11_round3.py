"""Vòng 3: bộ câu thử MỚI (viết từ kiến thức tiếng Việt chung, chưa dùng để viết luật) gồm cả câu giao hàng lồng ý gấp /
dễ vỡ. Chấm bằng parser.parse() trên CẢ yêu cầu (đúng như khi dự đoán). In các câu đọc sai."""
import sys
sys.path.insert(0, "src")
from common import *
import nlp2
from nlp2 import MissionParser2, _unaccent

# (văn bản, gấp?, dễ vỡ?)
T = [
    # --- vòng 3: văn nói, viết tắt, trang trọng, ý gấp đứng đầu ---
    ("Gấp! Mang hộp bút tới thư viện.", 1, 0), ("Khẩn cấp: giao bưu kiện tới ký túc xá.", 1, 0),
    ("Việc gấp đây, đưa hồ sơ tới phòng hành chính.", 1, 0), ("Đơn hỏa tốc: thùng hàng tới bãi xe.", 1, 0),
    ("Yêu cầu mới: Giao sách tới thư viện. Vui lòng giao trong thời gian sớm nhất.", 1, 0),
    ("Giao thuốc tới trạm y tế. Đề nghị thực hiện ngay.", 1, 0), ("Mang tài liệu tới giảng đường. Kính đề nghị ưu tiên xử lý.", 1, 0),
    ("Giao bóng tới nhà thể thao. Lẹ nha robot.", 1, 0), ("Mang cơm tới căn tin. Nhanh nha, đói lắm rồi.", 1, 0),
    ("Giao bưu kiện tới cổng trường. Gấp gấp gấp.", 1, 0), ("Mang hóa chất tới phòng thí nghiệm. Cần gấp trong 3 phút.", 1, 0),
    ("Giao túi đồ tới ký túc xá. Chậm trễ là không được.", 1, 0), ("Mang hộp giấy tới căn tin. Phải tới trước khi hết giờ.", 1, 0),
    ("Giao hộp bút tới thư viện. Người nhận sắp rời đi.", 1, 0), ("Mang tập đề tới giảng đường. Thầy cần liền.", 1, 0),
    ("Giao hồ sơ tới phòng hành chính. Hết hạn nộp trong ít phút.", 1, 0), ("Mang chìa khóa tới bãi xe. Xe sắp chạy rồi.", 1, 0),
    ("Giao thuốc tới trạm y tế. Bệnh nhân không chờ được.", 1, 0), ("Mang nước tới nhà thể thao. Giải đấu đang diễn ra, nhanh.", 1, 0),
    ("Giao bưu kiện tới cổng trường. Shipper bên ngoài đang chờ, nhanh lên.", 1, 0),
    ("Mang hộp bút tới thư viện. Cần có mặt tức thì.", 1, 0), ("Giao sách tới thư viện. Đi liền giùm.", 1, 0),
    ("Mang tài liệu tới căn tin. Không được phép trễ.", 1, 0), ("Giao hàng tới ký túc xá. Thời hạn rất sát.", 1, 0),
    ("Robot ơi, mang gấp giúp mình hộp thuốc tới trạm y tế nhé.", 1, 0), ("Nhờ robot chuyển gấp tập hồ sơ qua phòng hành chính.", 1, 0),
    ("Cần gấp: hộp giấy tới giảng đường.", 1, 0), ("Tới căn tin ngay, giao khay cơm.", 1, 0),
    ("Từ từ thôi! Mang hộp bút tới thư viện.", 0, 0), ("Không gấp: giao bưu kiện tới ký túc xá.", 0, 0),
    ("Việc không gấp, đưa hồ sơ tới phòng hành chính.", 0, 0), ("Đơn thường: thùng hàng tới bãi xe.", 0, 0),
    ("Yêu cầu mới: Giao sách tới thư viện. Thời gian giao linh hoạt.", 0, 0),
    ("Giao thuốc tới trạm y tế. Không yêu cầu giao ngay.", 0, 0), ("Mang tài liệu tới giảng đường. Không cần ưu tiên xử lý.", 0, 0),
    ("Giao bóng tới nhà thể thao. Thong thả nha robot.", 0, 0), ("Mang cơm tới căn tin. Chưa đói, cứ từ từ.", 0, 0),
    ("Giao bưu kiện tới cổng trường. Khỏi gấp.", 0, 0), ("Mang hóa chất tới phòng thí nghiệm. Sáng mai mới làm thí nghiệm.", 0, 0),
    ("Giao túi đồ tới ký túc xá. Chậm trễ chút cũng không sao.", 0, 0), ("Mang hộp giấy tới căn tin. Tới lúc nào cũng được.", 0, 0),
    ("Giao hộp bút tới thư viện. Người nhận ở đó cả ngày.", 0, 0), ("Mang tập đề tới giảng đường. Thầy chưa cần.", 0, 0),
    ("Giao hồ sơ tới phòng hành chính. Hạn nộp còn xa.", 0, 0), ("Mang chìa khóa tới bãi xe. Xe chưa đi đâu.", 0, 0),
    ("Giao thuốc tới trạm y tế. Không phải ca khẩn.", 0, 0), ("Mang nước tới nhà thể thao. Giải đấu chiều mới bắt đầu.", 0, 0),
    ("Giao bưu kiện tới cổng trường. Không ai chờ, đi thoải mái.", 0, 0),
    ("Mang hộp bút tới thư viện. Không cần có mặt ngay.", 0, 0), ("Giao sách tới thư viện. Không phải đi liền.", 0, 0),
    ("Mang tài liệu tới căn tin. Trễ cũng được.", 0, 0), ("Giao hàng tới ký túc xá. Thời hạn còn rộng.", 0, 0),
    ("Robot ơi, mang giúp mình hộp thuốc tới trạm y tế nhé, không gấp đâu.", 0, 0), ("Nhờ robot chuyển tập hồ sơ qua phòng hành chính, khi nào tiện.", 0, 0),
    ("Giao hộp giấy tới giảng đường. Không cần nhanh.", 0, 0), ("Tới căn tin giao khay cơm, không cần vội.", 0, 0),
    ("Mang quả bóng tới nhà thể thao.", 0, 0), ("Giao tập tài liệu tới thư viện. Cảm ơn robot.", 0, 0),
    ("Điểm giao: căn tin. Hàng: thùng nước uống.", 0, 0), ("Nhờ robot mang hộp bút tới giảng đường giúp mình. Cảm ơn nhiều.", 0, 0),
]
missions = [s["mission"] for sp in ("train", "validation") for s in load_split(sp)[2]]
p = MissionParser2().fit(missions)
tot = [0, 0]; bad = []
for form in ("có dấu", "không dấu"):
    for text, u, f in T:
        t = text if form == "có dấu" else _unaccent(text)
        r = p.parse(t)
        ok = (bool(r["urgent"]) == bool(u), bool(r["fragile"]) == bool(f))
        tot[0] += ok[0]; tot[1] += ok[1]
        if not all(ok):
            bad.append(f"[{form}] {t}  -> đoán gấp={int(r['urgent'])} dễ vỡ={int(r['fragile'])}, đúng là gấp={u} dễ vỡ={f}")
n = 2 * len(T)
print(f"VÒNG 3: gấp đúng {tot[0]}/{n} = {tot[0] / n:.3f} | dễ vỡ đúng {tot[1]}/{n} = {tot[1] / n:.3f}")
for b in bad: print("  ", b)
nlp2.save_embed_cache()
