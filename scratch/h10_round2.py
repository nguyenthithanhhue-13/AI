"""Vòng 2: bộ câu thử MỚI (viết từ kiến thức tiếng Việt chung, chưa dùng để viết luật) gồm cả câu giao hàng lồng ý gấp /
dễ vỡ. Chấm bằng parser.parse() trên CẢ yêu cầu (đúng như khi dự đoán). In các câu đọc sai."""
import sys
sys.path.insert(0, "src")
from common import *
import nlp2
from nlp2 import MissionParser2, _unaccent

# (văn bản, gấp?, dễ vỡ?)
T = [
    # --- gấp lồng trong câu giao hàng ---
    ("Giao gấp hộp bút tới thư viện.", 1, 0), ("Mang ngay tập tài liệu đến căn tin giúp mình.", 1, 0),
    ("Chuyển khẩn bưu kiện tới ký túc xá.", 1, 0), ("Đưa hộp thuốc tới trạm y tế, càng nhanh càng tốt.", 1, 0),
    ("Mang tập hồ sơ đến phòng hành chính, gấp lắm.", 1, 0), ("Gửi thùng hàng tới bãi xe trong vòng 5 phút.", 1, 0),
    ("Cần giao hỏa tốc hộp giấy tới giảng đường.", 1, 0), ("Nhanh chóng mang quả bóng tới nhà thể thao.", 1, 0),
    ("Mang hóa chất tới phòng thí nghiệm ngay lập tức.", 1, 0), ("Tới cổng trường thật nhanh để giao bưu kiện.", 1, 0),
    ("Giao túi đồ tới ký túc xá trước 9 giờ, không được trễ.", 1, 0), ("Đưa sách tới thư viện liền nhé, đang cần lắm.", 1, 0),
    ("Khẩn: mang hộp sơ cứu tới trạm y tế.", 1, 0), ("Ưu tiên giao tập đề thi tới giảng đường trước.", 1, 0),
    ("Mang khay cơm tới căn tin, người ta đang chờ gấp.", 1, 0), ("Chạy nhanh tới bãi xe giao chìa khóa giúp mình.", 1, 0),
    # --- không gấp lồng trong câu giao hàng ---
    ("Mang tập tài liệu tới thư viện, không cần gấp.", 0, 0), ("Giao hộp bút tới căn tin, lúc nào cũng được.", 0, 0),
    ("Đưa bưu kiện tới ký túc xá, cứ từ từ.", 0, 0), ("Gửi thùng hàng tới bãi xe, không vội đâu.", 0, 0),
    ("Mang hộp giấy tới giảng đường khi nào rảnh.", 0, 0), ("Chuyển tập hồ sơ đến phòng hành chính, mai cũng được.", 0, 0),
    ("Không cần giao gấp, mang túi đồ tới ký túc xá là được.", 0, 0), ("Giao quả bóng tới nhà thể thao, thong thả thôi.", 0, 0),
    ("Mang sách tới thư viện, chưa cần ngay.", 0, 0), ("Đưa hộp thuốc tới trạm y tế, không cần đi nhanh.", 0, 0),
    ("Tiện đường thì mang bưu kiện tới cổng trường.", 0, 0), ("Giao hóa chất tới phòng thí nghiệm trong hôm nay là được.", 0, 0),
    ("Đừng vội, cứ mang tập tài liệu tới căn tin.", 0, 0), ("Mang thùng nước tới nhà thể thao, trễ chút không sao.", 0, 0),
    # --- câu riêng: gấp ---
    ("Giao hộp bút tới thư viện. Bên nhận hối lắm rồi.", 1, 0), ("Mang tài liệu đến căn tin. Phải có trong 10 phút nữa.", 1, 0),
    ("Giao bưu kiện tới ký túc xá. Đừng để mình đợi.", 1, 0), ("Mang hồ sơ tới phòng hành chính. Việc này không trì hoãn được.", 1, 0),
    ("Giao thùng hàng tới bãi xe. Tốc độ là quan trọng nhất.", 1, 0), ("Mang hộp giấy tới giảng đường. Giao trễ là bị phạt.", 1, 0),
    ("Giao quả bóng tới nhà thể thao. Trận đấu sắp bắt đầu.", 1, 0), ("Mang sách tới thư viện. Đang rất cần, nhanh giúp.", 1, 0),
    ("Giao thuốc tới trạm y tế. Tình huống cấp cứu.", 1, 0), ("Mang hóa chất tới phòng thí nghiệm. Thí nghiệm đang chờ, khẩn.", 1, 0),
    ("Giao bưu kiện tới cổng trường. Khách sắp đi rồi.", 1, 0), ("Mang túi đồ tới ký túc xá. Phải nhanh mới kịp.", 1, 0),
    ("Giao khay cơm tới căn tin. Không được chần chừ.", 1, 0), ("Mang tập đề tới giảng đường. Còn 5 phút nữa thi.", 1, 0),
    ("Giao hộp bút tới thư viện. Đơn ưu tiên cao nhất.", 1, 0), ("Mang tài liệu tới phòng hành chính. Hạn nộp là ngay bây giờ.", 1, 0),
    ("Giao hàng tới bãi xe. Chủ xe đang đứng đợi.", 1, 0), ("Mang hộp giấy tới căn tin. Nhanh nhanh lên nhé.", 1, 0),
    # --- câu riêng: không gấp ---
    ("Giao hộp bút tới thư viện. Không có gì phải vội.", 0, 0), ("Mang tài liệu đến căn tin. Thời gian thoải mái.", 0, 0),
    ("Giao bưu kiện tới ký túc xá. Giao lúc nào tiện cũng được.", 0, 0), ("Mang hồ sơ tới phòng hành chính. Không cần ưu tiên.", 0, 0),
    ("Giao thùng hàng tới bãi xe. Đi chậm cũng được.", 0, 0), ("Mang hộp giấy tới giảng đường. Không ai hối cả.", 0, 0),
    ("Giao quả bóng tới nhà thể thao. Chiều mới cần.", 0, 0), ("Mang sách tới thư viện. Tuần sau mới dùng.", 0, 0),
    ("Giao thuốc tới trạm y tế. Không khẩn cấp.", 0, 0), ("Mang hóa chất tới phòng thí nghiệm. Chưa cần gấp.", 0, 0),
    ("Giao bưu kiện tới cổng trường. Cứ đi bình thường.", 0, 0), ("Mang túi đồ tới ký túc xá. Không gấp gáp gì.", 0, 0),
    ("Giao khay cơm tới căn tin. Việc này không gấp.", 0, 0), ("Mang tập đề tới giảng đường. Còn sớm chán.", 0, 0),
    ("Giao hộp bút tới thư viện. Mức ưu tiên bình thường.", 0, 0), ("Mang tài liệu tới phòng hành chính. Cuối ngày mới cần.", 0, 0),
    ("Giao hàng tới bãi xe. Lúc nào xong thì xong.", 0, 0), ("Mang hộp giấy tới căn tin. Đừng hấp tấp.", 0, 0),
    # --- dễ vỡ / không dễ vỡ, lồng và riêng ---
    ("Mang bình thủy tinh tới phòng thí nghiệm, nhẹ tay nhé.", 0, 1), ("Giao hộp ly tách tới căn tin cẩn thận kẻo vỡ.", 0, 1),
    ("Mang kính hiển vi tới phòng thí nghiệm. Thiết bị rất dễ hỏng.", 0, 1), ("Giao màn hình tới phòng hành chính. Tránh va đập.", 0, 1),
    ("Mang lọ hoa tới thư viện. Đồ dễ bể.", 0, 1), ("Giao thùng trứng tới căn tin. Đi êm thôi.", 0, 1),
    ("Mang mô hình tới giảng đường. Hàng mỏng manh, đừng làm rơi.", 0, 1), ("Giao ống nghiệm tới phòng thí nghiệm. Không được rung lắc.", 0, 1),
    ("Mang bóng đèn tới ký túc xá. Vỡ là bỏ.", 0, 1), ("Giao tranh kính tới phòng hành chính. Phải thật nhẹ nhàng.", 0, 1),
    ("Mang sách tới thư viện. Hàng bền, không sợ rơi.", 0, 0), ("Giao tập tài liệu tới căn tin. Toàn giấy, không vỡ được.", 0, 0),
    ("Mang quả bóng tới nhà thể thao. Rơi cũng không sao.", 0, 0), ("Giao túi quần áo tới ký túc xá. Không cần nhẹ tay.", 0, 0),
    ("Mang thùng nước tới bãi xe. Hàng chắc, không dễ vỡ.", 0, 0), ("Giao hộp bút tới giảng đường. Không có gì dễ vỡ.", 0, 0),
    ("Mang hộp nhựa tới căn tin. Đồ này không vỡ đâu.", 0, 0), ("Giao bưu kiện tới cổng trường. Đã đóng gói chắc chắn.", 0, 0),
    # --- kết hợp ---
    ("Giao gấp bình thủy tinh tới phòng thí nghiệm, nhẹ tay.", 1, 1), ("Mang lọ hoa tới thư viện, không vội nhưng dễ vỡ.", 0, 1),
    ("Giao sách tới thư viện ngay. Hàng bền.", 1, 0), ("Mang ly tách tới căn tin, từ từ thôi kẻo vỡ.", 0, 1),
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
print(f"VÒNG 2: gấp đúng {tot[0]}/{n} = {tot[0] / n:.3f} | dễ vỡ đúng {tot[1]}/{n} = {tot[1] / n:.3f}")
for b in bad: print("  ", b)
nlp2.save_embed_cache()
