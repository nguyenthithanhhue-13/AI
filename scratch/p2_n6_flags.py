"""Phép thử TỰ SOẠN cho cờ gấp / dễ vỡ (cách nói tiếng Việt thông dụng, không lấy từ test)."""
import sys
sys.path.insert(0, "src")
from common import *
import nlp3
p = nlp3.MissionParser3.load(OUT / (sys.argv[1] if len(sys.argv) > 1 else "nlp3_trainonly_kb.pkl"))
URG = ["Cần gấp lắm.", "Không được chậm trễ.", "Ưu tiên số một, đi ngay.", "Cần ngay trong 10 phút.", "Khẩn trương nhé!", "Giao liền giúp mình.",
       "Đang rất cần, nhanh lên.", "Hàng hỏa tốc.", "Phải tới trước 5 phút nữa.", "Càng sớm càng tốt.", "Đi nhanh giúp mình nhé.",
       "Việc này rất khẩn.", "Cần ngay lập tức."]
NOURG = ["Không gấp đâu.", "Mai giao cũng kịp.", "Cứ thong thả.", "Từ từ cũng được.", "Chiều nay giao cũng được.", "Không cần vội.",
         "Lúc nào tiện thì giao.", "Không có gì gấp cả.", "Chậm một chút cũng không sao."]
FRAG = ["Hàng dễ vỡ, đi cẩn thận.", "Kiện hàng này rất mỏng manh.", "Bên trong là đồ thủy tinh.", "Nhẹ tay nhé, đồ sứ đấy.",
        "Hàng kỵ va đập.", "Cẩn thận kẻo vỡ.", "Đồ gốm, tránh rung lắc.", "Chú ý: hàng dễ hỏng khi va chạm.", "Đồ điện tử nhạy cảm, đi êm nhé."]
NOFRAG = ["Hàng chắc chắn, không dễ vỡ.", "Rơi cũng chẳng sao.", "Hàng bền, không lo vỡ.", "Đồ không vỡ được đâu.", "Hàng cứng cáp lắm.",
          "Va chạm chút cũng không hề gì.", "Đồ nhựa, không sợ vỡ."]
types = ["library", "dorm", "sports", "clinic", "canteen", "parking", "lecture", "lab", "office", "gate"]
L = {t: [(i, i)] for i, t in enumerate(types)}
base = "Giao gói hàng tới thư viện giúp mình. "
for name, lst, key, val in (("gấp", URG, "urgent", True), ("không gấp", NOURG, "urgent", False),
                            ("dễ vỡ", FRAG, "fragile", True), ("không dễ vỡ", NOFRAG, "fragile", False)):
    ok = 0; bad = []
    for s in lst:
        for t in (base + s, nlp3._unaccent(base + s)):
            r = p.parse(t, set(types), L)
            ok += bool(r[key]) == val
            if bool(r[key]) != val and not t.isascii(): bad.append(s)
    print(f"{name}: {ok}/{2 * len(lst)}", "sai:", bad)
