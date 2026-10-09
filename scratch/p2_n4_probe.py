"""Phép thử TỰ SOẠN (không lấy từ test) cho bộ nhận diện mô tả qua bản đồ: các cách nói tiếng Việt khác cho "xa nhất về một
phía" và "gần X nhất", có dấu / không dấu, chèn vào khung câu thật. Đo tỉ lệ nhận đúng và tỉ lệ bắt nhầm câu thường."""
import sys
sys.path.insert(0, "src")
from common import *
import mapref
mapref.VOCAB = mapref.build_vocab([s["mission"]["text"] for sp in ("train", "validation") for s in load_split(sp)[2]])

MOST = {
    "north_most": ["địa điểm ở phía bắc nhất", "nơi nằm ở cực bắc bản đồ", "điểm cao nhất bản đồ", "tòa nhà nằm trên cùng",
                   "khu ở góc trên cùng", "nơi nằm ở phía trên cùng của bản đồ", "địa điểm nằm xa nhất về hướng bắc",
                   "chỗ nằm trên cùng", "địa điểm nằm cao nhất", "vị trí trên cùng của bản đồ", "nơi ở mép trên cùng",
                   "địa điểm nằm ở rìa trên nhất", "chỗ xa nhất phía bắc", "địa điểm ở tận cùng phía bắc", "nơi xa về phía bắc nhất",
                   "địa điểm nằm phía trên cùng bản đồ", "công trình nằm cao nhất trên bản đồ"],
    "south_most": ["địa điểm ở phía nam nhất", "nơi nằm ở cực nam bản đồ", "điểm thấp nhất bản đồ", "tòa nhà nằm dưới cùng",
                   "chỗ nằm dưới cùng", "nơi ở mép dưới cùng", "địa điểm nằm ở rìa dưới nhất", "vị trí dưới cùng của bản đồ",
                   "địa điểm ở tận cùng phía nam", "nơi nằm thấp nhất", "địa điểm nằm phía dưới cùng bản đồ"],
    "west_most": ["địa điểm ở phía tây nhất", "nơi nằm ở cực tây bản đồ", "chỗ nằm bên trái nhất", "tòa nhà nằm ngoài cùng bên trái",
                  "địa điểm sát mép trái bản đồ nhất", "nơi ở rìa trái cùng", "vị trí bên trái cùng của bản đồ", "địa điểm tận cùng phía tây",
                  "nơi xa nhất về bên trái", "địa điểm nằm phía trái nhất"],
    "east_most": ["địa điểm ở phía đông nhất", "nơi nằm ở cực đông bản đồ", "chỗ nằm bên phải nhất", "tòa nhà nằm ngoài cùng bên phải",
                  "địa điểm sát mép phải bản đồ nhất", "nơi ở rìa phải cùng", "vị trí bên phải cùng của bản đồ", "địa điểm tận cùng phía đông",
                  "nơi xa nhất về bên phải", "địa điểm nằm phía phải nhất"],
    "anchor_near": ["nơi gần thư viện nhất", "chỗ sát bên thư viện nhất", "điểm ở ngay cạnh thư viện nhất", "địa điểm cách thư viện gần nhất",
                    "nơi nằm gần thư viện hơn cả", "vị trí kế bên thư viện nhất", "tòa nhà gần thư viện nhất", "nơi ở sát thư viện nhất",
                    "địa điểm lân cận thư viện nhất", "địa điểm gần với thư viện nhất", "chỗ liền kề thư viện nhất", "nơi nằm sát vách thư viện nhất",
                    "địa điểm gần thư viện nhất trên bản đồ", "khu vực sát thư viện nhất", "địa điểm ở gần thư viện nhất"],
}
FRAMES = ["Giao gói hàng đến {} giúp mình.", "Có đơn giao hộp giấy ở {}.", "Người nhận ở {} đang chờ bưu kiện.",
          "Trước khi mang thùng hàng tới {}, nhớ ghé căn tin lấy khay cơm.", "Điểm giao: {}. Hàng: túi đồ."]
NEG = ["Giao gói hàng đến thư viện phía bắc giúp mình.", "Mang hộp giấy tới ký túc xá gần cổng trường hơn.",
       "Ghé bãi xe ở bên trái bản đồ trước rồi giao tới căn tin.", "Hàng dễ vỡ nhất định phải đi cẩn thận.",
       "Ưu tiên cao nhất, giao ngay tới phòng y tế.", "Giao tới giảng đường nằm phía nam.", "Cần gấp nhất có thể, mang tới thư viện.",
       "Đi đường gần nhất tới căn tin giúp mình.", "Chọn đường ngắn nhất tới cổng trường nhé."]
tot = ok = 0; miss = []
for kind, phr in MOST.items():
    for p in phr:
        for fr in FRAMES:
            for t in (fr.format(p), mapref.unaccent(fr.format(p))):
                d = mapref.detect(t); tot += 1
                good = d is not None and d[0] == kind
                ok += good
                if not good and fr == FRAMES[0] and not t.isascii():
                    miss.append((kind, p, d[0] if d else None))
print(f"nhận đúng {ok}/{tot}")
for m in miss: print("   SÓT", m)
fp = [(t, mapref.detect(t)) for t in NEG + [mapref.unaccent(t) for t in NEG] if mapref.detect(t)]
print("bắt nhầm câu thường:", fp)
