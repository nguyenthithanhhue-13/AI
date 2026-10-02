# Giải thích lời giải Phenikaa Campus Courier v2

Tài liệu này viết cho người mới. Mỗi phần có 4 mục: **(a)** phần này làm gì và vì sao, **(b)** kết quả đo thật,
**(c)** vài câu để giảng lại cho đội, **(d)** những giả thuyết đã thử nhưng sai.

Sơ đồ tổng thể:

```
ảnh ──CV──► world (đồ thị, địa điểm, robot, thời tiết) ─┐
                                                        ├─► chiến thuật(robot_id) ─► 0/1/2/3
mission ──NLP──► (goal, via, tham chiếu, gấp, dễ vỡ) ───┘
```

Ý tưởng quan trọng nhất: **tách bài thành 3 khâu và đo từng khâu riêng bằng `scenes.json`**. Nhờ vậy khi điểm thấp
ta biết ngay khâu nào có lỗi, không phải đoán mò.

---

## 1. Chiến thuật của 10 robot (`src/strategy.py`)

### (a) Làm gì, vì sao

Trước khi đụng tới ảnh hay câu chữ, tôi lấy thông tin **đúng** trong `scenes.json` (đồ thị, đích, thời tiết…) và hỏi:
"nếu biết hết mọi thứ thì có đoán được bước đi của từng robot không?". Nếu khâu này không đạt gần 100% thì CV và NLP
tốt mấy cũng vô ích, nên phải làm trước.

Cả 9 robot đầu hóa ra dùng **chung một bộ máy**: tìm đường rẻ nhất (Dijkstra) trên không gian trạng thái
*(giao lộ, hướng đang quay)*. Chi phí một đường đi = tổng chi phí các đoạn + tiền phạt mỗi lần đổi hướng.
Mỗi robot chỉ khác nhau ở **bảng giá**:

| robot | bảng giá (mỗi đoạn mặc định = 1) |
|---|---|
| 0 Tia Chớp | mọi đoạn = 1 (ít đoạn nhất) |
| 1 Yên Tĩnh | đoạn đông người = 6 |
| 2 Mái Hiên | đoạn có mái che = 0,5 |
| 3 Mây Mưa | trời mưa: đoạn có mái che = 0,2; trời khô: đoạn đông = 3 |
| 4 Sơn Dương | đi được bậc thang; đoạn đông = 2; **trời mưa thì bậc thang cộng thêm 4** |
| 5 Thẳng Tắp | rẽ trái/phải phạt 3, quay đầu phạt 30 |
| 6 Hỏa Tốc | gấp: giống robot 0; không gấp: đoạn đông = 5, đoạn có mái che = 0,7 |
| 7 Nâng Niu | hàng dễ vỡ: đoạn đông = 7, rẽ phạt 1,5, quay đầu phạt 15; không dễ vỡ: giống robot 0 |
| 8 Lề Phải | rẽ phải phạt 0,5, rẽ trái phạt 4,5, quay đầu phạt 9 |
| 9 Tham Lam | không tìm đường: xem bên dưới |

Các quy tắc chung:

- Đường đóng, đi ngược một chiều: cấm với mọi robot. Bậc thang: chỉ robot 4.
- Tiền phạt đổi hướng tính cả ở **bước đầu tiên** so với hướng mũi robot trong ảnh (vì thế hướng mũi rất quan trọng với robot 5, 7, 8).
- Có điểm ghé (via): đi tới via trước, rồi mới tới đích; chi phí là tổng hai chặng.
- Loại địa điểm có 2 bản mà yêu cầu không chỉ rõ: robot chọn bản làm **tổng chi phí của chính nó** nhỏ nhất.
- **Phá hòa** (nhiều bước đầu có chi phí bằng nhau): ưu tiên **đi thẳng > rẽ phải > rẽ trái > quay đầu**, so với hướng mũi robot.

Robot 9 (Tham Lam) không tìm đường. Nó (1) chọn **một** đích là bản gần nhất theo khoảng cách Manhattan
(|Δhàng| + |Δcột|), nếu hòa thì lấy bản đứng trước theo thứ tự (hàng, cột); (2) trong các bước đi hợp lệ, chọn bước
làm khoảng cách Manhattan còn lại nhỏ nhất, hòa thì tránh đoạn đông, hòa nữa thì dùng quy tắc phá hòa chung.

### (b) Kết quả đo thật

Dùng thông tin đúng từ `scenes.json` (`python src/eval_strategy.py`):

| split | macro accuracy | từng robot |
|---|---|---|
| train (20.000 dòng) | **1,0000** | cả 10 robot 1,0000 |
| validation (3.000 dòng) | **1,0000** | cả 10 robot 1,0000 |

Tham số chỉ được dò trên train; validation không dùng để dò mà vẫn đúng 100%, nên tôi tin bộ luật này là đúng luật gốc.

### (c) Câu để giảng lại cho đội

- "Mỗi robot là một người tìm đường ngắn nhất, chỉ khác nhau ở bảng giá: người ghét đông thì tính đoạn đông đắt hơn,
  người ghét rẽ thì tính tiền mỗi lần rẽ."
- "Chúng ta dò bảng giá bằng cách thử nhiều bộ số và đếm xem bộ nào khớp nhãn nhiều nhất; khi đạt 100% trên 2.000 cảnh
  thì gần như chắc chắn đã tìm ra luật thật."
- "So sánh các robot trên cùng một cảnh là manh mối tốt nhất: robot 7 khi chở hàng dễ vỡ đi giống robot 5 nhất, từ đó
  biết nó cũng ghét đổi hướng."

### (d) Giả thuyết đã thử nhưng sai

| giả thuyết | kết quả | bài học |
|---|---|---|
| Robot 4 chỉ là "đi được bậc thang + đoạn đông = 2" | 95,0% | Tách theo thời tiết: trời khô 100%, trời mưa 87% → mưa làm bậc thang "trơn" (phạt +4). |
| Robot 5 tối thiểu số lần rẽ trước, rồi mới tới quãng đường (phạt rẽ = 100) | 88% | Phạt rẽ chỉ là 3; quay đầu mới đắt (30). |
| Robot 5: quay đầu phạt 100 (coi như cấm) | 99,65% | 7 cảnh sai đều là robot chấp nhận quay đầu ở bước đầu → phạt hữu hạn, đúng là 30. |
| Robot 7 (dễ vỡ) chỉ tránh đường đông | 70,6% | Nó giống robot 5 tới 77% → thêm phạt đổi hướng thì lên 100%. |
| Robot 8 phạt rẽ như nhau cho trái và phải | 72–74% | "Lề Phải": rẽ phải gần như miễn phí (0,5), rẽ trái đắt (4,5). |
| Robot 9: lấy khoảng cách Manhattan nhỏ nhất tới **bất kỳ** bản nào của đích | 99,05% | Toàn bộ 19 cảnh sai đều có 2 bản cùng loại → nó chốt một đích trước rồi mới đi. |
| Robot 9 dùng khoảng cách Euclid / Chebyshev | thấp hơn Manhattan | Lưới đi theo hàng/cột nên Manhattan mới đúng. |
| Phạt đổi hướng không tính ở bước đầu | robot 5: 62,9%; robot 8: 72,6% | Hướng mũi robot trong ảnh được dùng ngay từ bước đầu. |
| Phá hòa: thẳng > **trái** > phải > quay đầu | robot 0: 98,2% | Phải đứng trước trái (đúng là 100%). |
| Phá hòa: phải > thẳng > trái > quay đầu | robot 0: 92,7% | Đi thẳng được ưu tiên nhất. |

---

## 2. NLP: đọc yêu cầu (`src/nlp2.py`, `src/nlp_tagger.py`; bản đầu ở `src/nlp.py`)

### (a) Làm gì, vì sao

Chiến thuật cần 6 trường: `goal`, `via`, `goal_ref`, `via_ref`, `urgent`, `fragile`.

**Bài học lớn nhất của cả bài nằm ở khâu này.** Bản đầu (`nlp.py`) chỉ nhận ra những tên gọi địa điểm đã có trong từ điển.
Trên validation nó rất tốt, nhưng khi chạy trên test thì bộ đếm của pipeline cho thấy 26% cảnh test không có lần nhắc nào
được nhận là đích (validation: 0,3%). Tức là **test dùng tên gọi và khung câu mà cả train lẫn validation đều không có**.
Tôi không được đọc câu test, nên cách đúng là: làm cho NLP chịu được cách nói lạ, và đo bằng "học train → đo validation"
(validation có 2/8 tên gọi mỗi loại và nhiều khung câu mà train không có, giống tình huống test).

Bản 2 gồm các bước:

1. **Chuẩn hóa**: chữ thường, bỏ dấu (`đ`→`d`), tách câu con theo `. ! ?`, bỏ lời chào và mã đơn.
2. **Từ điển tên gọi, học tự động** (như bản đầu). Câu gây nhiễu có khung sạch ("khong can ghe X", "X khong phai diem nhan"…)
   nên X là một tên gọi; loại của X suy ra từ nhãn goal/via. So khớp mờ theo từng từ để chịu lỗi gõ.
3. **Bộ gán nhãn cụm địa điểm theo ngữ cảnh** (`nlp_tagger.py`), để tìm cả tên gọi **chưa từng gặp**. Mỗi từ được hỏi
   "có thuộc tên địa điểm không?" dựa vào các từ khung xung quanh ("tới", "ghé", "ở", "trước", "giúp mình"…), từ mở đầu kiểu
   "phòng / khu / nơi / nhà", và dấu câu. Từ hiếm luôn bị thay bằng `UNK`, và khi huấn luyện thì chính tên địa điểm cũng bị
   che đi, để mô hình buộc phải học ngữ cảnh chứ không học thuộc tên.
4. **Đoán loại của tên gọi lạ**: n-gram ký tự so với tên đã biết + **bảng từ khóa tiếng Việt viết tay** ("sách", "khám",
   "xe", "ăn", "cổng"…). Kết quả là một phân bố xác suất chứ chưa chốt.
5. **Chốt loại bằng bản đồ** (`resolve_with_map`): chỉ xét các loại thật sự có trên bản đồ do CV đọc; loại đã bị nhắc
   trong câu gây nhiễu thì bỏ; nếu đích có tham chiếu phương hướng thì loại đó phải có 2 bản trên bản đồ; mốc gần/xa phải
   là loại chỉ có 1 bản.
6. **Vai trò của từng lần nhắc** (đích / điểm ghé / gây nhiễu / mốc): che tên địa điểm thành `TGT`/`PLC`, thay từ hiếm
   (món hàng, người nhận) bằng `X`, rồi TF-IDF + hồi quy logistic. Thêm ba luật tổng quát cho khung câu lạ:
   - câu phủ định hoặc kể chuyện đã qua ("không cần…", "đừng…", "bỏ qua…", "hôm qua…", "đã rời…") ⇒ gây nhiễu;
   - **luật cấu trúc**: trong các địa điểm không bị phủ định và không phải mốc, nếu chỉ có một thì nó là đích; nếu có hai
     thì một là đích, một là điểm ghé, phân xử bằng "trước khi / trước / xong / rồi / sau đó" và động từ đi kèm
     (ghé, lấy… so với giao, mang, tới…);
   - cụm đứng sau từ chỉ món hàng ("hộp thuốc", "thẻ thư viện") không phải địa điểm.
7. **Tham chiếu không gian**: bộ phân loại trên vài từ sau tên địa điểm + luật từ khóa ("phía/bên/mạn + bắc/nam/trên/dưới…",
   "gần/sát/cạnh X", "xa/cách xa X", "không xa X" = gần).
8. **Gấp / dễ vỡ**: bảng cụm từ học từ dữ liệu (câu phủ định tự thành cụm trung tính) → khớp mờ → luật từ khóa có phủ định.

### (b) Kết quả đo thật (`python src/eval_nlp2.py`)

"ALL" = cả 6 trường cùng đúng. "Điểm" = macro accuracy cuối cùng khi bản đồ được cho đúng, chỉ NLP là dự đoán.

| cách đo | bản | goal | via | goal_ref | via_ref | urgent | fragile | ALL | điểm |
|---|---|---|---|---|---|---|---|---|---|
| học train → đo validation (**cách nói lạ**) | đầu | 0,787 | 0,800 | 0,707 | 0,890 | 0,850 | 0,933 | 0,393 | 0,8760 |
| học train → đo validation (**cách nói lạ**) | 2 | 0,993 | 1,000 | 0,987 | 0,993 | 1,000 | 1,000 | 0,973 | **0,9957** |
| học train → đo train | 2 | 1,000 | 1,000 | 0,999 | 0,999 | 1,000 | 1,000 | 0,998 | 0,9995 |
| học train + 4/5 validation → đo 1/5 còn lại | đầu | 0,993 | 0,990 | 0,987 | 0,990 | 1,000 | 1,000 | 0,973 | 0,9927 |
| học train + 4/5 validation → đo 1/5 còn lại | 2 | 0,993 | 0,993 | 0,983 | 0,990 | 1,000 | 1,000 | 0,970 | 0,9900 |

Các số riêng lẻ:

- Bộ gán nhãn học trên train tìm ra 266/271 (98%) tên gọi chỉ có ở validation; 27/301 cụm nó tìm ra không phải địa điểm.
- Thí nghiệm "giả vờ bộ phân loại không biết khung câu điểm ghé nào" (`scratch/z8_ablate.py`): chỉ riêng luật cấu trúc đã
  tìm lại điểm ghé đúng 99,3% (train và validation), điểm 0,998 / 0,993.

**Cảnh báo trung thực:** con số 0,9957 hơi lạc quan. Tôi đã nhìn các câu validation bị sai khi viết luật và bảng từ khóa,
nên validation không còn là "cách nói hoàn toàn chưa thấy" đối với các luật đó. Với test, độ đúng của khâu đoán loại tên
gọi lạ là điều tôi **không đo được**.

### (c) Câu để giảng lại cho đội

- "Đừng hỏi 'câu này nói về nơi nào', hãy hỏi 'mỗi nơi được nhắc đóng vai gì': đích, điểm ghé, gây nhiễu hay mốc."
- "Tên địa điểm có thể mới, nhưng các từ quanh nó thì cũ. Mô hình tìm địa điểm bằng ngữ cảnh, đoán loại bằng từ khóa,
  rồi để bản đồ chốt: loại nào không có trên bản đồ thì không thể là đáp án."
- "Điểm validation cao chưa chắc điểm test cao. Phải có một phép đo mô phỏng 'gặp thứ chưa học' (ở đây: học train, đo validation)."

### (d) Giả thuyết đã thử nhưng sai

| giả thuyết | kết quả | bài học |
|---|---|---|
| **Test dùng cùng bộ tên gọi với validation** (nên học train + validation là đủ) | Ở test chỉ 64,5% cảnh có đích là tên gọi đã biết (validation 99%); bản đầu chỉ tìm thấy điểm ghé ở 28% số cảnh test, bản cuối 50,4% (validation: 50,3%) | Sai lầm lớn nhất. Phải thiết kế cho cách nói lạ ngay từ đầu. |
| Train là đủ để đọc validation (bản đầu) | ALL = 0,393; điểm 0,876 | Từ điển đóng không đủ. |
| So khớp mờ trên cả cụm (cho phép lệch 2 ký tự) | 329 lần "ben trong" bị nhận là "bep truong", 49 lần "cong van" → "cong vao" | So theo từng từ và loại các cụm quen thuộc. |
| Lấy mốc gần/xa là địa điểm kế tiếp bất kỳ trong câu | "cán bộ phòng đào tạo ở khu hành chính nằm gần cổng A" → mốc = chính nó | Mốc phải đứng cách lần nhắc ≤ 3 từ. |
| Bộ gán nhãn giữ nguyên dạng mọi từ đã gặp | Tên gọi có trong dữ liệu nhưng chưa vào từ điển bị học thuộc là "không phải địa điểm" (xác suất ≈ 0) | Chỉ từ rất phổ biến mới được giữ nguyên dạng; còn lại là `UNK`. |
| Tự sửa lỗi gõ cho mọi từ lạ (đổi về từ quen cách 1 ký tự) | Từ mới hợp lệ bị sửa bậy: "xong" → "cong", "khoe" → "khoa"; điểm tụt | Chỉ sửa lỗi gõ ở chỗ có ngữ cảnh rõ (tên trong từ điển, chữ "không"). |
| Câu chứa "dung" là câu "đừng…" | "dụng cụ" bỏ dấu cũng là "dung cu": điểm train tụt còn 0,9835 | Bỏ dấu làm nhiều từ trùng nhau ("ngay"/"ngày", "nằm"/"nam"); luật phải kèm từ đi sau. |
| Cụm lạ nào bộ gán nhãn tìm ra cũng là địa điểm | Cụm gấp lạ ("ưu tiên số một"), món hàng ("hộp thuốc"), động từ lạ ("vận chuyển") thành "điểm ghé" giả; điểm 0,9957 → 0,9833 | Cụm lạ phải mở đầu bằng danh từ chỉ nơi chốn hoặc có từ khóa mạnh, và không đứng sau từ chỉ món hàng. |
| Mốc gần/xa không thể trùng loại với điểm ghé | "…xa chỗ để xe hơn… tạt qua khu đậu xe" bị đoán sai | Mốc chỉ bị loại khỏi ứng viên cho đích, không loại khỏi điểm ghé. |

---

## 3. CV: đọc ảnh thành đồ thị (`src/cv_infer.py`, `src/cvfeat.py`, `src/mlp.py`)

### (a) Làm gì, vì sao

Máy chỉ có CPU và Windows chặn PyTorch, nên tôi không dùng CNN. Thay vào đó: **cắt mảnh ảnh nhỏ đúng chỗ, chuẩn hóa
kích thước, rồi cho một mạng MLP nhỏ (viết bằng numpy) phân loại**. Có 6 mạng, tổng cộng khoảng 10,4 triệu tham số
(giới hạn của đề là 200 triệu).

| bước | mô hình | đầu vào | đầu ra |
|---|---|---|---|
| 1. Bộ dò quét dày | `det` | cửa sổ 48px + ngữ cảnh 96px, trượt mỗi 4px trên cả ảnh | nền / giao lộ / địa điểm / robot / dòng chú giải / biểu tượng thời tiết |
| 2. Tách chú giải | `swatch` | mẫu ký hiệu + dòng chữ bên phải (chuẩn hóa theo cỡ chữ) | dòng này nghĩa là gì (18 loại) hay "không phải chú giải"; kiểu nét của mẫu; chữ thời tiết |
| 3. Khớp lưới | (hình học) | tọa độ các giao lộ | góc xoay, khoảng cách hàng/cột, (hàng, cột) của từng giao lộ |
| 4. Đoạn đường | `edge` | dải ảnh giữa hai giao lộ kề nhau, xoay về nằm ngang, co theo cỡ lưới | kiểu nét (không có / 3 kiểu / đóng), bậc thang, một chiều |
| 5. Căn tâm | `align` | mảnh quanh giao lộ | giao lộ đang lệch bao nhiêu so với tâm ký hiệu |
| 6. Giao lộ | `node` | mảnh đã căn tâm, co theo cỡ lưới | thường / robot + 4 hướng mũi / 10 loại địa điểm |
| 7. Thời tiết | `weather` | mảnh 80px quanh biểu tượng | mưa / khô |

Ba ý quan trọng:

1. **Đọc chú giải của chính ảnh đó.** Mô hình `edge` không đoán "thường/đông/mái che" mà đoán **kiểu nét** (ví dụ "nét đỏ",
   "nét đứt"). Mô hình `swatch` đọc chữ của từng dòng chú giải để biết dòng nào là "đường thường", "đông người", "có mái
   che", và nhìn mẫu của dòng đó để biết nó được vẽ bằng kiểu nét nào. Ghép hai thứ lại mới ra trạng thái. Nhờ vậy ảnh bị
   hoán đổi màu vẫn đọc đúng. Chú giải còn cho biết **màu của từng loại địa điểm** (kiểu "sketch" tô màu ngẫu nhiên theo
   từng ảnh) và tập các loại địa điểm có mặt.
2. **Chuẩn hóa kích thước.** Ký hiệu to nhỏ theo khoảng cách lưới (bán kính robot = 0,178 × khoảng cách lưới, sai lệch
   chỉ 0,004), chữ trong chú giải thì co theo từng dòng. MLP không tự chịu được việc phóng to thu nhỏ, nên mọi mảnh ảnh
   đều được co về cùng một cỡ trước khi phân loại.
3. **Âm bản khó.** Chạy bộ dò trên ảnh train, gom những chỗ nó báo nhầm, thêm vào dữ liệu và huấn luyện lại.

Dữ liệu train được tăng cường bằng một bản "xuống cấp" (làm mờ + nén JPEG) của mỗi ảnh sạch, vì validation/test nhiễu hơn train.

### (b) Kết quả đo thật

Mô hình học trên **train**, đo trên **validation** (300 ảnh, 23.006 đoạn đường).

Từng mô hình, khi được cho vị trí đúng:

| mô hình | độ đúng trên validation |
|---|---|
| `det` (từng cửa sổ) | 0,994 |
| `edge`: kiểu nét / bậc thang / một chiều | 0,99996 / 0,9996 / 0,9994 |
| `node` (lệch tâm nhỏ) | 0,9969 |
| `swatch`: ý nghĩa dòng / kiểu nét mẫu / chữ thời tiết | 0,9919 / 0,9956 / 0,9898 |
| `weather` | 0,9906 |
| `align` (đúng chính xác bậc lệch theo x / y) | 0,830 / 0,874 |

Cả chuỗi CV chạy thật từ ảnh (`python src/eval_cv.py validation dev`):

| thành phần | kết quả |
|---|---|
| Bảng chú giải → hoán vị màu/nét đúng | 300/300 ảnh |
| Thời tiết | 299/300 |
| Vị trí robot / hướng mũi | 299/300 và 300/300 |
| Tập giao lộ đúng hoàn toàn | 294/300 ảnh |
| Đoạn đường | thiếu 19, thừa 5, sai trạng thái 3, sai bậc thang 9, sai một chiều 11 trên 23.006 đoạn |
| Mọi địa điểm đúng loại, đúng chỗ | 282/300 ảnh |
| Toàn bộ world đúng tuyệt đối | 264/300 ảnh |
| **Điểm cuối với CV thật + mission đúng** | **0,9887** |

Thời gian: bộ dò khoảng 6 giây/ảnh trên một nhân CPU (11 nhân chạy song song: 300 ảnh mất khoảng 2,5 phút);
các bước còn lại dưới 0,1 giây/ảnh. Huấn luyện cả 6 mạng khoảng 35 phút
(bản cuối trên train + validation: 2.095 giây), cộng khoảng 60 phút cho hai vòng khai thác âm bản khó của bộ dò.

### (c) Câu để giảng lại cho đội

- "Không cần mạng lớn: cắt đúng chỗ, co về cùng cỡ, rồi một mạng vài lớp là đủ. Phần khó là cắt đúng và co đúng."
- "Mô hình chỉ nói 'đoạn này vẽ nét đỏ'. Nét đỏ nghĩa là gì thì phải hỏi chú giải của chính ảnh đó."
- "Mỗi lần điểm thấp, ta đo từng khâu riêng (giao lộ, đoạn đường, địa điểm, chú giải, thời tiết) để biết sửa khâu nào trước."

### (d) Giả thuyết đã thử nhưng sai

| giả thuyết | kết quả | bài học |
|---|---|---|
| Cắt mảnh giao lộ cỡ cố định là đủ | `node` chỉ 0,975; mũi robot to bị cắt mất | Co mảnh theo khoảng cách lưới. |
| Màu của địa điểm cố định theo loại | Sai với kiểu "sketch": riêng kiểu này chỉ đúng 88%, nhầm TT ↔ YT | Màu thay đổi theo ảnh; phải so với màu mẫu trong chú giải của ảnh đó (đúng 329/331). |
| Bộ dò chỉ cần cửa sổ 48px | Kiểu "print": 76 "mẫu chú giải" giả nằm trên đường (thật chỉ 12) | Thêm cửa sổ ngữ cảnh 96px + âm bản khó: số báo nhầm trên ảnh train giảm từ 105.539 xuống 5.929. |
| Vùng chú giải = khung bao mọi mẫu dò được | Một báo nhầm trên bản đồ làm khung nuốt mất cả mảng giao lộ; điểm chỉ 0,77–0,89 | Đọc thử chữ để xác nhận từng dòng, và không nhận dòng nào làm khung nuốt thêm giao lộ. |
| Đọc chữ chú giải bằng mảnh cắt cố định | ý nghĩa dòng chỉ đúng 0,934 | Căn mép trái chữ → 0,949; co theo cỡ chữ → 0,984; thêm lớp "không phải chú giải" → 0,992. |
| Cỡ chữ chú giải tỉ lệ với khoảng cách dòng | tương quan chỉ 0,5 | Chữ co theo từng dòng; phải đo chiều cao mực của chính dòng đó. |
| Biểu tượng thời tiết luôn ở góc ảnh | Sai ở 13/300 ảnh, còn làm mất robot | Ở nhiều ảnh, biểu tượng nằm ngay trong dòng "Thời tiết" của chú giải. |
| Chỉ đọc được 2/3 dòng đường thì suy ra dòng thứ ba | sai 4/14 ảnh | Kết hợp chữ với bố cục (ba dòng luôn liền nhau, đúng thứ tự) → 300/300. |
| Căn tâm cho mọi giao lộ | giao lộ thường bị phân loại sai thêm (sketch: 0 → 15 lỗi) | Chỉ căn tâm cho robot và địa điểm. |
| Thử nhiều vị trí lệch rồi lấy trung bình / lấy chỗ tự tin nhất | không tốt hơn, có kiểu vẽ còn tệ hơn | Mô hình không đáng tin khi lệch tâm; đừng dùng độ tự tin của nó để chọn vị trí. |
| Mọi cụm tọa độ đều là một hàng/cột của lưới | Vạch bậc thang bị tưởng là giao lộ tạo ra "cột ma", lệch chỉ số nửa bản đồ | Cụm nhỏ nằm lưng chừng giữa hai cột là báo nhầm, loại đi. |

Lỗi còn lại (6/300 ảnh validation làm sai bước đi): nhãn chữ kiểu "print" khi ảnh xoay hoặc mờ (3 ảnh), kiểu "night"
sai đoạn đường hoặc sót giao lộ (2 ảnh), sai thời tiết (1 ảnh).

---

## 4. Ghép và sinh bài nộp (`src/pipeline.py`, `src/check_submission.py`)

### (a) Làm gì, vì sao

- Mỗi cảnh có 10 dòng dùng chung ảnh và yêu cầu, nên ảnh và yêu cầu chỉ được xử lý **một lần**, lưu cache
  (`cache/det_*.pkl`, `cache/world_*.pkl`, `cache/nlp2_*.pkl`) rồi cả 10 robot dùng lại.
- CV chạy song song trên 11 nhân CPU. Sửa phần hình học, NLP hay chiến thuật thì không phải quét lại ảnh, chỉ mất vài giây.
- CV và NLP gặp nhau ở bước `resolve_with_map`: loại địa điểm mà NLP chưa chắc được chốt bằng các loại CV thấy trên bản đồ.
  Nếu không tìm được đường thì đi một bước hợp lệ theo thứ tự phá hòa.
- Bản nộp cuối dùng mô hình học trên **train + validation** (CV và NLP). Test chỉ được dùng để chạy dự đoán.
- `check_submission.py` kiểm tra định dạng file, đếm tham số, và in **bộ đếm sức khỏe**: những con số tổng hợp do chính
  pipeline sinh ra trên validation và trên test (bao nhiêu % cảnh có đích là tên gọi đã biết, có điểm ghé, không tìm được
  đường…). Không in câu hay ảnh test, không dùng để huấn luyện. Chính bộ đếm này đã báo động rằng test khác validation.

### (b) Kết quả đo thật

Validation, macro accuracy (`python src/pipeline.py eval dev`). CV học trên train. Khi dự đoán không dùng `scenes.json`.

| cấu hình | điểm |
|---|---|
| bản đồ đúng + yêu cầu đúng (trần) | 1,0000 |
| bản đồ đúng + NLP 5-fold | 0,9900 |
| bản đồ đúng + NLP chỉ học train (cách nói lạ) | 0,9957 |
| CV + yêu cầu đúng | 0,9887 |
| **CV + NLP 5-fold (cách nói đã quen)** | **0,9807** |
| **CV + NLP chỉ học train (cách nói lạ)** | **0,9863** |

Từng robot (CV + NLP 5-fold): R0 0,980 · R1 0,980 · R2 0,980 · R3 0,977 · R4 0,980 · R5 0,977 · R6 0,980 · R7 0,983 · R8 0,980 · R9 0,990.
Từng robot (CV + NLP chỉ học train): R0 0,987 · R1 0,983 · R2 0,987 · R3 0,980 · R4 0,987 · R5 0,983 · R6 0,987 · R7 0,987 · R8 0,987 · R9 0,997.

Bộ đếm sức khỏe của lượt chạy cuối (`python src/check_submission.py final`):

| bộ đếm (tỉ lệ trên số cảnh) | validation | test |
|---|---|---|
| đích là tên gọi đã biết | 99,0% | 64,5% |
| có điểm ghé | 50,3% | 50,4% |
| có tham chiếu không gian cho đích | 37,7% | 31,0% |
| gấp | 38,3% | 31,8% |
| dễ vỡ | 36,7% | 40,6% |
| robot 0 không tìm được đường | 0,0% | 1,8% |
| chú giải đọc được bằng chữ + bố cục | 99,3% | 99,2% |
| ảnh CV bị lỗi | 0 | 0 |

Cách đọc: phần ảnh ở test "khỏe" như validation (chú giải đọc được 99,2%). Phần chữ thì khác hẳn: hơn 1/3 số cảnh test
có đích là tên gọi lạ. Tỉ lệ "gấp" và "tham chiếu" ở test thấp hơn validation 6–7 điểm, nhiều khả năng do còn cụm từ lạ
chưa nhận ra. Vì vậy **điểm test thật sẽ thấp hơn 0,98**, và thấp hơn bao nhiêu thì chỉ bảng xếp hạng mới cho biết.

File nộp: `outputs/predictions.json` (12.000 số nguyên 0..3, đúng thứ tự `test/observations.json`).
Tổng tham số: 10.497.175 (6 mạng MLP cho ảnh: 10.428.180; 6 hồi quy logistic cho chữ: 68.995).

### (c) Câu để giảng lại cho đội

- "Thay lần lượt từng khâu bằng đáp án đúng; khâu nào thay vào mà điểm tăng nhiều nhất là khâu cần sửa trước."
- "Không được nhìn test, nhưng được đếm: nếu tỉ lệ 'đích là tên gọi đã biết' ở test thấp hẳn so với validation thì biết
  ngay mô hình đang gặp thứ nó chưa học, dù không đọc một câu test nào."
- "Hai mô hình độc lập (ảnh và chữ) kiểm tra chéo nhau: chữ đoán loại địa điểm, ảnh cho biết loại nào thật sự có mặt."

### (d) Giả thuyết đã thử nhưng sai

| giả thuyết | kết quả | bài học |
|---|---|---|
| Lần chạy end-to-end đầu tiên đã dùng được | 0,7707 | Phải soi từng khâu; lỗi lớn nhất khi đó là khung chú giải nuốt bản đồ. |
| So sánh bản đồ dự đoán với bản đồ thật bằng cách dời gốc về ô nhỏ nhất | một điểm báo nhầm ở rìa làm bảng đo báo "sai 141 đoạn" dù bước đi vẫn đúng | Khớp theo vị trí pixel trước khi so; số lỗi thật là 19 đoạn thiếu. |
| Điểm validation 0,983 (bản NLP đầu) là ước lượng tốt cho test | Bộ đếm sức khỏe cho thấy 26% cảnh test không có lần nhắc nào được nhận là đích | Validation chỉ đại diện cho test nếu hai bên cùng phân bố; phải kiểm tra điều đó bằng số đếm. |

---

## 5. Giới hạn còn lại

1. **Tên gọi địa điểm lạ ở test** (khoảng 35% số cảnh): loại được đoán bằng từ khóa + n-gram ký tự + bản đồ. Độ đúng của
   bước này trên test không đo được; tên nào không chứa từ khóa trong bảng viết tay sẽ dễ sai.
2. **Cụm từ gấp và tham chiếu không gian lạ**: tỉ lệ ở test thấp hơn validation 6–7 điểm. "Gấp" chỉ ảnh hưởng robot 6;
   tham chiếu ảnh hưởng mọi robot khi có hai địa điểm cùng loại.
3. **Khung câu gây nhiễu lạ** không chứa từ phủ định quen thuộc sẽ bị luật cấu trúc hiểu thành điểm ghé.
4. **Nhãn chữ kiểu "print"** khi ảnh xoay hoặc mờ: khoảng 3% nhãn kiểu này bị đọc sai (bước căn tâm đôi khi trượt).
5. **1,8% cảnh test không tìm được đường** cho robot 0 (validation: 0%): đồ thị hoặc đích đang sai ở các cảnh đó;
   pipeline khi ấy đi một bước hợp lệ theo thứ tự phá hòa.
6. Số đo "cách nói lạ" (0,9957 / 0,9863) hơi lạc quan vì các luật được viết sau khi xem lỗi trên validation.

---

## 6. Nếu thi lại một đề cùng dạng: thứ tự và thời gian

Giả sử có khoảng 10 giờ.

| thứ tự | việc | thời gian | ghi chú |
|---|---|---|---|
| 1 | Đọc đề, in vài phần tử dữ liệu, xem ảnh của từng kiểu vẽ; **kiểm tra thư viện chạy được** | 30 phút | Tìm ngay: chú giải nằm ở đâu, có hoán đổi không, ký hiệu to nhỏ theo cái gì. |
| 2 | **Chiến thuật trước**, dùng thông tin đúng | 1,5 giờ | Một bộ máy tìm đường có tham số, dò lưới song song, soi cảnh sai. Chưa đạt ~100% thì chưa làm việc khác. |
| 3 | Khung end-to-end thô + bảng đo từng khâu + file nộp thử + **bộ đếm sức khỏe trên test** | 1 giờ | Nộp thử một bản sớm để có điểm bảng xếp hạng làm mốc. |
| 4 | NLP **thiết kế cho cách nói lạ ngay từ đầu** | 2 giờ | Tìm cụm theo ngữ cảnh, đoán loại bằng từ khóa, chốt bằng bản đồ; đo bằng "học train → đo validation". |
| 5 | CV | 3,5 giờ | Thứ tự: bộ dò → khớp lưới → đoạn đường → chú giải → giao lộ. Chuẩn hóa kích thước ngay từ đầu. |
| 6 | Vòng cải tiến theo bảng đo và theo điểm bảng xếp hạng | 1 giờ | Luôn sửa khâu đang làm mất nhiều điểm nhất. |
| 7 | Huấn luyện bản cuối (train + validation), chạy test, kiểm tra file nộp | 1 giờ | |

Bốn việc nên làm khác đi so với lần này:

1. **Đừng tin rằng test giống validation.** Chạy pipeline trên test thật sớm và so các bộ đếm tổng hợp với validation.
   Lần này việc đó chỉ được làm ở cuối, nên bản NLP đầu tiên được xây cho sai mục tiêu.
2. Cache kết quả bộ dò ngay từ đầu. Trước khi có cache, mỗi lần thử mất 2,5 phút; sau đó chỉ còn 6 giây.
3. Kiểm tra môi trường trước khi thiết kế. Lần này PyTorch bị Windows chặn nên phải viết mạng nơ-ron bằng numpy.
4. Vẽ kết quả lên ảnh ngay khi một khâu sai. Phần lớn lỗi CV lớn được tìm ra bằng cách nhìn một ảnh, không phải bằng số.
