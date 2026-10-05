# Phenikaa Campus Courier v2 — lời giải

## Tóm tắt bài

- Có 10 robot, mỗi robot có một chiến thuật di chuyển cố định. Mỗi **cảnh** gồm một ảnh sơ đồ campus và một yêu cầu
  tiếng Việt; cảnh sinh ra 10 dòng (robot 0..9). Cần đoán **bước đi đầu tiên** của từng robot:
  `0=UP, 1=DOWN, 2=LEFT, 3=RIGHT` (hướng tuyệt đối trên ảnh).
- Ảnh: lưới giao lộ 5..9 hàng/cột (có giao lộ và đoạn đường bị thiếu), 4 trạng thái đường (thường / đông / mái che /
  đóng), bậc thang, một chiều, 5..9 địa điểm thuộc 10 loại (có thể trùng 2 bản), robot + hướng mũi, thời tiết.
  Có 4 kiểu vẽ; **chú giải là quy ước của riêng ảnh đó** (màu/kiểu nét có thể bị hoán đổi); ảnh có thể xoay nhẹ, mờ, nén JPEG.
- Yêu cầu: đích (goal), điểm ghé (via, không bắt buộc), tham chiếu không gian (bắc/nam/đông/tây, gần/xa X), mức gấp,
  hàng dễ vỡ; có phủ định, địa điểm gây nhiễu, câu không dấu, lỗi gõ. Validation/test có cách nói không có trong train.
- Điểm = trung bình accuracy của 10 robot (macro accuracy). Test không có `scenes.json`.
- Luật: không gọi API AI bên ngoài khi dự đoán; tổng tham số ≤ 200 triệu; không dùng nội dung test để huấn luyện,
  viết luật hay chỉnh tham số.

## Cách giải (3 khâu, đo riêng từng khâu)

```
ảnh ──CV──► world (đồ thị, địa điểm, robot, thời tiết) ─┐
                                                        ├─► chiến thuật(robot_id) ─► hướng đi
mission ──NLP──► (goal, via, tham chiếu, gấp, dễ vỡ) ───┘
```

| khâu | file | cách làm |
|---|---|---|
| Chiến thuật | `src/strategy.py` | Dijkstra trên (giao lộ, hướng) với bảng giá riêng từng robot, suy ra từ nhãn train |
| NLP | `src/nlp2.py`, `src/nlp_tagger.py` | Từ điển tên gọi khai thác tự động + bộ gán nhãn cụm địa điểm theo ngữ cảnh (đọc được tên gọi lạ) + TF-IDF / hồi quy logistic + luật từ khóa tiếng Việt; loại địa điểm lạ được chốt bằng bản đồ |
| CV | `src/cv_infer.py`, `src/cvfeat.py`, `src/mlp.py` | 6 mạng MLP nhỏ viết bằng numpy trên các mảnh ảnh cắt; đọc chú giải của từng ảnh |
| Ghép | `src/pipeline.py` | Mỗi cảnh xử lý ảnh và mission một lần, cache, dùng lại cho 10 robot |

Chi tiết, số đo và phần "giảng lại cho đội" nằm trong [GIAI_THICH.md](GIAI_THICH.md).

## Cập nhật mới nhất (06/10): train và validation đều 100%

Xem [CAP_NHAT_06-10.md](CAP_NHAT_06-10.md). Tóm tắt: thêm 4 CNN nhỏ (đoạn đường, giao lộ có lớp "không phải giao lộ",
chú giải, thời tiết), sửa cách tìm biểu tượng thời tiết, sửa lỗi gõ chỉ hướng / "không" / tên gọi lệch một từ.
Validation (CV thật + NLP chỉ học train): **1,0000**; train (cấu hình nộp bài): **1,0000**; NLP giấu tên gọi: 0,9973.
File nộp mới: `outputs/predictions.json` (bản sao `outputs/cac_ban_nop_cu/predictions_v19_cnn_cv_typo.json`).
Các bảng bên dưới là số đo cũ.

## Kết quả đo trên validation (macro accuracy)

Khi dự đoán không dùng `scenes.json`. CV học trên train. NLP đo theo hai cách: "5-fold" (mỗi câu được đọc bởi mô hình
không học câu đó, nhưng đã học các câu validation khác) và "chỉ học train" (mô phỏng gặp cách nói hoàn toàn mới).

| cấu hình | điểm |
|---|---|
| bản đồ đúng + yêu cầu đúng (chỉ đo chiến thuật) | 1,0000 |
| bản đồ đúng + NLP 5-fold / NLP chỉ học train | 0,9900 / 0,9957 |
| CV + yêu cầu đúng | 0,9887 |
| **CV + NLP 5-fold** | **0,9807** |
| **CV + NLP chỉ học train** | **0,9863** |

Từng robot (CV + NLP 5-fold): R0 0,980 · R1 0,980 · R2 0,980 · R3 0,977 · R4 0,980 · R5 0,977 · R6 0,980 · R7 0,983 · R8 0,980 · R9 0,990.

**Điểm test thật sẽ thấp hơn các số trên.** Bộ đếm tổng hợp của pipeline cho thấy câu ở test khác validation: chỉ 64,5%
cảnh test có đích là tên gọi đã biết (validation 99%). Với các cảnh còn lại, loại địa điểm được đoán bằng từ khóa và
bản đồ; độ đúng của bước đó trên test không đo được. Xem mục 4 và 5 của GIAI_THICH.md.

## Điểm bảng xếp hạng và các bản nộp

| file trong `outputs/cac_ban_nop_cu/` | nội dung | điểm bảng xếp hạng |
|---|---|---|
| `predictions_v2_lb0.9322.json` | NLP bản 2 (tìm tên gọi lạ theo ngữ cảnh + từ khóa + bản đồ) | **0,9322** (robot yếu nhất 0,8611) |
| `predictions_v19_cnn_cv_typo.json` (= `predictions.json`) | 4 CNN cho CV + sửa lỗi gõ NLP; train và validation 1,0000 | chưa nộp |
| `predictions_v4_e5_cnn.json` | v3a + CNN nhỏ đọc nhãn địa điểm (ghép với MLP); validation "CV + mission đúng" 0,9887 → 0,9917 | chưa nộp |
| `predictions_v3a_e5_negation.json` | thêm mô hình nghĩa pretrained e5 cho tên gọi lạ và câu gấp / dễ vỡ; thời tiết chỉ lấy từ biểu tượng | chưa nộp |
| `predictions_v3b_no_generic_negation.json` | như v3a nhưng tắt luật "từ phủ định chung" | chưa nộp |

Phân tích trên validation (`scratch/a1_sensitivity.py`): đọc sai "gấp" chỉ kéo robot 6 xuống, "dễ vỡ" chỉ kéo robot 7,
thời tiết chỉ kéo robot 3 và 4. Robot yếu nhất 0,8611 nhiều khả năng là robot 6 hoặc 7.
Đo trung thực giá trị của e5 (`scratch/e3_ablate_e5.py`, học train → đo validation, tắt luật từ khóa): 0,9600 → 0,9760
khi e5 chỉ học dữ liệu train, 0,9893 khi thêm các ví dụ viết tay trong `src/nlp_knowledge.py`.

Cần thêm thư viện `onnxruntime`, `tokenizers`, `huggingface_hub` và tải mô hình một lần: `python src/embed.py download`.

## Tuân thủ quy định

- Dự đoán chạy hoàn toàn offline (numpy, OpenCV, scikit-learn); không gọi API nào.
- Tổng tham số 131.244.682 (giới hạn 200 triệu), gồm mô hình pretrained công khai multilingual-e5-small (117.653.760) và 4 CNN nhỏ, tất cả chạy offline bằng onnxruntime.
- Test chỉ được dùng để chạy dự đoán: không huấn luyện, không dò tham số, không đọc câu hay xem ảnh test, không bổ sung
  từ điển từ test. Bảng từ khóa tiếng Việt trong `nlp2.py` được viết từ kiến thức ngôn ngữ chung và từ train/validation.
  Thứ duy nhất lấy từ lượt chạy test là các **bộ đếm tổng hợp** do `check_submission.py` in ra (ví dụ "bao nhiêu % cảnh có
  đích là tên gọi đã biết"); chúng cho biết test khác validation, và từ đó tôi cải tiến NLP rồi đo lại trên train/validation.
  Nếu đội muốn chắc chắn tuyệt đối về quy định, hãy hỏi BTC xem việc dùng bộ đếm như vậy có được phép không.
- Khi dự đoán chỉ dùng ảnh, mission và robot_id.

## Cài đặt

Windows, Python 3.13, chỉ cần CPU.

```powershell
pip install numpy opencv-python pillow scipy scikit-learn
```

Khi dự đoán không cần PyTorch (MLP viết bằng numpy, CNN chạy bằng onnxruntime). PyTorch bản CUDA chỉ cần để huấn luyện
lại các CNN; trên máy này nó được cài ở `F:\pylibs` (xem [CAP_NHAT_06-10.md](CAP_NHAT_06-10.md)).

## Chạy lại từ đầu

```powershell
python src/eval_strategy.py              # kiểm tra chiến thuật với thông tin đúng (100% train và validation)
python src/eval_nlp2.py                  # độ đúng NLP theo từng trường (học train -> đo validation, và 5-fold)

python src/cv_data.py train              # cắt mảnh ảnh huấn luyện (~30 giây)
python src/cv_data.py validation
python src/cv_train.py dev               # học trên train, đo trên validation (~45 phút)
python src/cv_hardneg.py train dev       # khai thác âm bản khó cho bộ dò (~30 phút), rồi huấn luyện lại bộ dò:
python src/cv_train.py dev det           #   (lặp lại hai dòng này 2 lần; lần thứ hai thêm --append vào cv_hardneg.py)
python src/run_cv.py validation dev      # chạy CV trên validation, lưu cache (~2,5 phút)
python src/eval_cv.py validation dev     # độ đúng từng thành phần CV
python src/pipeline.py eval dev          # macro accuracy trên validation, không dùng scenes.json khi dự đoán

python src/cv_train.py final             # học lại trên train + validation
python src/pipeline.py test final        # chạy CV + NLP trên test, sinh outputs/predictions.json (~12 phút)
python src/check_submission.py final     # kiểm tra định dạng file nộp, đếm tham số, in bộ đếm sức khỏe validation/test
```

Lưu ý: muốn chạy lại một bước sau khi sửa code thì xóa file cache tương ứng trong `cache/`
(`det_*.pkl` = kết quả bộ dò, `world_*.pkl` = world cuối, `nlp2_*.pkl` = mission đã đọc) và `outputs/nlp2_final.pkl`
(mô hình NLP cuối). `run_cv.py` luôn tính lại `world_*.pkl` nhưng dùng lại `det_*.pkl` nếu có.

File nộp: `outputs/predictions.json` (bản mới nhất, giống `predictions_v19_cnn_cv_typo.json`; các bước huấn luyện CNN ở CAP_NHAT_06-10.md). Bản cũ để so sánh trên bảng xếp hạng nếu muốn:
`outputs/cac_ban_nop_cu/predictions_v1_nlp_lexicon_only.json` (NLP bản đầu, chỉ nhận tên gọi đã có trong từ điển).

## Cấu trúc thư mục

```
src/        code chính
scratch/    các thử nghiệm (dò tham số chiến thuật, soi lỗi...)
outputs/    predictions.json, mô hình đã huấn luyện
cache/      kết quả trung gian theo cảnh (mảnh ảnh, world, mission đã đọc)
```
