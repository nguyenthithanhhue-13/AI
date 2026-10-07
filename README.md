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
v19 được **0,9778** trên bảng xếp hạng. Lượt 2–3 (06/10): NLP chịu được khung câu mới tự soạn (`scratch/h27_paraphrase.py`:
146 phép thử đạt 1,0000), CNN đoạn đường học thêm ảnh làm méo mạnh, chiến thuật robot được xác nhận "ghim" chặt
(`scratch/h26_policy_margin.py`).
Lượt 4 (06/10 chiều, xem đầu CAP_NHAT_06-10.md): đoán loại tên gọi mới (`TYPE_FIX`), món hàng không cướp vai địa điểm /
không mang cờ (`ITEM_FIX`), tắt cờ dễ vỡ do riêng mô hình nghĩa e5 bật (`E5_POS_FRAGILE`), "bên tay phải" / "mé trái"
(`SPATIAL_FIX2`). Validation đo công bằng nhất (CV dev + NLP chỉ học train, `scratch/h34_fair_e2e.py`): **1,0000**.
Bảng xếp hạng: v31c = 0,9733 (< v28 = 0,9778) cho thấy cờ dễ vỡ do e5 bật trên test phần lớn đúng -> đã bật lại (v32a).
Lượt 5 (06/10 khuya, xem đầu CAP_NHAT_06-10.md): phép thử mới đặt tên lạ vào câu validation thật
(`scratch/h48_alias_swap.py`) tìm ra các lỗi hệ thống về tham chiếu rõ ràng, chọn cặp đích / điểm ghé, khung "Hàng cho X:",
dấu tiếng Việt khử nhập nhằng, ranh giới tên, cờ dễ vỡ cách nói mới.
File nộp mới: `outputs/predictions.json` (xem bảng "Điểm bảng xếp hạng" bên dưới). Các bảng kết quả validation bên dưới là
số đo cũ.

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
| **`predictions_v41_negref.json`** (ứng viên) | v35 + đọc câu đợt 20 (`src/nlp2.py`; mã v35: `nlp2_v35_src.py.bak`): tham chiếu gần / xa BỊ PHỦ ĐỊNH ("X không xa Y" = gần, "không ở gần Y" = xa — 0 câu trong train + validation, bộ phân loại đọc ngược; `REF_NEG_FIX`), giới từ "sang / vào / về / ra" (`PREP_CANON`) và các sửa tổng quát khác (CAP_NHAT đợt 20). Bản đồ test + bộ đọc đã lưu của v35. Khác v35: 20 bản đồ / 120 dòng | chưa nộp |
| `predictions_v40_cv_multi.json` | v35 + CV tăng cường NHIỀU LỚP (đợt 19, `scratch/make_v40.sh`): CNN tinh chỉnh trên ảnh xoay / thu nhỏ / mất độ phân giải / mờ / đổi màu-tương phản / nhiễu / JPEG chồng nhau, MLP căn tâm + giao lộ học lại, CNN đọc mọi đoạn đường, gộp CNN đoạn đường cũ + mới (`models_finalm`). Bền hơn hẳn trên validation làm méo nặng; trên test chỉ đổi 2 bản đồ (đoạn đường) và **0 dòng dự đoán** -> trùng v35 (xem CAP_NHAT đợt 19) | không nộp (= v35) |
| `predictions_v39_refit_vote.json` | v35 + bỏ phiếu giữa 7 lần học lại bộ đọc câu: khác v35 1 cảnh / 5 dòng (tung đồng xu) | không nộp |
| `predictions_v38_tta_vote.json` | v35 + bỏ phiếu CV đa nhiễu: trùng v35 | không nộp |
| `predictions_v35_cand.json` (= `predictions.json`) | v34c, bộ đọc học lại bằng mã mới | **0,9828** (robot yếu nhất 0,9722) — bài được tính |
| `predictions_v36C.json`, `predictions_v36A.json` | A/B: tắt `CACH_RULE` + `GOAL_FRAME_RULE` + `REACH_RULE` / tắt `NEW_FRAME_FIX` | 0,9772 / 0,9800 (các luật đó ĐÚNG trên test) |
| `predictions_v34c_spans_flags.json` | v33 + đợt 17: sửa lỗi của v33 ("Nhớ né X ra" bị thành điểm ghé), khung "Điểm giao: X.", chủ ngữ "Ban quản lý / Cán bộ <nơi>", cụm món hàng không phải địa điểm ("trứng gà"), "cổng phía trước", "trạm xá", dấu khử nhập nhằng ở mép tên, cờ dễ vỡ / gấp cách nói mới (mã mới + bộ đọc đã học của v32b). Khác v32b: 45 dòng / 16 cảnh | **0,9828** |
| `predictions_v33_ref_frames_flags.json` | v32b + lượt 5 (CÓ LỖI "né X" -> thay bằng v34c): tham chiếu rõ ràng, chọn cặp đích / điểm ghé, khung "Hàng cho X:", dấu khử nhập nhằng, ranh giới tên, cờ cách nói mới (mã mới + bộ đọc đã học của v32b). Khác v32b: 35 dòng / 11 cảnh | chưa nộp |
| `predictions_v33r_refit.json` | như v33 nhưng học lại bộ đọc bằng mã mới: thêm 4 cảnh đổi do các quyết định vai trò sát nút (≈ tung đồng xu). Khác v32b: 56 dòng / 15 cảnh | dự phòng |
| `predictions_v32b_cv_strong_final.json` (= v32a) | v32a + hai CNN chú giải / giao lộ mới (làm méo mạnh): bản đồ test không đổi | **0,9828** (cao nhất) |
| `predictions_v32a_nlp_via_e5on.json` | v31c nhưng BẬT LẠI cờ dễ vỡ của e5 + `VIA_FIX` (dấu ";" là ranh giới vế, phủ định / "khỏi" / "chớ", "từ X sang Y"...). Khác v28: 92 dòng / 20 cảnh | (= v32b) |
| `predictions_v31c_items_spatial_e5frag.json` | v30 + tắt cờ dễ vỡ do riêng e5 bật + "bên tay phải" / "mé" + sửa hồi quy "giao là X ở", "đến tận X" (khác v28: 64 dòng / 50 cảnh, 44 dòng của robot 7) | **0,9733** (tắt cờ e5 là sai) |
| `predictions_v30_typing_items.json` | v28 + đoán loại tên mới + món hàng (khác v28: 37 dòng / 11 cảnh; có hồi quy nhỏ đã sửa ở v31) | không nộp |
| `predictions_v29_fragile_ambiguity.json` | v28 + cờ dễ vỡ theo ngữ cảnh ("cẩn thận kẻo trễ") — trùng v28 trên test | không nộp |
| `predictions_v28_appositive.json` (= v27c = v27 trên test) | v26 + khung nhãn thứ tự cho mọi vai trò, "A trước, B sau", "đến X lấy" (trừ khi người nhận tự tới), đồng vị ngữ "X, nơi…" (khác v26 ở 2 cảnh test) | **0,9778** |
| `predictions_v26_order_frames.json` | v23 + lỗi gõ theo cặp từ / dấu, câu tương phản, gây nhiễu mới, thứ tự hai chặng (khác v23 ở 2 cảnh test) | chưa nộp |
| `predictions_v23_typing_knn_desc_more2.json` (= v24 = v25 trên test) | v22 + đoán loại tên mới (láng giềng + mô tả + CTX 0,8) + kho tên 750 mục | chưa nộp |
| `predictions_v22_nlp_newframes_edgestrong.json` | v19 + NLP khung câu mới / lỗi gõ chữ then chốt + CNN đoạn đường làm méo mạnh + sửa CV lượt 2 | chưa nộp |
| `predictions_v21_nlp_frames_knowledge.json` | v19 + NLP dò tên lạ theo khung câu + ~350 cách gọi viết tay | chưa nộp |
| `predictions_v19_cnn_cv_typo.json` | 4 CNN cho CV + sửa lỗi gõ NLP; train và validation 1,0000 | **0,9778** |
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

File nộp: `outputs/predictions.json` = `outputs/cac_ban_nop_cu/predictions_v35_cand.json` (bài được tính, 0,9828). Tạo lại
từ mô hình đã lưu (`outputs/models_final/`, `outputs/nlp2_final.pkl`, `outputs/e5_small/`): `python src/pipeline.py test final`
(nhớ xóa `cache/det_test_final.pkl`, `cache/world_test_final.pkl`, `cache/nlp2_test_final_final.pkl` nếu muốn tính lại từ
đầu). Kiểm tra tái lập không đụng tới cache / file nộp: `PROCS=6 python scratch/h87_repro.py` (chạy lại toàn bộ suy luận
dưới tên tạm `finalrep` rồi so từng dòng với v35). Các bước huấn luyện CNN ở CAP_NHAT_06-10.md. Bản cũ để so sánh:
`outputs/cac_ban_nop_cu/predictions_v1_nlp_lexicon_only.json` (NLP bản đầu, chỉ nhận tên gọi đã có trong từ điển).
Phần tăng cường nhiều lớp của đợt 19 (`cv_data.py ... multi`, `scratch/train_multi.sh`, `models_finalm`, `EDGE_CNN_TH`,
`edge2/`) là tùy chọn, KHÔNG dùng trong bài nộp (mặc định của mã vẫn cho đúng v35).

Ứng viên v41 (đợt 20): `src/nlp2.py` hiện là mã đợt 20. Tạo lại `predictions_v41_negref.json` từ bản đồ test đã tính +
bộ đọc đã lưu: `python scratch/make_v41.py v41_negref` (bước (a); bước (b) học lại bộ đọc chỉ để đối chiếu — học lại
bằng mã v35 gốc cũng lệch 4 bản đồ so với bộ đọc đã lưu, nên luôn dùng `outputs/nlp2_final.pkl`). Với mã đợt 20,
`python src/pipeline.py test final` (sau khi xóa `cache/nlp2_test_final_final.pkl`) cho đúng v41.

## Cấu trúc thư mục

```
src/        code chính
scratch/    các thử nghiệm (dò tham số chiến thuật, soi lỗi...)
outputs/    predictions.json, mô hình đã huấn luyện
cache/      kết quả trung gian theo cảnh (mảnh ảnh, world, mission đã đọc)
```
