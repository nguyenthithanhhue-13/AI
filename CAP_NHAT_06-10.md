# Cập nhật ngày 05–06/10: train và validation đều đạt 100%

## Kết quả (macro accuracy, khi dự đoán KHÔNG dùng scenes.json)

| phép đo | trước | sau |
|---|---|---|
| validation: CV thật + mission đúng | 0,9917 | **1,0000** |
| validation: CV thật + NLP chỉ học train (cách nói lạ) | 0,9880 | **1,0000** |
| validation: bản đồ đúng + NLP chỉ học train | 0,9957 | **1,0000** |
| train: CV thật + NLP cấu hình nộp bài | — | **1,0000** (cả 10 robot) |
| train: CV thật + mission đúng (mô hình dev, học train) | 0,9965 | 0,9999 |
| NLP "giấu tên gọi" (5 nhóm, train + validation; gần với test nhất) | ~0,985 | **0,9973** |
| NLP 5-fold trên validation | 0,9893 | 0,9943 |

Từng khâu CV trên validation (mô hình học train): giao lộ, đoạn đường, robot, hướng mũi, thời tiết, chú giải
đều 300/300 cảnh đúng. Trên train: tất cả 2000/2000 trừ loại địa điểm (5 cảnh lệch, chỉ 1 cảnh làm sai bước đi).

Chiến thuật 10 robot không đổi (vẫn 100% với thông tin đúng).

## Các thay đổi, theo mức quan trọng

### CV

1. **CNN cho đoạn đường** (`src/cv_edge.py`). MLP cũ sai 1 / 10 / 16 mảnh (kiểu nét / bậc thang / một chiều) trên
   26.738 mảnh validation; CNN sai 0 / 0 / 0. CNN chậm trên CPU nên chỉ được hỏi khi MLP không chắc (xác suất < 0,999,
   khoảng 4,7% số đoạn) — cách này bắt được 23/23 lỗi của MLP.
2. **Lớp "không phải giao lộ" cho CNN giao lộ** (`src/cv_cnn.py`, `src/cv_data.py`). Bộ dò quét dày báo nhầm ở 53/300
   ảnh validation (chữ tiêu đề, vạch bậc thang, mũi tên). Một đỉnh nhầm lọt vào lưới làm gộp hai cột và lệch chỉ số cả
   bản đồ (cảnh 81). Với đỉnh dò sạch thì khớp lưới đúng 300/300. CNN giờ có lớp thứ 16 học từ các mảnh "nhiều mực"
   không phải giao lộ; `drop_nonnodes` loại các đỉnh đó trước khi khớp lưới.
3. **CNN giao lộ được tập với tâm lệch tới 20% unit** (trước 7%), và được đọc thêm một lần tại tâm đã căn. Bộ dò
   hay lệch 10–20px ở nhãn chữ rộng của kiểu "print".
4. **Trọng số CNN khi ghép với MLP: 0,8** (trước là trung bình hình học 0,5). Soi từng ca sai: CNN đúng (≥ 0,95) trong
   mọi ca, MLP sai và tự tin. Train: 0,5 → 0,9977; 0,8 → 0,9999; 1,0 (chỉ CNN) → 0,9963.
5. **CNN đọc chú giải** (`src/cv_swatch.py`): lỗi đọc ý nghĩa dòng từ ~143 xuống 0–1 trên 17.635 dòng validation;
   3 ảnh train bị đọc sai hoán vị màu đường (93 đoạn sai trạng thái) nay đúng hết.
6. **CNN thời tiết** (`src/cv_weather.py`): 0/1800 lỗi. MLP cũ đọc biểu tượng "nhiều mây" thành mưa (cảnh 104).
7. **Vị trí biểu tượng thời tiết**: chỉ ở dòng chú giải hoặc góc trên phải (x/W 0,90–0,97, y/H 0,02–0,09 trên 100%
   ảnh train + validation). Trước đây đĩa vàng của robot bị lấy làm biểu tượng thời tiết rồi bị xóa khỏi danh sách giao lộ
   (cảnh 191: mất robot, cả 10 robot sai).
8. Sửa lỗi: `import os` nằm trong `Models.__init__` làm `os` thành biến cục bộ; khởi tạo mô hình crash và pool treo.

### NLP

1. **Sửa lỗi gõ của từ chỉ hướng** (`nlp.repair_spatial`): "phía ren" → "phía trên", "bên tria" → "bên trái",
   "pia tây" → "phía tây". Chỉ sửa trong khung chỉ hướng; trên 2.300 yêu cầu train + validation sửa đúng 5 chỗ, không
   sửa nhầm chỗ nào.
2. **Lỗi gõ của "không"**: ngưỡng tần suất bảo vệ từ thật tăng 5 → 20. Từ thật gần "không" xuất hiện ≥ 54 lần
   (phong 960, hong 199, chong 54); lỗi gõ ≤ 5 lần (khogn 5, khng 4...). Ngưỡng cũ làm "khôgn cần ghé X" thành điểm ghé.
3. **Khớp mờ tên gọi nhiều từ khi chỉ lệch một từ** ("hong lab" ~ "phong lab"): trước đây bị chặn vì "hong" là từ
   thông dụng. Cụm quen thuộc ("cong van") vẫn bị chặn như cũ.

## Môi trường

- PyTorch bản CUDA cài ở `F:\pylibs` (ổ C gần đầy): chạy huấn luyện với `PYTHONPATH=F:/pylibs`.
  Khi dự đoán chỉ cần `onnxruntime`, không cần PyTorch.
- GTX 1650 4 GB: **không** dùng `cudnn.benchmark`, AMP hay channels_last (chậm 6–10 lần); không để cả mảng dữ liệu
  trên GPU (tràn bộ nhớ, chậm 10 lần).
- Đặt `PYTHONUTF8=1 PYTHONIOENCODING=utf-8` khi chạy trên Windows (code in tiếng Việt).
- `cv_swatch.py` xuất ONNX bằng trình xuất mới (`dynamo=True`, cần gói `onnxscript`, đã cài vào `F:\pylibs`) vì
  `AdaptiveAvgPool2d((1, 4))` không xuất được bằng trình cũ. File `swatch_cnn.onnx` đi kèm `swatch_cnn.onnx.data`
  (trọng số ngoài) — phải giữ cả hai. Đã kiểm tra ONNX khớp PyTorch 100% trên 2.000 dòng chú giải.
- Đường dẫn dữ liệu: `src/common.py` tự tìm `delivery_public` ở thư mục cha nếu không có trong `AI/`.

## File nộp

`outputs/predictions.json` (bản sao `outputs/cac_ban_nop_cu/predictions_v19_cnn_cv_typo.json`): hợp lệ theo
`check_submission.py`, tổng tham số 131.244.682. Khác bản 0,9600 ở 194 / 12.000 đáp án (54 / 1.200 cảnh).
Bản trước khi chạy lượt này được giữ ở `outputs/cac_ban_nop_cu/predictions_truoc_v19.json`.

## Chạy lại

```powershell
python src/cv_data.py train ; python src/cv_data.py validation        # cắt mảnh (gồm cả dữ liệu CNN giao lộ)
$env:PYTHONPATH="F:/pylibs"
python src/cv_cnn.py dev 8 ; python src/cv_edge.py dev 6 ; python src/cv_swatch.py dev 12 ; python src/cv_weather.py dev 14
python src/run_cv.py validation dev ; python src/eval_cv.py validation dev
python src/pipeline.py eval dev                                         # validation end-to-end
python src/run_cv.py train dev ; python scratch/q8_train_final.py       # train end-to-end (cấu hình nộp bài)
python scratch/h1_alias_holdout.py                                      # NLP giấu tên gọi

python src/cv_cnn.py final 8 ; python src/cv_edge.py final 6 ; python src/cv_swatch.py final 12 ; python src/cv_weather.py final 14
python src/pipeline.py test final ; python src/check_submission.py final
```

## Tuân thủ quy định

- Không đọc câu hay xem ảnh test, không gán nhãn, không dùng bộ đếm tổng hợp của test để chọn luật trong lượt này.
  Mọi sửa đổi được tìm ra và đo trên train / validation. Test chỉ dùng để chạy dự đoán.
- Tổng tham số vẫn dưới 200 triệu (xem `python src/check_submission.py final`).

## Lưu ý trung thực

Train và validation 100% không có nghĩa bảng xếp hạng sẽ 100%. Test có tên gọi và khung câu chưa từng gặp
(khoảng 35% cảnh có đích là tên gọi lạ). Phép đo gần với tình huống đó nhất là "giấu tên gọi": 0,9973.

## Bổ sung tối 06/10 (đọc câu): 0,9778 → 0,9806

File tốt nhất: `outputs/predictions.json` = `outputs/cac_ban_nop_cu/predictions_v20_lb0.9806.json` (**0,9806 = 1765/1800**).

### Sửa lỗi làm chương trình dừng

`MissionParser2._frame_spans` sắp xếp các bộ `(a, b, loại, tên)`; khi hai cụm trùng vị trí, Python so `None` với chuỗi và báo
`TypeError: '<' not supported between instances of 'NoneType' and 'str'`. Đã sửa: sắp xếp theo vị trí `(a, b)`.

### Thay đổi NLP (đều đo trên train + validation, không làm giảm phép đo nào)

| cờ trong `src/nlp2.py` | nội dung | tác động ở test |
|---|---|---|
| `USE_PLACES_DESC` | ~280 cách gọi gián tiếp trong `nlp_knowledge.PLACES_DESC` ("nơi mượn sách", "bác trông xe") | cùng dòng dưới: +5 đáp án |
| `PRESENT_UNK_TH = 0.9` | tên LẠ có loại đoán rất chắc mà loại đó không có trên bản đồ → gây nhiễu, không ép sang loại khác | (bản v20, 0,9806) |
| `KW_ACC_CHECK`, `KW_WHOLE_WORD`, `KEYWORDS_MORE` | từ khóa địa điểm: phân biệt chữ trùng khi bỏ dấu ("giám đốc"/"đọc", "an ninh"/"ăn"), chỉ khớp trọn từ, thêm từ khóa | đổi 2 đáp án (v21) |
| `E5_FRAGILE_MIN = 0.9` | mô hình nghĩa phải chắc ≥ 0,9 mới tự gắn "dễ vỡ" khi không có từ khóa | đổi 7 đáp án robot 7 (v22, chưa nộp) |
| `DUP_NOREF = 1.0` (tắt) | đã thử ưu tiên loại có 1 bản khi không kèm hướng: giảm điểm, bỏ | — |

### Chạy lại trên máy có đủ mô hình CNN

Mô hình đọc câu phải học lại vì code và ví dụ đã đổi:

```powershell
git pull origin using_cnn
Remove-Item cache/nlp2_*.pkl, outputs/nlp2_final.pkl -ErrorAction SilentlyContinue
python src/pipeline.py test final
python src/check_submission.py final
```

Lưu ý: `outputs/nlp2_final.pkl` trong repo vẫn là bản cũ (học trước các thay đổi này), nên phải xóa để học lại.
Ba mô hình `edge_cnn.onnx`, `weather_cnn.onnx`, `swatch_cnn.onnx(.data)` chưa có trên GitHub (thư mục `outputs/` bị
`.gitignore`); muốn đưa lên: `git add -f outputs/models_final/*.onnx* outputs/models_dev/*.onnx*`.

### Cách các file v20–v22 được tạo trên máy không có CNN

`scratch/d3_delta.py`: lấy file 0,9778 làm nền, chỉ thay đáp án ở cảnh mà thay đổi NLP làm đổi kết quả đọc câu, và chỉ khi
bản đồ MLP cũ cho ra đúng 10 đáp án của file nền ở cảnh đó. Chạy `pipeline.py test final` trên máy có CNN sẽ cho kết quả
chuẩn hơn (có thể lệch vài cảnh so với v20–v22).

### Đã kiểm tra và không thấy vấn đề (bộ đếm tổng hợp ở test, `scratch/d1`–`d6`)

Sót hướng bắc / nam, sót gần / xa, sót điểm ghé, câu hai vế trộn về "dễ vỡ", cảnh bị điền mặc định toàn 0.
