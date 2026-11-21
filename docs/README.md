# Tài liệu dự án

| Thư mục | Nội dung | Tạo lại bằng |
|---|---|---|
| [trinh-bay-y-tuong/](trinh-bay-y-tuong/) | bản trình bày ý tưởng (`.md` và `.docx`) | `python docs/trinh-bay-y-tuong/xuat_docx.py` |
| [bao-cao-ket-qua/](bao-cao-ket-qua/) | **BÁO CÁO KẾT QUẢ DỰ ÁN STEM** (`bao-cao-ket-qua.md` → `.docx`), hình trong `bieu-do/` | các bước dưới đây |
| [slides/](slides/) | slide trình chiếu HTML, có nút chạy demo trực tiếp | `python docs/slides/serve.py` |

Mọi lệnh đều chạy từ thư mục gốc repo, bằng `.venv/bin/python`, sau khi đã cài `requirements-docs.txt`.

## Tái lập báo cáo từng bước

`bash reproduce.sh` chạy cả chuỗi dưới đây. Mỗi bước đọc đầu ra của bước trước nên có thể chạy riêng từng bước. Dữ liệu lấy từ Kaggle qua [`assets.py`](../assets.py): dataset checkpoint được ghim ở phiên bản 2, nên các lần chạy cho cùng số liệu.

### 1. Tải dữ liệu

```bash
python assets.py checkpoints keypoints   # assets/ (670 MB) + vsl400-keypoint/ (1,2 GB)
python -m preprocess.download_signconnect  # video QIPEDC, chỉ cần cho hình khác miền
```

### 2. Chấm lại 17 checkpoint (khoảng 10 phút trên Apple M-series, dùng MPS nếu có)

```bash
OUT=reports/ket-qua-2026-11-21
python $OUT/run_eval.py . . $OUT    # $OUT/<thử nghiệm>.npz (logits vsl / oov / qip), meta.json
python $OUT/analyze.py $OUT         # report.json
python $OUT/derive.py $OUT          # derived.json
python $OUT/de_xuat.py $OUT .       # de-xuat.json
```

| Tệp | Dùng cho phần nào của báo cáo |
|---|---|
| `report.json` | Top-1/Top-5, macro-F1 kèm khoảng tin cậy 95% (bootstrap 1.000 lần, chỉ tính các lớp có trong mẫu), ECE, tỷ lệ chấp nhận nhầm từ chưa học, kết quả QIPEDC, tổ hợp mô hình, các cặp hay nhầm |
| `derived.json` | đường cong độ phủ và độ chính xác theo ngưỡng τ (chọn τ = 0,7 cho demo) |
| `de-xuat.json` | mô hình đề xuất so với mô hình gốc: cặp khó (McNemar), khả năng sửa lỗi khi tổ hợp, tốc độ ký, tay gần mặt |
| `curves.json` | đường học theo epoch, lấy từ log của các phiên bản notebook (`parse_logs.py`) |
| `vi-du-du-doan.json` | các clip minh hoạ: top-5 của CTR-GCN, F1 của từng từ |

Kiểm tra nhanh: nếu chạy lại `infer_signconnect.py --checkpoints assets/checkpoints/j76_CTR-GCN_best.pth`, kết quả phải là `acc=0.5992 macro_f1=0.6404`, đúng như trong `report.json`.

`curves.json` đọc log huấn luyện của từng phiên bản notebook. Log này chỉ tải được bằng tài khoản sở hữu notebook, nên tệp được giữ sẵn trong repo.

### 3. Vẽ hình

```bash
python docs/bao-cao-ket-qua/ve_bieu_do.py        # biểu đồ kết quả (cột, đường, nhiệt, phân tán, ...)
python docs/bao-cao-ket-qua/ve_de_xuat.py        # biểu đồ mô hình đề xuất
python docs/bao-cao-ket-qua/ve_gcn.py            # sơ đồ cách GCN hoạt động
python docs/bao-cao-ket-qua/ve_khung_hinh.py     # khung hình từ video kèm khung xương, các cặp dễ nhầm
python docs/bao-cao-ket-qua/ve_khung_de_xuat.py  # clip mô hình đề xuất đoán đúng
```

`ve_khung_hinh.py` và `ve_khung_de_xuat.py` tải riêng từng clip VSL400 cần dùng vào `data/vsl400-videos/` và làm mờ mặt trước khi lưu hình. Nếu đã có sẵn video, truyền thư mục đó làm tham số đầu tiên.

### 4. Xuất `.docx`

```bash
python docs/bao-cao-ket-qua/xuat_docx.py      # bao-cao-ket-qua.md -> bao-cao-ket-qua.docx (A4, Times New Roman 14)
```

Mục lục đã có sẵn tên các đề mục. Để điền số trang, mở file trong Word, chọn **Yes** khi được hỏi "update fields", hoặc bấm vào mục lục rồi nhấn **F9** → *Update page numbers only*.

Trên macOS có Pages, cũng có thể ghi số trang thẳng vào file: chạy `python docs/bao-cao-ket-qua/tinh_so_trang.py` rồi chạy lại `xuat_docx.py`. Cách này cần cấp quyền *Automation → Pages* cho terminal.

## Slide

```bash
python docs/slides/serve.py           # http://127.0.0.1:8765/slides/
python docs/slides/serve.py --no-browser
```

| Phím | Tác dụng |
|---|---|
| → / Space / PageDown / Enter | slide sau |
| ← / PageUp | slide trước |
| F | toàn màn hình |

- Slide dùng thẳng hình trong `docs/bao-cao-ket-qua/bieu-do/`, nên vẽ lại hình (bước 3) thì slide cũng cập nhật theo.
- Ở slide Demo, nút **Webcam** chạy `demo.py`, còn nút **Video mẫu** chạy `demo.py --video` trên 8 clip. Clip và checkpoint tự tải nếu chưa có.
- Không có server (mở file `index.html` trực tiếp) thì slide phát bản ghi `slides/assets/demo.mp4`.

## Dataset checkpoint trên Kaggle

[`huynguang/vsl-sign-checkpoints`](https://www.kaggle.com/datasets/huynguang/vsl-sign-checkpoints), public, phiên bản 2:

```
manifest.json                    thử nghiệm -> mô hình, is_leg, T, epoch, macro-F1, notebook + version nguồn
files.json                       danh sách tệp và kích thước (assets.py dùng để tải từng tệp)
checkpoints/<exp>_best.pth       trọng số tốt nhất theo macro-F1 (state_dict, class_names, model_name, is_leg, max_frames)
checkpoints/<exp>_last.pth       trạng thái huấn luyện để --resume (optimizer, scheduler, scaler, EMA)
labels/<exp>.json                thứ tự lớp; signconnect_j76_common.json = 147 từ chung với QIPEDC
pose_state_mlp_best.pth          mô hình nhận biết nghỉ/đang ký cho demo
signconnect-keypoints/*.npy      keypoint của 252 clip QIPEDC
```

Mỗi thử nghiệm lấy checkpoint từ phiên bản notebook mới nhất còn lưu nó. Các dataset `*/checkpoints` riêng của từng tài khoản chỉ là đầu vào để huấn luyện tiếp, không phải bản mới nhất.

Muốn cập nhật dataset (cần token của tài khoản `huynguang`):

```bash
# thư mục gồm các tệp trên và dataset-metadata.json có "id": "huynguang/vsl-sign-checkpoints"
kaggle datasets version -p <thư-mục> -m "mô tả" --dir-mode zip
```

Sau đó tăng `CHECKPOINT_VERSION` trong `assets.py`.
