# Nhận diện ngôn ngữ ký hiệu tiếng Việt từ khung xương

Dự án STEM: nhận diện 422 từ ký hiệu tiếng Việt (VSL400) từ webcam. MediaPipe lấy 76 điểm khung xương
mỗi khung hình, sau đó mạng nơ-ron đồ thị phân loại cả chuỗi. Mô hình mặc định là **CTR-GCN J76**:

| | Top-1 | Macro-F1 | Ghi chú |
|---|---|---|---|
| VSL400 test, 5.163 clip | 94,3% | 94,7% | Top-5 98,5%, khoảng 31 ms mỗi ký hiệu trên CPU |
| QIPEDC, 252 clip, người ký và cách quay khác | 59,9% | 64,0% | phép thử khác miền |

Mọi checkpoint và dữ liệu đều nằm trên Kaggle ở dạng **public**. Code tự tải về khi cần, không cần tài khoản hay token.

| Dataset Kaggle | Nội dung | Dung lượng |
|---|---|---|
| [`huynguang/vsl-sign-checkpoints`](https://www.kaggle.com/datasets/huynguang/vsl-sign-checkpoints) | checkpoint mới nhất của 17 thử nghiệm (`_best` và `_last`), nhãn, mô hình nghỉ/ký, keypoint QIPEDC, `manifest.json` | 670 MB |
| [`huynguang/vsl400-keypoints-final`](https://www.kaggle.com/datasets/huynguang/vsl400-keypoints-final) | keypoint VSL400 `vsl400-keypoint/{train,test}/<từ>/<id>.npy`, mỗi tệp `(T, 76, 3)` | 1,2 GB |
| [`nguyenanfms/vsl-vietnamese-sign-language-v2`](https://www.kaggle.com/datasets/nguyenanfms/vsl-vietnamese-sign-language-v2) | video VSL400 gốc 224×224, chỉ tải vài clip cho demo và hình minh hoạ | – |

## 1. Cài đặt

Cần Python 3.10 trở lên. Webcam là tuỳ chọn, vì demo chạy được trên video mẫu.

```bash
python -m venv .venv            # hoặc: uv venv
.venv/bin/pip install -r requirements.txt          # đủ cho demo
.venv/bin/pip install -r requirements-docs.txt     # thêm phần vẽ biểu đồ và xuất báo cáo
```

Mô hình MediaPipe (`mediapipe_models/*.task`) tự tải ở lần chạy đầu.

## 2. Demo

```bash
.venv/bin/python demo.py                              # webcam; lần đầu tải CTR-GCN J76 + mô hình nghỉ/ký (8 MB)
.venv/bin/python assets.py videos                     # 8 video mẫu vào demo-video/
.venv/bin/python demo.py --video demo-video           # chạy trên video mẫu, không cần webcam
.venv/bin/python demo.py --model j76_AAGCN            # đổi mô hình: tên bất kỳ trong bảng ở mục 5
.venv/bin/python demo.py --video demo-video --record out.mp4 --no-window   # ghi lại ra tệp
```

Cách dùng: đứng cách camera khoảng 1–1,5 m, thấy được cả vai và hai tay. Hạ tay xuống là kết thúc một ký hiệu.

| Độ tin cậy | Demo làm gì |
|---|---|
| ≥ 0,7 | thêm từ vào câu |
| 0,4 – 0,7 | hiện 5 gợi ý, bấm **1–5** để chọn hoặc **R** để ký lại |
| < 0,4 | báo "không nhận ra", thường là từ chưa học |
| quá nửa số khung mất tay hoặc vai | yêu cầu ký lại |

Phím: **1–5** chọn gợi ý, **R** ký lại, **Backspace** xoá từ cuối, **C** xoá câu, **Q** hoặc **Esc** để thoát.

Bản rút gọn chỉ có vòng lặp webcam: `.venv/bin/python infer_camera.py [--model j68_ST-GCN]`.

## 3. Trình chiếu slide

```bash
.venv/bin/python docs/slides/serve.py         # mở http://127.0.0.1:8765/slides/
```

- **→ / Space / PageDown** sang slide sau, **←** lùi, **F** toàn màn hình.
- Ở slide Demo, nút **Bắt đầu demo** chạy `demo.py` ngay trên máy đang trình chiếu, bằng webcam hoặc video mẫu (video tự tải nếu chưa có). Server chỉ nghe ở `127.0.0.1`.
- Mở thẳng `docs/slides/index.html` trong trình duyệt cũng được, khi đó nút demo phát bản ghi `docs/slides/assets/demo.mp4`.

## 4. Tái lập kết quả

Bắt đầu từ bước nào cũng được, vì mỗi bước tự tải đúng dữ liệu nó cần.

```bash
bash reproduce.sh              # mọi thứ: chấm lại 17 checkpoint → số liệu → biểu đồ → bao-cao-ket-qua.docx
bash reproduce.sh --figures    # chỉ vẽ lại hình và xuất .docx từ số liệu đã có trong reports/
```

Chi tiết từng bước nằm trong [docs/README.md](docs/README.md):

1. **Chấm lại checkpoint.** `reports/ket-qua-2026-11-21/run_eval.py` tính logits của mọi mô hình trên VSL400 test (từ đã học và từ chưa học) và QIPEDC. Sau đó `analyze.py` tính Top-1, macro-F1 kèm khoảng tin cậy bootstrap, ECE và tỷ lệ chấp nhận nhầm từ chưa học; `derive.py` vẽ đường cong theo ngưỡng τ; `de_xuat.py` so sánh các mô hình đề xuất với mô hình gốc.
2. **Huấn luyện lại.** Chạy `training/runs/NN_*.sh`, mỗi script một thử nghiệm. Các script này giữ nguyên siêu tham số đã dùng trên Kaggle.
3. **Báo cáo.** Chạy các script `docs/bao-cao-ket-qua/ve_*.py` để vẽ hình, rồi `xuat_docx.py` để xuất `.docx`.

### Huấn luyện lại

Trên Kaggle, notebook cần 2× T4 và thêm hai dataset public `huynguang/vsl400-keypoints-final` và `huynguang/vsl-sign-checkpoints` qua **Add Input**:

```bash
bash training/runs/04_ctr_gcn_j76.sh             # huấn luyện từ đầu
RESUME=1 bash training/runs/04_ctr_gcn_j76.sh    # tiếp tục từ j76_CTR-GCN_last.pth trên Kaggle
```

`training/run-all.ipynb` chạy lần lượt mọi script trong `training/runs/`.

Trên máy riêng, ví dụ với một GPU:

```bash
.venv/bin/python assets.py keypoints
.venv/bin/python -m training.run --root vsl400-keypoint --model CTR-GCN --tag j76 --legs \
    --frames 64 --epochs 70 --micro-batch 32 --accumulate 2 --min-train-samples 20 --select-by macro_f1
# hoặc dùng nguyên script (cần 2 GPU): DATA_ROOT=vsl400-keypoint CKPT_ROOT=assets/checkpoints bash training/runs/04_ctr_gcn_j76.sh
```

Checkpoint mới được ghi vào `checkpoints/{tag}_{model}_{best,last}.pth`. Lưu ý: tập test cũng là tập chọn epoch, nên số liệu VSL400 là cận trên. Phép thử QIPEDC thì hoàn toàn độc lập.

## 5. Các thử nghiệm

`python assets.py` tải toàn bộ checkpoint và in bảng này. Thông tin gốc nằm trong `assets/manifest.json`. `j68` là 68 khớp (bỏ chân), `j76` là đủ 76 khớp, `j76rest` là có thêm lớp "nghỉ". Mỗi checkpoint là phiên bản notebook mới nhất còn giữ nó; các lần chạy được nối tiếp qua nhiều tài khoản Kaggle.

| Thử nghiệm | Mô hình | Khớp | T | Epoch tốt nhất | Macro-F1 VSL400 | Top-1 QIPEDC | `_last.pth` | Notebook Kaggle |
|---|---|---|---|---|---|---|---|---|
| `j68_MS-TCN` | MS-TCN | 68 | 64 | 67 | 93,11 | 59,1 | – | `huynguang/notebook5c8cabe9aa` v1 |
| `j76_MS-TCN` | MS-TCN | 76 | 64 | 59 | 93,18 | 54,4 | – | `huynguang/notebook5c8cabe9aa` v1 |
| `j68_ST-GCN` | ST-GCN | 68 | 64 | 59 | 94,49 | 60,7 | – | `nqhuy23/notebookd52c51678e` v1 |
| `j76_ST-GCN` | ST-GCN | 76 | 64 | 59 | 94,76 | 59,5 | – | `nqhuy23/notebookd52c51678e` v3 |
| `j68_CTR-GCN`¹ | CTR-GCN | 68 | 64 | 19 | 91,91 | 56,3 | – | `huynguang/notebook5c8cabe9aa` v2 |
| **`j76_CTR-GCN`** | CTR-GCN | 76 | 64 | 62 | 94,69 | 59,9 | ep 64 | `dinhnhatky2003/notebook618354fc0d` v10 |
| `j68_AAGCN` | AAGCN | 68 | 64 | 32 | 93,64 | 61,1 | – | `dinhnhatky2003/notebook618354fc0d` v2 |
| `j76_AAGCN` | AAGCN | 76 | 64 | 40 | 93,49 | 60,7 | ep 70 | `nqhuy23/notebookd52c51678e` v11, v12 |
| `j76_Transformer` | Transformer | 76 | 64 | 42 | 91,41 | 55,6 | ep 70 | `dinhnhatky2003/notebook618354fc0d` v5 |
| `j68_MultiRate-Attn-STGCN` | MultiRate-Attn-STGCN | 68 | 96 | 67 | 94,13 | 56,0 | ep 70 | `dinhnhatky2003/notebook618354fc0d` v8 |
| `j76_MultiRate-Attn-STGCN` | MultiRate-Attn-STGCN | 76 | 96 | 67 | 93,80 | 57,1 | ep 70 | `lngmnht/notebook77d9eb4a1e` v8 |
| `j76_CTR-GCN-BT` | CTR-GCN-BT | 76 | 64 | 42 | 94,19 | 58,3 | ep 47 | `dinhnhatky2003/notebook618354fc0d` v17 |
| `j76_AnchorGCN-A` | AnchorGCN-A | 76 | 64 | 48 | 94,22 | 57,9 | ep 48 | `lngmnht/notebook77d9eb4a1e` v14 |
| `j76_AnchorGCN`¹ | AnchorGCN | 76 | 64 | 14 | 87,30 | 52,8 | ep 14 | `dinhnhatky2003/notebook618354fc0d` v12 |
| `j76rest_AnchorGCN`¹ | AnchorGCN | 76 | 64 | 14 | 87,00 | 49,2 | ep 14 | `abchuy/notebook5b648360c4` v1 |
| `j76rest_AnchorGCN-AS` | AnchorGCN-AS | 76 | 64 | 38 | 93,00 | 53,2 | ep 42 | `nqhuy23/notebookd52c51678e` v19 |
| `j76rest_AnchorGCN-T32` | AnchorGCN-T32 | 76 | 64 | 39 | 92,84 | 59,9 | ep 39 | `huynguang/notebook5c8cabe9aa` v20 |

¹ Lần chạy dừng sớm vì hết giờ GPU. Checkpoint vẫn được giữ, nhưng báo cáo không dùng để so sánh.

Dùng trong code:

```python
import assets
assets.checkpoint("j76_AAGCN")           # assets/checkpoints/j76_AAGCN_best.pth, chỉ tải tệp này
assets.checkpoint("j76_AAGCN", "last")   # trạng thái huấn luyện để --resume
assets.labels("j76_CTR-GCN")             # thứ tự 422 lớp
assets.manifest()["j76_CTR-GCN"]         # epoch, macro-F1, notebook nguồn
```

## 6. Cấu trúc thư mục

```
assets.py               tải dữ liệu public từ Kaggle (checkpoint, keypoint, video mẫu)
demo.py                 demo đầy đủ: hàng đợi ký hiệu, kiểm soát chất lượng, ngưỡng từ chối, gợi ý top-5
infer_camera.py         vòng lặp webcam tối giản
infer_signconnect.py    chấm checkpoint trên keypoint QIPEDC (phép thử khác miền)
extract_signconnect.py  trích keypoint từ video QIPEDC
preprocess/             MediaPipe, chuẩn hoá theo vai, chọn khung, dataset
models/                 MS-TCN, ST-GCN, CTR-GCN(-BT), AAGCN, Transformer, MultiRate, AnchorGCN (xem models/README.md)
training/               vòng huấn luyện (DDP/AMP/EMA), training/runs/*.sh cho từng thử nghiệm
reports/ket-qua-2026-11-21/  script chấm lại và số liệu của báo cáo (report.json, derived.json, ...)
docs/                   ý tưởng, báo cáo kết quả (.md và .docx), slide (xem docs/README.md)
```

Các thư mục tải về không nằm trong git: `assets/`, `vsl400-keypoint/`, `demo-video/`, `data/`, `mediapipe_models/`.

## Dữ liệu và quyền riêng tư

- Video VSL400 thuộc về nhóm tác giả bộ dữ liệu. Repo không chứa video; code chỉ tải từng clip cần cho demo và hình minh hoạ, và không phát tán lại.
- Mọi khuôn mặt trong hình của báo cáo và slide đều đã làm mờ.
- QIPEDC (SignConnect) lấy từ Hugging Face `Lakeserl/SignConnect` (MIT).
