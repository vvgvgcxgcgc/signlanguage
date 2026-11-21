#!/usr/bin/env bash
# Tái lập số liệu, biểu đồ và file báo cáo từ dữ liệu public trên Kaggle.
#
#   bash reproduce.sh            # chấm lại 17 checkpoint, tính số liệu, vẽ biểu đồ, xuất .docx
#   bash reproduce.sh --figures  # chỉ vẽ lại hình và xuất .docx từ số liệu đã có trong reports/
#
# Cần: .venv với requirements-docs.txt. Lần đầu tải khoảng 1,9 GB (checkpoint + keypoint).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-.venv/bin/python}"
OUT=reports/ket-qua-2026-11-21

if [ "${1:-}" != "--figures" ]; then
  "$PY" assets.py checkpoints keypoints          # huynguang/vsl-sign-checkpoints + vsl400-keypoints-final
  "$PY" "$OUT/run_eval.py" . . "$OUT"             # logits từng mô hình -> $OUT/*.npz, meta.json
  "$PY" "$OUT/analyze.py" "$OUT"                  # -> report.json (Top-1, macro-F1 + CI, ECE, OOV, QIPEDC)
  "$PY" "$OUT/derive.py" "$OUT"                   # -> derived.json (đường cong theo ngưỡng τ)
  "$PY" "$OUT/de_xuat.py" "$OUT" .                # -> de-xuat.json (mô hình đề xuất: cặp khó, McNemar)
fi

# Video QIPEDC cho các hình khác miền (Hugging Face: Lakeserl/SignConnect, khoảng 26 MB).
[ -d data/signconnect/Videos ] || "$PY" -m preprocess.download_signconnect

"$PY" docs/bao-cao-ket-qua/ve_bieu_do.py          # biểu đồ kết quả
"$PY" docs/bao-cao-ket-qua/ve_gcn.py              # sơ đồ GCN
"$PY" docs/bao-cao-ket-qua/ve_de_xuat.py         # biểu đồ mô hình đề xuất
"$PY" docs/bao-cao-ket-qua/ve_khung_hinh.py       # khung hình từ video (tự tải từng clip từ Kaggle)
"$PY" docs/bao-cao-ket-qua/ve_khung_de_xuat.py
"$PY" docs/bao-cao-ket-qua/xuat_docx.py           # -> docs/bao-cao-ket-qua/bao-cao-ket-qua.docx
echo "Xong: docs/bao-cao-ket-qua/bao-cao-ket-qua.docx"
