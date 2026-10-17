# Ensemble MS-TCN + ST-GCN, j76

Trung bình softmax của `checkpoints/MS-TCN/j76_MS-TCN_best.pth` và `checkpoints/ST-GCN/j76_ST-GCN_best.pth`, ngày 17 Oct 2026. Hai nhánh nặng bằng nhau.

- Label: `labels/j76_CTR-GCN.json`, 422 gloss. Thứ tự lớp của cả hai checkpoint trùng file này.
- Test: 5,163 clip hợp lệ. 50 thư mục test nằm ngoài danh sách label nên không được chấm.
- Eval mode, không augment. Loss là cross-entropy, label smoothing 0.1. Với ensemble, loss tính trên xác suất đã trung bình, cùng hệ số smoothing.
- Cả hai nhánh giữ chân (76 joint), 64 frame.
- Tập test này cũng là tập dùng để chọn từng checkpoint lúc train.

## Tổng

| Model | Epoch | Params | Loss | Accuracy | Top-5 | Macro F1 | Đúng / 5,163 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MS-TCN | 59 | 707,254 | 1.306 | 92.70% | 98.39% | 93.20% | 4,786 |
| ST-GCN | 59 | 3,296,662 | 1.219 | 94.29% | 98.61% | 94.72% | 4,868 |
| Ensemble | — | 4,003,916 | 1.205 | 94.36% | 98.68% | 94.73% | 4,872 |

Ensemble hơn ST-GCN 4 clip và hơn MS-TCN 86 clip. Macro F1 gần như dính ST-GCN. Loss và top-5 nhích rõ hơn accuracy.

Hai nhánh đồng ý top-1 trên 94.6% clip (4,885). 4,712 clip cả hai đúng, 74 clip chỉ MS-TCN đúng, 156 clip chỉ ST-GCN đúng, 221 clip cả hai sai.

Trên 278 clip hai nhánh khác top-1, ensemble đúng 160 clip và sai 118. 158/160 là chọn đúng nhánh đã đúng sẵn. 2 clip còn lại cả hai nhánh đều sai nhưng sai khác gloss, trung bình rơi vào lớp đúng. 173 clip cả hai cùng đoán sai một gloss thì ensemble giữ nguyên lỗi đó.

Khi hai nhánh khác nhau, ensemble theo ST-GCN 151 clip, theo MS-TCN 122 clip, và chọn một gloss thứ ba 5 clip.

## Phân bố F1

Support trung bình 12.2 clip/gloss. MS-TCN có 148 gloss F1 = 1, ST-GCN có 190, ensemble có 194.

| Khoảng F1 | MS-TCN | ST-GCN | Ensemble |
|---|---:|---:|---:|
| < 0.50 | 1 | 1 | 1 |
| 0.50–0.80 | 28 | 21 | 20 |
| 0.80–0.90 | 61 | 43 | 50 |
| 0.90–0.95 | 77 | 68 | 59 |
| ≥ 0.95 | 255 | 289 | 292 |

## Gloss yếu nhất

F1 ensemble thấp nhất.

| Gloss | Support | F1 TCN | F1 GCN | F1 ensemble | Recall ensemble |
|---|---:|---:|---:|---:|---:|
| Dơ | 15 | 0.462 | 0.370 | 0.480 | 0.400 |
| Thứ tư | 13 | 0.500 | 0.571 | 0.500 | 0.538 |
| Thứ ba | 13 | 0.538 | 0.583 | 0.538 | 0.538 |
| Thứ năm | 12 | 0.636 | 0.667 | 0.609 | 0.583 |
| Con heo | 15 | 0.611 | 0.500 | 0.611 | 0.733 |
| Cô | 15 | 0.649 | 0.743 | 0.686 | 0.800 |
| Xôi | 15 | 0.552 | 0.690 | 0.690 | 0.667 |
| Chị | 16 | 0.600 | 0.714 | 0.690 | 0.625 |
| Khóc | 17 | 0.667 | 0.645 | 0.710 | 0.647 |
| Nhẹ | 17 | 0.710 | 0.750 | 0.710 | 0.647 |
| Hát | 15 | 0.769 | 0.714 | 0.714 | 0.667 |
| Tháng một | 12 | 0.692 | 0.769 | 0.720 | 0.750 |

## Chỗ ensemble tách khỏi hai nhánh

Sáu gloss mỗi phía. Δ ở bảng trên là F1 ensemble trừ F1 cao hơn trong hai nhánh. Δ ở bảng dưới là F1 ensemble trừ F1 thấp hơn trong hai nhánh.

Cao hơn cả hai:

| Gloss | n | F1 TCN | F1 GCN | F1 ensemble | Δ |
|---|---:|---:|---:|---:|---:|
| Đồ dùng | 13 | 0.815 | 0.786 | 0.889 | +0.074 |
| Anh | 17 | 0.839 | 0.867 | 0.938 | +0.071 |
| Nhạt | 14 | 0.600 | 0.750 | 0.815 | +0.065 |
| Dễ | 21 | 0.872 | 0.900 | 0.950 | +0.050 |
| Không quen | 12 | 0.957 | 0.957 | 1.000 | +0.043 |
| Quả dứa | 12 | 0.957 | 0.917 | 1.000 | +0.043 |

Thấp hơn cả hai:

| Gloss | n | F1 TCN | F1 GCN | F1 ensemble | Δ |
|---|---:|---:|---:|---:|---:|
| Sử dụng | 6 | 0.833 | 0.923 | 0.769 | −0.064 |
| Không nên | 13 | 0.870 | 0.846 | 0.783 | −0.064 |
| Giám đốc | 11 | 0.952 | 0.957 | 0.909 | −0.043 |
| Ca sĩ | 13 | 0.880 | 0.880 | 0.846 | −0.034 |
| Thứ hai | 13 | 0.815 | 0.815 | 0.786 | −0.029 |
| Thứ năm | 12 | 0.636 | 0.667 | 0.609 | −0.028 |

## Cặp hay nhầm

Ensemble. Chỉ giữ cặp mà gloss đúng bị đoán thành một gloss khác từ 2 lần trở lên. n/N là số lần nhầm trên support của gloss đúng.

| Đúng | Đoán thành | n/N | Tỷ lệ |
|---|---|---:|---:|
| Dơ | Con heo | 9/15 | 60% |
| Chị | Cô | 6/16 | 38% |
| Thứ năm | Thứ tư | 5/12 | 42% |
| Nhẹ | Bây giờ | 5/17 | 29% |
| Không cần | Đắng | 4/12 | 33% |
| Con heo | Dơ | 4/15 | 27% |
| Xôi | Ướt | 4/15 | 27% |
| Nghề nghiệp | Làm việc | 4/16 | 25% |
| Mùa mưa | Mưa | 3/12 | 25% |
| Bà nội | Ông nội | 3/13 | 23% |

Nhầm hai chiều: Dơ ↔ Con heo.
