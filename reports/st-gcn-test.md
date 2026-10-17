# ST-GCN trên tập test

Infer lại `checkpoints/ST-GCN/j68_ST-GCN_best.pth` và `checkpoints/ST-GCN/j76_ST-GCN_best.pth` ngày 17 Oct 2026.

- Label: `labels/j76_CTR-GCN.json`, 422 gloss. Thứ tự lớp trong cả hai checkpoint trùng file này.
- Test: 5,163 clip hợp lệ. 50 thư mục test nằm ngoài danh sách label nên không được chấm.
- Eval mode, không augment. Loss là cross-entropy, label smoothing 0.1.
- j68 bỏ 8 điểm chân (68 joint). j76 giữ chân (76 joint).
- Tập test này cũng là tập dùng để chọn checkpoint lúc train.

## Tổng

| Model | Epoch | Params | Loss | Accuracy | Top-5 | Macro F1 | Đúng / 5,163 |
|---|---:|---:|---:|---:|---:|---:|---:|
| j68 | 59 | 3,265,046 | 1.221 | 94.09% | 98.61% | 94.45% | 4,858 |
| j76 | 59 | 3,296,662 | 1.219 | 94.29% | 98.61% | 94.72% | 4,868 |

Số lưu trong checkpoint lúc chọn best: j68 acc 94.13% / top-5 98.63% / macro F1 94.49%; j76 acc 94.33% / top-5 98.61% / macro F1 94.76%. Lần chạy này lệch tối đa 0.04 điểm.

Hai model đồng ý top-1 trên 97.3% clip. 4,806 clip cả hai đúng, 52 clip chỉ j68 đúng, 62 clip chỉ j76 đúng, 243 clip cả hai sai.

## Phân bố F1

Support trung bình 12.2 clip/gloss. j68 có 181 gloss F1 = 1, j76 có 190.

| Khoảng F1 | j68 | j76 |
|---|---:|---:|
| < 0.50 | 1 | 1 |
| 0.50–0.80 | 20 | 21 |
| 0.80–0.90 | 59 | 43 |
| 0.90–0.95 | 64 | 68 |
| ≥ 0.95 | 278 | 289 |

## Gloss yếu nhất

min(F1 j68, F1 j76) thấp nhất.

| Gloss | Support | F1 j68 | F1 j76 | Recall j68 | Recall j76 |
|---|---:|---:|---:|---:|---:|
| Dơ | 15 | 0.385 | 0.370 | 0.333 | 0.333 |
| Con heo | 15 | 0.588 | 0.500 | 0.667 | 0.533 |
| Thứ tư | 13 | 0.615 | 0.571 | 0.615 | 0.615 |
| Thứ ba | 13 | 0.615 | 0.583 | 0.615 | 0.538 |
| Khóc | 17 | 0.667 | 0.645 | 0.588 | 0.588 |
| Nhầm lẫn | 11 | 0.667 | 0.696 | 0.636 | 0.727 |
| Thứ năm | 12 | 0.800 | 0.667 | 0.833 | 0.667 |
| Xôi | 15 | 0.714 | 0.690 | 0.667 | 0.667 |
| Cô | 15 | 0.706 | 0.743 | 0.800 | 0.867 |
| Hát | 15 | 0.786 | 0.714 | 0.733 | 0.667 |
| Thú vị | 15 | 0.714 | 0.741 | 0.667 | 0.667 |
| Chị | 16 | 0.714 | 0.714 | 0.625 | 0.625 |

## Chỗ hai model tách nhau

Δ = F1 j76 − F1 j68. Sáu gloss mỗi phía có |Δ| lớn nhất.

j68 cao hơn:

| Gloss | n | F1 j68 | F1 j76 | Δ |
|---|---:|---:|---:|---:|
| Thứ năm | 12 | 0.800 | 0.667 | −0.133 |
| Nếm | 12 | 0.880 | 0.750 | −0.130 |
| Con heo | 15 | 0.588 | 0.500 | −0.088 |
| Quả dứa | 12 | 1.000 | 0.917 | −0.083 |
| Ca sĩ | 13 | 0.960 | 0.880 | −0.080 |
| Gọi | 12 | 1.000 | 0.923 | −0.077 |

j76 cao hơn:

| Gloss | n | F1 j68 | F1 j76 | Δ |
|---|---:|---:|---:|---:|
| Quả đào | 12 | 0.857 | 1.000 | +0.143 |
| Chậm lại | 7 | 0.875 | 1.000 | +0.125 |
| Xuất viện | 5 | 0.889 | 1.000 | +0.111 |
| Nghỉ ngơi | 17 | 0.903 | 1.000 | +0.097 |
| Cay | 11 | 0.870 | 0.957 | +0.087 |
| Giám đốc | 11 | 0.870 | 0.957 | +0.087 |

## Cặp hay nhầm với nhau

Chỉ giữ cặp mà gloss đúng bị đoán thành một gloss khác từ 2 lần trở lên. n/N là số lần nhầm trên support của gloss đúng.

### Cả hai model cùng đoán sai một kiểu

| Đúng | Đoán thành | n/N | Tỷ lệ |
|---|---|---:|---:|
| Dơ | Con heo | 9/15 | 60% |
| Chị | Cô | 6/16 | 38% |
| Con heo | Dơ | 5/15 | 33% |
| Nhẹ | Bây giờ | 5/17 | 29% |
| Không cần | Đắng | 4/12 | 33% |
| Thú vị | Rửa tay | 4/15 | 27% |
| Xôi | Ướt | 4/15 | 27% |
| Bà nội | Ông nội | 3/13 | 23% |
| Hát | Nhầm lẫn | 3/15 | 20% |
| Khóc | Chú ý | 3/17 | 18% |
| Con vịt | Tìm | 2/11 | 18% |
| Thứ năm | Thứ tư | 2/12 | 17% |

Nhầm hai chiều: Dơ ↔ Con heo.

### j68

| Đúng | Đoán thành | n/N |
|---|---|---:|
| Dơ | Con heo | 9/15 |
| Chị | Cô | 6/16 |
| Nhẹ | Bây giờ | 6/17 |
| Con heo | Dơ | 5/15 |
| Không cần | Đắng | 4/12 |
| Thú vị | Rửa tay | 4/15 |
| Xôi | Ướt | 4/15 |
| Khóc | Chú ý | 4/17 |
| Quả đào | Nhà | 3/12 |
| Bà nội | Ông nội | 3/13 |

### j76

| Đúng | Đoán thành | n/N |
|---|---|---:|
| Dơ | Con heo | 9/15 |
| Con heo | Dơ | 7/15 |
| Chị | Cô | 6/16 |
| Nhẹ | Bây giờ | 5/17 |
| Không cần | Đắng | 4/12 |
| Thứ năm | Thứ tư | 4/12 |
| Hát | Nhầm lẫn | 4/15 |
| Thú vị | Rửa tay | 4/15 |
| Xôi | Ướt | 4/15 |
| Bà nội | Ông nội | 3/13 |
