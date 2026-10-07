# MS-TCN trên tập test

Infer lại `checkpoints/MS-TCN/j68_MS-TCN_best.pth` và `checkpoints/MS-TCN/j76_MS-TCN_best.pth` ngày 7 Oct 2026.

- Label: `labels/j76_CTR-GCN.json`, 422 gloss. Thứ tự lớp trong cả hai checkpoint trùng file này.
- Test: 5,163 clip hợp lệ. 50 thư mục test nằm ngoài danh sách label nên không được chấm.
- Eval mode, không augment. Loss là cross-entropy, label smoothing 0.1.
- j68 bỏ 8 điểm chân (68 joint). j76 giữ chân (76 joint).
- Tập test này cũng là tập dùng để chọn checkpoint lúc train.

## Tổng

| Model | Epoch | Params | Loss | Accuracy | Top-5 | Macro F1 | Đúng / 5,163 |
|---|---:|---:|---:|---:|---:|---:|---:|
| j68 | 67 | 690,742 | 1.311 | 92.66% | 98.22% | 93.07% | 4,784 |
| j76 | 59 | 707,254 | 1.306 | 92.70% | 98.39% | 93.20% | 4,786 |

Số lưu trong checkpoint lúc chọn best: j68 acc 92.70% / top-5 98.22% / macro F1 93.11%; j76 acc 92.68% / top-5 98.39% / macro F1 93.18%. Lần chạy này lệch tối đa 0.05 điểm.

Hai model đồng ý top-1 trên 95.6% clip. 4,702 clip cả hai đúng, 82 clip chỉ j68 đúng, 84 clip chỉ j76 đúng, 295 clip cả hai sai.

## Phân bố F1

Support trung bình 12.2 clip/gloss. j68 có 154 gloss F1 = 1, j76 có 148.

| Khoảng F1 | j68 | j76 |
|---|---:|---:|
| < 0.50 | 1 | 1 |
| 0.50–0.80 | 29 | 28 |
| 0.80–0.90 | 67 | 61 |
| 0.90–0.95 | 74 | 77 |
| ≥ 0.95 | 251 | 255 |

## Gloss yếu nhất

min(F1 j68, F1 j76) thấp nhất.

| Gloss | Support | F1 j68 | F1 j76 | Recall j68 | Recall j76 |
|---|---:|---:|---:|---:|---:|
| Dơ | 15 | 0.333 | 0.462 | 0.267 | 0.400 |
| Thứ tư | 13 | 0.560 | 0.500 | 0.538 | 0.538 |
| Thứ ba | 13 | 0.519 | 0.538 | 0.538 | 0.538 |
| Xôi | 15 | 0.615 | 0.552 | 0.533 | 0.533 |
| Chị | 16 | 0.552 | 0.600 | 0.500 | 0.562 |
| Con heo | 15 | 0.579 | 0.611 | 0.733 | 0.733 |
| Nhạt | 14 | 0.645 | 0.600 | 0.714 | 0.643 |
| Chú ý | 14 | 0.621 | 0.710 | 0.643 | 0.786 |
| Khóc | 17 | 0.621 | 0.667 | 0.529 | 0.588 |
| Tháng một | 12 | 0.621 | 0.692 | 0.750 | 0.750 |
| Cô | 15 | 0.625 | 0.649 | 0.667 | 0.800 |
| Tháng chín | 12 | 0.636 | 0.667 | 0.583 | 0.667 |

## Chỗ hai model tách nhau

Δ = F1 j76 − F1 j68. Sáu gloss mỗi phía có |Δ| lớn nhất.

j68 cao hơn:

| Gloss | n | F1 j68 | F1 j76 | Δ |
|---|---:|---:|---:|---:|
| Thứ năm | 12 | 0.800 | 0.636 | −0.164 |
| Bánh tét | 14 | 0.828 | 0.667 | −0.161 |
| Nghề nghiệp | 16 | 0.909 | 0.774 | −0.135 |
| Phía sau | 7 | 1.000 | 0.875 | −0.125 |
| Cà phê | 11 | 1.000 | 0.909 | −0.091 |
| Hứa | 11 | 1.000 | 0.909 | −0.091 |

j76 cao hơn:

| Gloss | n | F1 j68 | F1 j76 | Δ |
|---|---:|---:|---:|---:|
| Con vịt | 11 | 0.800 | 0.952 | +0.152 |
| Tháng tư | 12 | 0.762 | 0.909 | +0.147 |
| Giày | 12 | 0.818 | 0.957 | +0.138 |
| Dơ | 15 | 0.333 | 0.462 | +0.128 |
| Cái nồi | 14 | 0.815 | 0.929 | +0.114 |
| Tháng ba | 12 | 0.769 | 0.880 | +0.111 |

## Cặp hay nhầm với nhau

Chỉ giữ cặp mà gloss đúng bị đoán thành một gloss khác từ 2 lần trở lên. n/N là số lần nhầm trên support của gloss đúng.

### Cả hai model cùng đoán sai một kiểu

| Đúng | Đoán thành | n/N | Tỷ lệ |
|---|---|---:|---:|
| Dơ | Con heo | 8/15 | 53% |
| Chị | Cô | 7/16 | 44% |
| Xôi | Ướt | 5/15 | 33% |
| Không cần | Đắng | 4/12 | 33% |
| Nhẹ | Bây giờ | 4/17 | 24% |
| Mùa mưa | Mưa | 3/12 | 25% |
| Bà nội | Ông nội | 3/13 | 23% |
| Bây giờ | Nhẹ | 3/15 | 20% |
| Con heo | Dơ | 3/15 | 20% |
| Cô | Chị | 3/15 | 20% |
| Thú vị | Rửa tay | 3/15 | 20% |
| Thơm | Nhạt | 3/16 | 19% |

Nhầm hai chiều: Dơ ↔ Con heo, Chị ↔ Cô, Bây giờ ↔ Nhẹ.

### j68

| Đúng | Đoán thành | n/N |
|---|---|---:|
| Dơ | Con heo | 10/15 |
| Chị | Cô | 7/16 |
| Bây giờ | Nhẹ | 5/15 |
| Cô | Chị | 5/15 |
| Hát | Nhầm lẫn | 5/15 |
| Xôi | Ướt | 5/15 |
| Không cần | Đắng | 4/12 |
| Tháng chín | Tháng một | 4/12 |
| Thứ tư | Thứ ba | 4/13 |
| Con heo | Dơ | 4/15 |

### j76

| Đúng | Đoán thành | n/N |
|---|---|---:|
| Dơ | Con heo | 9/15 |
| Chị | Cô | 7/16 |
| Bánh tét | Bánh chưng | 6/14 |
| Xôi | Ướt | 6/15 |
| Thứ năm | Thứ tư | 5/12 |
| Nhẹ | Bây giờ | 5/17 |
| Không cần | Đắng | 4/12 |
| Con heo | Dơ | 4/15 |
| Nghề nghiệp | Làm việc | 4/16 |
| Thơm | Nhạt | 4/16 |
