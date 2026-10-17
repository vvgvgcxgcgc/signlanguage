# AAGCN j76 trên tập test

Infer `checkpoints/AAGCN/j76_AAGCN_best.pth` ngày 17 Oct 2026. Thư mục checkpoint chỉ có bản j76. ST-GCN j76 infer cùng lượt để so. Hàng MS-TCN lấy từ `reports/ms-tcn-test.md`, cùng tập test.

- Label: `labels/j76_CTR-GCN.json`, 422 gloss. Thứ tự lớp trong checkpoint trùng file này.
- Test: 5,163 clip hợp lệ. 50 thư mục test nằm ngoài danh sách label nên không được chấm.
- Eval mode, không augment. Loss là cross-entropy, label smoothing 0.1.
- j76 giữ chân (76 joint), 64 frame.
- Tập test này cũng là tập dùng để chọn checkpoint lúc train. Best được chọn theo macro F1, rơi vào epoch 40.

## Tổng

| Model | Epoch | Params | Loss | Accuracy | Top-5 | Macro F1 | Đúng / 5,163 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MS-TCN j76 | 59 | 707,254 | 1.306 | 92.70% | 98.39% | 93.20% | 4,786 |
| AAGCN j76 | 40 | 3,950,705 | 1.256 | 93.12% | 98.33% | 93.50% | 4,808 |
| ST-GCN j76 | 59 | 3,296,662 | 1.219 | 94.29% | 98.61% | 94.72% | 4,868 |

Số lưu trong checkpoint AAGCN lúc chọn best: acc 93.11% / top-5 98.35% / macro F1 93.49%. Lần chạy này lệch tối đa 0.02 điểm.

AAGCN hơn MS-TCN 22 clip và thấp hơn ST-GCN 60 clip. Macro F1 thấp hơn ST-GCN 1.22 điểm.

AAGCN và ST-GCN đồng ý top-1 trên 94.9% clip. 4,730 clip cả hai đúng, 78 clip chỉ AAGCN đúng, 138 clip chỉ ST-GCN đúng, 217 clip cả hai sai.

## Phân bố F1

Support trung bình 12.2 clip/gloss. AAGCN có 156 gloss F1 = 1, ST-GCN có 190.

| Khoảng F1 | AAGCN | ST-GCN |
|---|---:|---:|
| < 0.50 | 1 | 1 |
| 0.50–0.80 | 27 | 21 |
| 0.80–0.90 | 60 | 43 |
| 0.90–0.95 | 76 | 68 |
| ≥ 0.95 | 258 | 289 |

## Gloss yếu nhất

F1 AAGCN thấp nhất.

| Gloss | Support | F1 AAGCN | F1 ST-GCN | Recall AAGCN | Recall ST-GCN |
|---|---:|---:|---:|---:|---:|
| Thứ tư | 13 | 0.480 | 0.571 | 0.462 | 0.615 |
| Dơ | 15 | 0.500 | 0.370 | 0.400 | 0.333 |
| Thứ ba | 13 | 0.538 | 0.583 | 0.538 | 0.538 |
| Nếm | 12 | 0.621 | 0.750 | 0.750 | 0.750 |
| Bánh chưng | 14 | 0.643 | 0.846 | 0.643 | 0.786 |
| Bánh tét | 14 | 0.643 | 0.897 | 0.643 | 0.929 |
| Chị | 16 | 0.643 | 0.714 | 0.562 | 0.625 |
| Xôi | 15 | 0.643 | 0.690 | 0.600 | 0.667 |
| Con heo | 15 | 0.647 | 0.500 | 0.733 | 0.533 |
| Thú vị | 15 | 0.667 | 0.741 | 0.600 | 0.667 |
| Hát | 15 | 0.667 | 0.714 | 0.667 | 0.667 |
| Thứ năm | 12 | 0.667 | 0.667 | 0.583 | 0.667 |

## Chỗ AAGCN tách khỏi ST-GCN

Δ = F1 AAGCN − F1 ST-GCN. Sáu gloss mỗi phía có |Δ| lớn nhất.

AAGCN cao hơn:

| Gloss | n | F1 AAGCN | F1 ST-GCN | Δ |
|---|---:|---:|---:|---:|
| Quần tây | 11 | 0.917 | 0.769 | +0.147 |
| Con heo | 15 | 0.647 | 0.500 | +0.147 |
| Tắm rửa | 11 | 0.952 | 0.818 | +0.134 |
| Dơ | 15 | 0.500 | 0.370 | +0.130 |
| Con vịt | 11 | 0.952 | 0.842 | +0.110 |
| Cậu | 14 | 0.929 | 0.839 | +0.090 |

ST-GCN cao hơn:

| Gloss | n | F1 AAGCN | F1 ST-GCN | Δ |
|---|---:|---:|---:|---:|
| Bánh tét | 14 | 0.643 | 0.897 | −0.254 |
| Bánh chưng | 14 | 0.643 | 0.846 | −0.203 |
| Chậm lại | 7 | 0.824 | 1.000 | −0.176 |
| Sử dụng | 6 | 0.769 | 0.923 | −0.154 |
| Mũ | 11 | 0.741 | 0.880 | −0.139 |
| Thèm | 12 | 0.818 | 0.957 | −0.138 |

Dơ và Con heo nhích hơn ST-GCN, nhưng F1 vẫn quanh 0.50–0.65. Bánh tét và Bánh chưng tụt mạnh, và hai gloss này nhầm chéo nhau.

## Cặp hay nhầm

Chỉ giữ cặp mà gloss đúng bị đoán thành một gloss khác từ 2 lần trở lên. n/N là số lần nhầm trên support của gloss đúng.

### Cả AAGCN và ST-GCN cùng đoán sai một kiểu

| Đúng | Đoán thành | n/N | Tỷ lệ |
|---|---|---:|---:|
| Dơ | Con heo | 7/15 | 47% |
| Chị | Cô | 6/16 | 38% |
| Không cần | Đắng | 4/12 | 33% |
| Thú vị | Rửa tay | 4/15 | 27% |
| Xôi | Ướt | 4/15 | 27% |
| Nhẹ | Bây giờ | 4/17 | 24% |
| Hát | Nhầm lẫn | 3/15 | 20% |
| Bà nội | Ông nội | 2/13 | 15% |
| Ca sĩ | Hát | 2/13 | 15% |
| Thứ ba | Thứ tư | 2/13 | 15% |
| Thứ tư | Thứ ba | 2/13 | 15% |
| Bánh chưng | Bánh tét | 2/14 | 14% |

Nhầm hai chiều trong bảng này: Thứ ba ↔ Thứ tư.

### AAGCN

| Đúng | Đoán thành | n/N |
|---|---|---:|
| Dơ | Con heo | 8/15 |
| Nhẹ | Bây giờ | 7/17 |
| Chị | Cô | 6/16 |
| Thứ tư | Thứ ba | 5/13 |
| Bánh chưng | Bánh tét | 5/14 |
| Bánh tét | Bánh chưng | 5/14 |
| Thú vị | Rửa tay | 5/15 |
| Không cần | Đắng | 4/12 |
| Xôi | Ướt | 4/15 |
| Dữ | Nếm | 3/12 |

Nhầm hai chiều: Bánh chưng ↔ Bánh tét.

### ST-GCN j76

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
