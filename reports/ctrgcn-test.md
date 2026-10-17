# CTR-GCN j76 trên tập test

Infer `checkpoints/CTR-GCN/j76_CTR-GCN_best.pth` ngày 17 Oct 2026. Thư mục checkpoint chỉ có bản j76. ST-GCN j76 infer cùng lượt để so. Hàng MS-TCN lấy từ `reports/ms-tcn-test.md`, hàng AAGCN lấy từ `reports/aagcn-test.md`, cùng tập test.

- Label: `labels/j76_CTR-GCN.json`, 422 gloss. Thứ tự lớp trong checkpoint trùng file này.
- Test: 5,163 clip hợp lệ. 50 thư mục test nằm ngoài danh sách label nên không được chấm.
- Eval mode, không augment. Loss là cross-entropy, label smoothing 0.1.
- j76 giữ chân (76 joint), 64 frame.
- Tập test này cũng là tập dùng để chọn checkpoint lúc train. Best được chọn theo macro F1, rơi vào epoch 43.

## Tổng

| Model | Epoch | Params | Loss | Accuracy | Top-5 | Macro F1 | Đúng / 5,163 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MS-TCN j76 | 59 | 707,254 | 1.306 | 92.70% | 98.39% | 93.20% | 4,786 |
| AAGCN j76 | 40 | 3,950,705 | 1.256 | 93.12% | 98.33% | 93.50% | 4,808 |
| CTR-GCN j76 | 43 | 1,672,145 | 1.234 | 94.11% | 98.66% | 94.52% | 4,859 |
| ST-GCN j76 | 59 | 3,296,662 | 1.219 | 94.29% | 98.61% | 94.72% | 4,868 |

Số lưu trong checkpoint CTR-GCN lúc chọn best: acc 94.17% / top-5 98.66% / macro F1 94.57%. Lần chạy này lệch tối đa 0.06 điểm.

CTR-GCN thấp hơn ST-GCN 9 clip. Macro F1 thấp hơn 0.20 điểm, top-5 cao hơn 0.05 điểm. Param bằng một nửa ST-GCN (1.67M so với 3.30M) và thấp hơn AAGCN 51 clip dù ít param hơn nhiều.

CTR-GCN và ST-GCN đồng ý top-1 trên 95.7% clip. 4,776 clip cả hai đúng, 83 clip chỉ CTR-GCN đúng, 92 clip chỉ ST-GCN đúng, 212 clip cả hai sai.

## Phân bố F1

Support trung bình 12.2 clip/gloss. CTR-GCN có 184 gloss F1 = 1, ST-GCN có 190.

| Khoảng F1 | CTR-GCN | ST-GCN |
|---|---:|---:|
| < 0.50 | 0 | 1 |
| 0.50–0.80 | 23 | 21 |
| 0.80–0.90 | 48 | 43 |
| 0.90–0.95 | 65 | 68 |
| ≥ 0.95 | 286 | 289 |

CTR-GCN không còn gloss nào dưới 0.50. Gloss dưới ngưỡng đó của ST-GCN là Dơ.

## Gloss yếu nhất

F1 CTR-GCN thấp nhất.

| Gloss | Support | F1 CTR-GCN | F1 ST-GCN | Recall CTR-GCN | Recall ST-GCN |
|---|---:|---:|---:|---:|---:|
| Thứ ba | 13 | 0.545 | 0.583 | 0.462 | 0.538 |
| Dơ | 15 | 0.593 | 0.370 | 0.533 | 0.333 |
| Khóc | 17 | 0.615 | 0.645 | 0.471 | 0.588 |
| Thứ tư | 13 | 0.621 | 0.571 | 0.692 | 0.615 |
| Con heo | 15 | 0.625 | 0.500 | 0.667 | 0.533 |
| Tháng một | 12 | 0.667 | 0.769 | 0.750 | 0.833 |
| Xôi | 15 | 0.667 | 0.690 | 0.600 | 0.667 |
| Chị | 16 | 0.667 | 0.714 | 0.625 | 0.625 |
| Cô | 15 | 0.667 | 0.743 | 0.733 | 0.867 |
| Nhẹ | 17 | 0.667 | 0.750 | 0.588 | 0.706 |
| Nhầm lẫn | 11 | 0.692 | 0.696 | 0.818 | 0.727 |
| Bây giờ | 15 | 0.706 | 0.750 | 0.800 | 0.800 |

## Chỗ CTR-GCN tách khỏi ST-GCN

Δ = F1 CTR-GCN − F1 ST-GCN. Sáu gloss mỗi phía có |Δ| lớn nhất.

CTR-GCN cao hơn:

| Gloss | n | F1 CTR-GCN | F1 ST-GCN | Δ |
|---|---:|---:|---:|---:|
| Dơ | 15 | 0.593 | 0.370 | +0.222 |
| Thứ năm | 12 | 0.870 | 0.667 | +0.203 |
| Tắm rửa | 11 | 1.000 | 0.818 | +0.182 |
| Con heo | 15 | 0.625 | 0.500 | +0.125 |
| Con vịt | 11 | 0.952 | 0.842 | +0.110 |
| Khô | 12 | 1.000 | 0.909 | +0.091 |

ST-GCN cao hơn:

| Gloss | n | F1 CTR-GCN | F1 ST-GCN | Δ |
|---|---:|---:|---:|---:|
| Tháng chín | 12 | 0.750 | 0.909 | −0.159 |
| Chậm lại | 7 | 0.875 | 1.000 | −0.125 |
| Kem | 12 | 0.800 | 0.917 | −0.117 |
| Trường học | 19 | 0.829 | 0.944 | −0.115 |
| Có | 12 | 0.889 | 1.000 | −0.111 |
| Tháng một | 12 | 0.667 | 0.769 | −0.103 |

Dơ và Con heo là hai gloss CTR-GCN kéo lên nhiều nhất. Chị giữ nguyên 6/16 nhầm sang Cô, bằng ST-GCN.

## Cặp hay nhầm

Chỉ giữ cặp mà gloss đúng bị đoán thành một gloss khác từ 2 lần trở lên. n/N là số lần nhầm trên support của gloss đúng.

### Cả CTR-GCN và ST-GCN cùng đoán sai một kiểu

| Đúng | Đoán thành | n/N | Tỷ lệ |
|---|---|---:|---:|
| Dơ | Con heo | 7/15 | 47% |
| Chị | Cô | 6/16 | 38% |
| Nhẹ | Bây giờ | 5/17 | 29% |
| Không cần | Đắng | 4/12 | 33% |
| Xôi | Ướt | 4/15 | 27% |
| Con heo | Dơ | 3/15 | 20% |
| Hát | Nhầm lẫn | 3/15 | 20% |
| Thú vị | Rửa tay | 3/15 | 20% |
| Bà nội | Ông nội | 2/13 | 15% |
| Thứ ba | Thứ tư | 2/13 | 15% |
| Bánh chưng | Bánh tét | 2/14 | 14% |
| Ngọt | Muốn | 2/14 | 14% |

Nhầm hai chiều: Dơ ↔ Con heo.

### CTR-GCN

| Đúng | Đoán thành | n/N |
|---|---|---:|
| Dơ | Con heo | 7/15 |
| Nhẹ | Bây giờ | 7/17 |
| Chị | Cô | 6/16 |
| Thứ ba | Thứ tư | 5/13 |
| Hát | Nhầm lẫn | 5/15 |
| Xôi | Ướt | 5/15 |
| Nghề nghiệp | Làm việc | 5/16 |
| Khóc | Chú ý | 5/17 |
| Không cần | Đắng | 4/12 |
| Con heo | Dơ | 4/15 |

Nhầm hai chiều: Dơ ↔ Con heo.

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
