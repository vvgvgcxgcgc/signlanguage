# ST-GCN vs CTR-GCN vs AAGCN

So sánh ba graph model trong `models/stgcn.py`, `models/ctrgcn.py`, `models/aagcn.py`. Số param và tốc độ đo lại ngày 15 Oct 2026 từ code hiện tại, `num_classes=422`.

- Chỉ ST-GCN có kết quả test (lấy từ `demo-video/confused-pairs.ipynb`). CTR-GCN và AAGCN chưa có checkpoint best trong `checkpoints/`, nên phần accuracy của hai model này đang trống.
- Tập test cũng là tập chọn checkpoint (`training/experiments.yaml`), nên mọi accuracy ở đây là upper bound.

## Phần giống nhau

Ba model chỉ khác nhau ở bên trong block. Mọi thứ còn lại dùng chung:

- Input `(B, T, J, 4)`, qua `MaskedSkeletonEmbed` lên 64 channel.
- 9 block theo `BLOCKS` trong `stgcn.py`: channel 64, 64, 64, 128, 128, 128, 256, 256, 256, stride 2 ở block 4 và 7. Thời gian đi 64 → 32 → 16.
- Graph 3 partition từ `models.graph`, adjacency dạng `[source, target]`, chuẩn hóa theo cột.
- `masked_global_pool` rồi `Linear(256, num_classes)`.
- Cùng config train: SGD lr 0.1, momentum 0.9, weight decay 4e-4, 70 epoch, warmup 5, label smoothing 0.1, EMA 0.999, AMP, batch hiệu dụng 128, chọn best theo macro F1.

## Tổng

| Model | Params J=68 | Params J=76 | So với ST-GCN | CPU ms / batch 16 (J=68) |
|---|---:|---:|---:|---:|
| ST-GCN | 3,265,046 | 3,296,662 | 1.00x | 916 |
| CTR-GCN | 1,640,529 | 1,672,145 | 0.50x | 1,743 |
| AAGCN | 3,919,089 | 3,950,705 | 1.20x | 1,442 |

Tốc độ đo trên CPU Mac, 4 thread, `T=64`, eval mode, `no_grad`, trung bình 5 lần. Chỉ dùng để so tương đối, không phải số GPU.

CTR-GCN ít param nhất nhưng lại chậm nhất. Mỗi partition của nó dựng một topology `(B, C, J, J)` riêng cho từng channel, nên phép einsum lớn hơn hẳn hai model kia, vốn chỉ dùng `(K, J, J)` hoặc `(B, J, J)`.

## Param nằm ở đâu

J=68. Embed 5,184 và head 108,454 giống nhau ở cả ba nên không ghi vào bảng.

| Phần | ST-GCN | CTR-GCN | AAGCN |
|---|---:|---:|---:|
| Spatial (graph conv + topology) | 780,144 | 1,058,731 | 1,152,601 |
| Temporal | 2,323,776 | 426,048 | 2,323,776 |
| Attention | 0 | 0 | 284,274 |
| Residual | 47,488 | 42,112 | 44,800 |

- Temporal chiếm 71% param của ST-GCN. CTR-GCN cắt phần này xuống còn 26%, nên tổng của nó chỉ bằng một nửa.
- Phần spatial của CTR-GCN và AAGCN nặng hơn ST-GCN, vì có thêm query/key và topology học được.
- Phần temporal của AAGCN giống hệt ST-GCN. Param dư so với ST-GCN đến từ spatial và attention.

## Spatial: topology đến từ đâu

| | ST-GCN | CTR-GCN | AAGCN |
|---|---|---|---|
| Graph khai báo | buffer, cố định | khởi tạo `nn.Parameter`, học được | buffer, cố định |
| Phần học thêm | `edge_importance` nhân vào | per-channel refine cộng vào | `offset` cộng vào, khởi tạo 1e-6 |
| Suy ra từ clip | không | có, `tanh(q_u − k_v)` | có, `softmax(q·k)` theo source |
| Topology riêng mỗi channel | không | có | không |
| Shape topology | `(K, J, J)` | `(B, C, J, J)` mỗi partition | `(B, J, J)` mỗi partition |
| Gate cho phần suy ra | — | `alpha`, khởi tạo 0 | `alpha`, khởi tạo 0 |

**ST-GCN.** Một conv 1x1 chiếu ra `out × K` channel, rồi aggregate với `adjacency * edge_importance`. `edge_importance` khởi tạo bằng 1, kích thước `(K, J, J)` cho mỗi block. Phép nhân chỉ scale được những edge đã khai báo: edge nào bằng 0 trong graph thì mãi bằng 0. Graph khai báo thiếu edge là model chịu luôn.

**CTR-GCN.** Mỗi partition có query, key (`max(in // 8, 8)` channel), value và expand. Query và key được mean theo thời gian trước, nên mỗi clip chỉ có một topology chứ không phải mỗi frame một cái. Hiệu `q_u − k_v` qua `tanh` rồi được expand lên `out` channel, nhân `alpha` và cộng vào `topology` dùng chung của block. Kết quả là mỗi output channel có quan hệ joint riêng: channel tả handshape có thể dựa vào link giữa các ngón tay, còn channel tả vị trí cánh tay thì bỏ qua. `topology` là parameter nên edge thiếu cũng học thêm được.

**AAGCN.** Mỗi partition cộng ba ma trận: graph khai báo (buffer), `offset` học tự do, và `alpha * softmax(q·k)` suy ra từ clip. Query và key ở đây có `out // 4` channel, score chia cho `C × T`, softmax theo trục source cho khớp với adjacency chuẩn hóa theo cột. Topology suy ra dùng chung cho mọi channel, khác CTR-GCN. Vì có `offset` nên edge sai có thể bị học ngược về 0, và edge thiếu học thêm được.

Lúc khởi tạo cả ba đều chỉ là graph conv thường trên skeleton khai báo: `edge_importance = 1`, còn `alpha = 0` ở CTR-GCN và AAGCN.

CTR-GCN và AAGCN có thêm residual bên trong graph conv: `relu(BN(gcn) + down(x))`. ST-GCN không có cái này.

## Temporal

| | ST-GCN | CTR-GCN | AAGCN |
|---|---|---|---|
| Kiểu | một conv 9x1, full channel | 4 nhánh song song, mỗi nhánh `out // 4` | một conv 9x1, full channel |
| Receptive field mỗi block | 9 frame | 9 frame (5x1 dilation 2) | 9 frame |
| Activation trong nhánh | — | GELU | — |
| Dropout | 0.2 | không | 0.2 |

Bốn nhánh của CTR-GCN:

1. 1x1 → BN → GELU → 5x1 dilation 1
2. 1x1 → BN → GELU → 5x1 dilation 2
3. 1x1 → BN → GELU → max pool 3x1
4. 1x1 có stride → BN

Bốn nhánh được concat lại. Mỗi conv 5x1 chỉ chạy trên một phần tư số channel, nên số param giảm khoảng 5.5 lần so với conv 9x1 full channel. Receptive field vẫn là 9 frame nhờ nhánh dilation 2.

## Attention (chỉ AAGCN)

Sau residual của mỗi block có ba gate sigmoid, đều ở dạng `x * g + x` nên không thể tắt hẳn feature:

1. Gate theo joint: mean theo thời gian, rồi conv1d kernel 9 trên trục joint.
2. Gate theo frame: mean theo joint, rồi conv1d kernel 9 trên trục thời gian.
3. Gate theo channel: kiểu squeeze-excitation, reduction 2.

Tổng 284,274 param, khoảng 7% của model. Tắt được bằng `attention=False`.

Gate theo joint dùng conv1d trên thứ tự index của joint. Hai joint đứng cạnh nhau trong mảng chưa chắc là hai joint gần nhau trên cơ thể, nên kernel 9 ở đây là một prior hơi tùy tiện.

## Kết quả test

| Model | J | Accuracy | Macro F1 |
|---|---:|---:|---:|
| MS-TCN (mốc) | 68 | 92.66% | 93.07% |
| MS-TCN (mốc) | 76 | 92.70% | 93.20% |
| ST-GCN | 68 | 94.09% | 94.45% |
| ST-GCN | 76 | 94.29% | 94.72% |
| CTR-GCN | 68 / 76 | chưa có | chưa có |
| AAGCN | 68 / 76 | chưa có | chưa có |

- ST-GCN hơn MS-TCN khoảng 1.4–1.6 điểm accuracy, nên graph có giúp.
- Cả MS-TCN và ST-GCN cùng nhầm nhiều nhất ở Dơ → Con heo và Chị → Cô. Đây là những cặp khác nhau chủ yếu ở handshape hoặc vị trí nhỏ trên mặt. Tranh chấp kiểu này chính là chỗ topology riêng mỗi channel của CTR-GCN được thiết kế để xử lý, nên đây là thứ cần nhìn đầu tiên khi có kết quả CTR-GCN.
- `training/runs/04_ctr_gcn_j76.sh` và `08_aagcn_j76.sh` đang trỏ `RESUME` tới checkpoint `_last.pth` trên Kaggle, tức là hai run này đã chạy dở nhưng chưa có best để chấm.

## Chọn model nào

| Tình huống | Model |
|---|---|
| Cần một baseline graph đơn giản để so | ST-GCN |
| Ít param, deploy nhẹ, cần phân biệt handshape | CTR-GCN |
| Nghi graph khai báo sai hoặc thiếu edge | AAGCN |
| Cần inference CPU nhanh nhất trong ba | ST-GCN |
| Ít data | ST-GCN hoặc CTR-GCN. AAGCN nhiều param nhất và có ba nguồn topology nên dễ overfit hơn |

Thứ tự làm việc trong `models/README.md` vẫn đúng: chạy ST-GCN làm baseline graph, tune CTR-GCN, chỉ thêm AAGCN khi cần đa dạng cho ensemble hoặc khi CTR-GCN có dấu hiệu bị graph khai báo giới hạn.
