"""Vẽ các sơ đồ của bản trình bày ý tưởng, mỗi hình xuất một file PDF và một file GIF.

GIF hiện dần từng bước của sơ đồ, PDF là hình đầy đủ để in.

    .venv/bin/python docs/trinh-bay-y-tuong/ve_so_do.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "so-do"
sys.path.insert(0, str(ROOT))

plt.rcParams.update({"font.family": "Arial Unicode MS", "pdf.fonttype": 42})

INK = "#1f2933"
MUTED = "#616e7c"
BLUE = ("#e3effb", "#2b6cb0")
GREEN = ("#e3f5ea", "#2f855a")
AMBER = ("#fdf1d8", "#b7791f")
ROSE = ("#fbe6ea", "#c53048")
GREY = ("#f0f2f5", "#7b8794")
PURPLE = ("#eee8fb", "#6b46c1")


class Figure:
    """Khung vẽ theo đơn vị dữ liệu. Mỗi `step()` mở một nhóm hiện cùng lúc trong GIF."""

    def __init__(self, name: str, width: float, height: float, title: str) -> None:
        self.name = name
        self.fig, self.ax = plt.subplots(figsize=(width / 10, height / 10), dpi=110)
        self.fig.subplots_adjust(0, 0, 1, 1)
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
        self.ax.axis("off")
        self.width, self.height = width, height
        self.groups: list[list] = []
        self._seen: set[int] = set()
        self.ax.text(2, height - 3, title, fontsize=14, fontweight="bold", color=INK, va="center")
        self._seen.update(id(a) for a in self.ax.get_children())

    def step(self) -> None:
        self._collect()
        self.groups.append([])

    def _collect(self) -> None:
        if not self.groups:
            return
        for artist in self.ax.get_children():
            if id(artist) not in self._seen:
                self._seen.add(id(artist))
                self.groups[-1].append(artist)

    def box(self, x, y, w, h, text, color=BLUE, size=10, bold=False):
        self.ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.1",
            facecolor=color[0], edgecolor=color[1], linewidth=1.4,
        ))
        self.ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size * 1.22,
                     color=INK, fontweight="bold" if bold else "normal", linespacing=1.3)

    def band(self, x, y, w, h, label, color=GREY):
        self.ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
            facecolor="none", edgecolor=color[1], linewidth=1.1, linestyle=(0, (4, 3)),
        ))
        self.ax.text(x + 1.2, y + h - 1.8, label, fontsize=10, color=color[1], fontweight="bold", va="center")

    def arrow(self, start, end, color=INK, style="-", rad=0.0, label=None, offset=(0, 1.3)):
        self.ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.4, color=color,
            linestyle=style, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0,
        ))
        if label:
            self.ax.text((start[0] + end[0]) / 2 + offset[0], (start[1] + end[1]) / 2 + offset[1], label,
                         ha="center", va="center", fontsize=8.5, color=MUTED,
                         bbox=dict(facecolor="white", edgecolor="none", pad=1.2))

    def text(self, x, y, text, size=9, color=MUTED, ha="left", bold=False):
        self.ax.text(x, y, text, fontsize=size, color=color, ha=ha, va="center", linespacing=1.3,
                     fontweight="bold" if bold else "normal")

    def save(self) -> None:
        self._collect()
        for group in self.groups:
            for artist in group:
                artist.set_visible(True)
        self.fig.savefig(OUT / f"{self.name}.pdf", facecolor="white")
        frames = []
        for count in range(1, len(self.groups) + 1):
            for index, group in enumerate(self.groups):
                for artist in group:
                    artist.set_visible(index < count)
            self.fig.canvas.draw()
            rgba = np.asarray(self.fig.canvas.buffer_rgba())[..., :3]
            frames.append(Image.fromarray(rgba).convert("P", palette=Image.ADAPTIVE, colors=128))
        durations = [900] * (len(frames) - 1) + [3500]
        frames[0].save(OUT / f"{self.name}.gif", save_all=True, append_images=frames[1:],
                       duration=durations, loop=0, optimize=True)
        plt.close(self.fig)


def tong_quan() -> None:
    f = Figure("hinh-1-tong-quan-he-thong", 150, 82, "Hình 1. Kiến trúc tổng quan của hệ thống")
    f.step()
    f.band(2, 42, 146, 33, "Luồng suy luận thời gian thực trên máy người dùng", BLUE)
    f.box(5, 52, 18, 15, "Webcam\n25 khung/giây\ncắt vuông", GREY)
    f.step()
    f.box(29, 52, 22, 15, "MediaPipe\nPose + Hands\n76 điểm (x, y, z)", BLUE)
    f.arrow((23, 59.5), (29, 59.5))
    f.step()
    f.box(57, 52, 24, 15, "Lọc khung lỗi\nTách ký hiệu\nPoseStateMLP + hàng đợi", AMBER)
    f.arrow((51, 59.5), (57, 59.5))
    f.step()
    f.box(87, 52, 24, 15, "Chuẩn hoá theo vai\nNội suy về T khung\nTensor (T, J, 4)", AMBER)
    f.arrow((81, 59.5), (87, 59.5), label="một clip")
    f.step()
    f.box(117, 52, 28, 15, "Mô hình phân loại\nST-GCN / CTR-GCN\n/ tổ hợp", GREEN, bold=True)
    f.arrow((111, 59.5), (117, 59.5))
    f.step()
    f.box(117, 44.5, 17, 5.5, "Top-5, độ tin cậy", GREEN, size=8)
    f.box(87, 44.5, 24, 5.5, "Giao diện: chữ, giọng nói", PURPLE, size=8)
    f.arrow((125.5, 52), (125.5, 50))
    f.arrow((117, 47.2), (111, 47.2))
    f.step()
    f.band(2, 4, 146, 33, "Luồng ngoại tuyến: dữ liệu và huấn luyện", GREEN)
    f.box(5, 11, 22, 16, "VSL400\nvideo đã gán nhãn\ntrain / test", GREY)
    f.box(33, 11, 22, 16, "Trích keypoint\nfile .npy\n(T, 76, 3)", BLUE)
    f.box(61, 11, 24, 16, "Tiền xử lý\nvà tăng cường\ndữ liệu", AMBER)
    f.box(91, 11, 24, 16, "Huấn luyện và\nso sánh mô hình\nGPU", GREEN)
    f.box(121, 11, 24, 16, "Đánh giá, chọn\nvà đóng gói\ncheckpoint", GREEN)
    for x0, x1 in ((27, 33), (55, 61), (85, 91), (115, 121)):
        f.arrow((x0, 19), (x1, 19))
    f.step()
    f.arrow((140, 27), (140, 52), color=GREEN[1], style="--", label="triển khai", offset=(0, 0))
    f.save()


def khung_xuong() -> None:
    from models.graph import BODY_BONES, BODY_LANDMARKS, CROSS_BONES, HAND_BONES, HAND_LANDMARKS, LEG_BONES

    clip = np.load(ROOT / "vsl400-keypoint/train/Nhớ/780776.npy")
    hand = clip[:, 34:]
    frame = clip[int(np.argmax((hand != 0).all(-1).sum(1)))]
    index = {name: i for i, name in enumerate(BODY_LANDMARKS)}
    for i, name in enumerate(HAND_LANDMARKS):
        index[name + "_0"] = 34 + 2 * i
        index[name + "_1"] = 34 + 2 * i + 1

    def pair_list():
        for bones, group in ((BODY_BONES, "body"), (LEG_BONES, "leg")):
            for a, b in bones:
                yield a, b, group
        for a, b in HAND_BONES:
            yield a + "_0", b + "_0", "hand"
            yield a + "_1", b + "_1", "hand"
        for a, b in CROSS_BONES:
            yield a, b, "cross"

    f = Figure("hinh-2-do-thi-khung-xuong", 120, 92, "Hình 2. Đồ thị khung xương 76 điểm dùng làm đầu vào")
    colors = {"body": BLUE[1], "leg": GREY[1], "hand": GREEN[1], "cross": ROSE[1]}
    xy = frame[:, :2].copy()
    used = (xy != 0).any(1)
    upper = np.array([i for i in range(76) if used[i] and not 25 <= i <= 32])
    lo, hi = xy[upper].min(0), xy[upper].max(0)
    scale = 72 / (hi[1] - lo[1])
    width = (hi[0] - lo[0]) * scale
    origin = np.array([34 - width / 2, 6])

    def project(point):
        return origin + np.array([point[0] - lo[0], hi[1] - point[1]]) * scale

    for group, title in (("body", "34 điểm thân: 33 điểm Pose và điểm cổ (gốc đồ thị)"),
                         ("hand", "42 điểm bàn tay: 21 điểm mỗi tay"),
                         ("cross", "Cạnh nối cổ tay Pose với cổ tay Hands")):
        f.step()
        for a, b, kind in pair_list():
            ia, ib = index[a], index[b]
            if kind != group or not used[ia] or not used[ib]:
                continue
            pa, pb = project(xy[ia]), project(xy[ib])
            if not (0 < pa[1] < 88 and 0 < pb[1] < 88):
                continue
            f.ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=colors[kind], linewidth=1.6)
        members = {"body": range(34), "hand": range(34, 76), "cross": ()}[group]
        points = np.array([project(xy[i]) for i in members if used[i] and not 25 <= i <= 32]).reshape(-1, 2)
        if len(points):
            f.ax.scatter(points[:, 0], points[:, 1], s=16, color=colors[group], zorder=3)
        legend_y = {"body": 66, "hand": 58, "cross": 50}[group]
        f.ax.plot([70, 74], [legend_y, legend_y], color=colors[group], linewidth=3)
        f.text(76, legend_y, title.replace(": ", ":\n", 1) if len(title) > 34 else title, size=9, color=INK)
    f.step()
    f.text(68, 38, "Mỗi điểm mang 4 kênh: x, y, z, cờ hợp lệ.\nĐiểm bị mất giữ nguyên giá trị 0, cờ = 0.\n"
           "Biến thể J = 68 bỏ 8 điểm chân\n(đầu gối, cổ chân, gót, mũi chân; không vẽ ở đây).\n"
           "Toạ độ vẽ từ một khung thật trong VSL400.", size=9)
    f.save()


def quy_trinh() -> None:
    f = Figure("hinh-3-quy-trinh-nghien-cuu", 150, 74, "Hình 3. Quy trình nghiên cứu năm bước")
    stages = [
        ("B1. Dữ liệu", "VSL400\nLọc lớp đủ mẫu\nTrích 76 keypoint\nKiểm tra chất lượng", GREY),
        ("B2. Biểu diễn", "Chuẩn hoá theo vai\nMặt nạ điểm thiếu\nNội suy thời gian\nTăng cường", AMBER),
        ("B3. Mô hình", "Mốc: MS-TCN\nGCN: ST-GCN,\nCTR-GCN, AAGCN\nCải tiến, tổ hợp", GREEN),
        ("B4. Đánh giá", "Tập kiểm tra khoá\nĐánh giá khác miền\nPhân tích cặp nhầm\nĐo độ trễ", ROSE),
        ("B5. Sản phẩm", "Tách ký hiệu\nDemo webcam\nThử với người Điếc\nThu phản hồi", PURPLE),
    ]
    for i, (head, body, color) in enumerate(stages):
        f.step()
        x = 3 + i * 29.3
        f.box(x, 50, 25, 11, head, color, size=11, bold=True)
        f.box(x, 21, 25, 26, body, ("white", color[1]), size=9.5)
        if i:
            f.arrow((x - 4.3, 55.5), (x, 55.5))
    f.step()
    f.ax.add_patch(FancyArrowPatch(
        (134, 21), (15, 21), arrowstyle="-|>", mutation_scale=13, linewidth=1.3, color=MUTED,
        linestyle="--", connectionstyle="bar,fraction=-0.1", shrinkA=0, shrinkB=0,
    ))
    f.text(75, 7, "Vòng lặp: lỗi và phản hồi ở B4, B5 quyết định dữ liệu cần bổ sung và thay đổi mô hình",
           size=10, ha="center")
    f.save()


def kien_truc() -> None:
    f = Figure("hinh-4-kien-truc-mo-hinh", 150, 86, "Hình 4. Khung mô hình dùng chung và điểm khác nhau giữa các nhánh GCN")
    f.step()
    f.box(3, 58, 19, 15, "Đầu vào\n(B, T, J, 4)\nT = 64", GREY)
    f.step()
    f.box(27, 58, 19, 15, "Masked\nSkeletonEmbed\n64 kênh", AMBER)
    f.arrow((22, 65.5), (27, 65.5))
    blocks = [("3 khối\n64 kênh", "T = 64"), ("3 khối\n128 kênh", "T → 32"), ("3 khối\n256 kênh", "T → 16")]
    for i, (body, tail) in enumerate(blocks):
        f.step()
        x = 51 + i * 21
        f.box(x, 58, 17, 15, body, GREEN, bold=True)
        f.text(x + 8.5, 55.5, tail, ha="center")
        f.arrow((x - 4 if i else 46, 65.5), (x, 65.5))
    f.step()
    f.box(114, 58, 15, 15, "Gộp toàn\ncục có\nmặt nạ", AMBER, size=9.5)
    f.box(133, 58, 14, 15, "Linear\nSoftmax\nN = 422", PURPLE, size=9.5)
    f.arrow((110, 65.5), (114, 65.5))
    f.arrow((129, 65.5), (133, 65.5))
    f.step()
    f.band(3, 28, 144, 22, "Bên trong mỗi khối", GREEN)
    f.ax.add_patch(FancyArrowPatch((80, 58), (66, 50), arrowstyle="-", linewidth=1, color=GREEN[1], linestyle=":"))
    f.box(14, 31, 30, 13, "Tích chập đồ thị\ntổng hợp theo khớp (không gian)", BLUE, size=9.5)
    f.box(56, 31, 30, 13, "Tích chập thời gian\nnhân 9 × 1 (ST-GCN)", BLUE, size=9.5)
    f.box(98, 31, 20, 13, "Cộng residual\nReLU", AMBER, size=9.5)
    f.arrow((44, 37.5), (56, 37.5))
    f.arrow((86, 37.5), (98, 37.5))
    f.arrow((7, 37.5), (14, 37.5))
    f.arrow((118, 37.5), (140, 37.5))
    f.step()
    f.band(3, 3, 144, 22, "Cách xây dựng topo đồ thị", ROSE)
    f.box(6, 6, 32, 13, "ST-GCN\nĐồ thị cố định 3 phân vùng\n× edge_importance", ("white", ROSE[1]), size=9)
    f.box(41, 6, 32, 13, "CTR-GCN\nTopo dùng chung + tinh chỉnh\ntheo từng kênh", ("white", ROSE[1]), size=9)
    f.box(76, 6, 32, 13, "AAGCN\nĐồ thị + phần học thêm\n+ topo suy từ clip, attention", ("white", ROSE[1]), size=9)
    f.box(111, 6, 33, 13, "MS-TCN (mốc so sánh)\nKhông dùng đồ thị,\ntích chập giãn theo thời gian", ("white", GREY[1]), size=9)
    f.save()


def tach_ky_hieu() -> None:
    f = Figure("hinh-5-tach-ky-hieu-thoi-gian-thuc", 150, 84, "Hình 5. Tách ký hiệu liên tiếp từ luồng webcam")
    f.step()
    f.box(4, 54, 24, 14, "NGHỈ\ngiữ 1 khung\nmới nhất", GREY, bold=True)
    f.step()
    f.box(48, 54, 24, 14, "ĐANG KÝ HIỆU\nthêm mọi khung\nhợp lệ vào hàng đợi", GREEN, bold=True)
    f.arrow((28, 64), (48, 64), label="P(action) > 0,3")
    f.step()
    f.box(92, 54, 24, 14, "ĐÓNG CLIP\nthêm khung nghỉ\nkết thúc", AMBER, bold=True)
    f.arrow((72, 64), (92, 64), label="P(action) ≤ 0,3")
    f.step()
    f.box(122, 66, 25, 9, "Chấm điểm: đệm 5 + 5 khung\nnghỉ, nội suy về T", GREEN, size=8.5)
    f.box(122, 49, 25, 9, "≥ 50% khung bị loại:\nyêu cầu ký lại", ROSE, size=8.5)
    f.arrow((116, 64), (122, 69), label=None)
    f.arrow((116, 58), (122, 54), label=None)
    f.step()
    f.ax.add_patch(FancyArrowPatch((104, 54), (16, 54), arrowstyle="-|>", mutation_scale=13, linewidth=1.3,
                                   color=MUTED, linestyle="--", connectionstyle="arc3,rad=-0.16",
                                   shrinkA=0, shrinkB=0))
    f.text(60, 44, "chờ tư thế nghỉ mới rồi nhận ký hiệu tiếp theo", ha="center")
    f.step()
    t = np.linspace(0, 1, 500)
    p = 0.08 + 0.85 * (np.exp(-((t - 0.3) / 0.09) ** 4) + np.exp(-((t - 0.73) / 0.1) ** 4))
    x0, y0, w, h = 12, 6, 132, 24
    f.ax.plot(x0 + t * w, y0 + p * h, color=BLUE[1], linewidth=2)
    f.ax.plot([x0, x0 + w], [y0 + 0.3 * h] * 2, color=ROSE[1], linewidth=1.2, linestyle="--")
    f.ax.plot([x0, x0 + w], [y0, y0], color=INK, linewidth=1)
    f.ax.plot([x0, x0], [y0, y0 + h], color=INK, linewidth=1)
    f.ax.fill_between(x0 + t * w, y0, y0 + h, where=p > 0.3, color=GREEN[0], zorder=0)
    f.text(x0 - 1, y0 + h, "1", ha="right")
    f.text(x0 - 1, y0, "0", ha="right")
    f.text(x0 - 1, y0 + 0.3 * h, "0,3", ha="right", color=ROSE[1])
    f.text(x0 + 0.30 * w, y0 + 0.5 * h, "Ký hiệu 1", ha="center", color=GREEN[1], bold=True)
    f.text(x0 + 0.73 * w, y0 + 0.5 * h, "Ký hiệu 2", ha="center", color=GREEN[1], bold=True)
    f.text(x0, 3, "Thời gian. Đường cong P(action) chỉ có tính minh hoạ, không phải dữ liệu đo.", size=8.5)
    f.save()


def danh_gia() -> None:
    f = Figure("hinh-6-thiet-ke-danh-gia", 150, 72, "Hình 6. Thiết kế đánh giá tách bạch giữa chọn mô hình và kiểm tra")
    f.step()
    f.box(4, 44, 34, 15, "VSL400 train\n422 gloss đủ ≥ 20 mẫu", GREY, size=9)
    f.step()
    f.box(48, 51, 25, 9, "Train: học trọng số", GREEN, size=9.5)
    f.box(48, 39, 25, 9, "Validation: chọn mô hình,\nsiêu tham số, ngưỡng", AMBER, size=9)
    f.arrow((38, 53), (48, 55.5))
    f.arrow((38, 49), (48, 44), label=None)
    f.step()
    f.box(84, 45, 25, 11, "Mô hình cuối\nđược khoá", PURPLE, bold=True)
    f.arrow((73, 55.5), (84, 52))
    f.arrow((73, 43.5), (84, 49))
    f.step()
    f.band(3, 4, 144, 30, "Ba mức kiểm tra, mỗi tập chỉ dùng sau khi đã khoá mô hình", ROSE)
    f.box(6, 8, 42, 18, "Mức 1: Cùng miền\nVSL400 test\n5.163 clip, 422 gloss", ROSE, size=9.5)
    f.box(54, 8, 42, 18, "Mức 2: Khác miền\nVideo QIPEDC (SignConnect)\nphần giao 147 gloss, 252 clip", ROSE, size=9.5)
    f.box(102, 8, 42, 18, "Mức 3: Người dùng thật\nNgười Điếc ký trước webcam\nở điều kiện đời thường", ROSE, size=9.5)
    f.arrow((96.5, 45), (96.5, 34), color=MUTED)
    f.step()
    f.box(116, 44, 31, 15, "Chỉ số: Top-1, Top-5,\nMacro-F1, cặp nhầm,\nđộ trễ, ý kiến người dùng", ("white", INK), size=9)
    f.save()


def ke_hoach() -> None:
    tasks = [
        ("Tổng quan tài liệu, chốt phạm vi", 0, 1, GREY),
        ("Chuẩn hoá dữ liệu, trích keypoint", 0, 2, AMBER),
        ("Huấn luyện mốc và các nhánh GCN", 1, 3, GREEN),
        ("Cải tiến: đa tốc độ, mốc tương đối, tổ hợp", 2, 4, GREEN),
        ("Đánh giá ba mức, phân tích lỗi", 3, 5, ROSE),
        ("Xây dựng demo thời gian thực", 2, 5, BLUE),
        ("Thử nghiệm với cộng đồng Điếc", 4, 6, PURPLE),
        ("Hoàn thiện báo cáo và sản phẩm", 5, 6, GREY),
    ]
    f = Figure("hinh-7-ke-hoach-thuc-hien", 150, 66, "Hình 7. Kế hoạch dự kiến trong sáu tháng")
    left, month = 54, 15
    top = 55
    f.step()
    for m in range(7):
        f.ax.plot([left + m * month] * 2, [5, top + 3], color="#d9dee4", linewidth=1)
        if m < 6:
            f.text(left + (m + 0.5) * month, top + 3.5, f"Tháng {m + 1}", ha="center", color=INK, size=9.5, bold=True)
    for i, (name, start, end, color) in enumerate(tasks):
        f.step()
        y = top - 6 * i
        f.text(2, y - 1, name, color=INK, size=9.5)
        f.ax.add_patch(FancyBboxPatch((left + start * month, y - 3), (end - start) * month, 4,
                                      boxstyle="round,pad=0,rounding_size=0.6",
                                      facecolor=color[0], edgecolor=color[1], linewidth=1.2))
    f.save()


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for draw in (tong_quan, khung_xuong, quy_trinh, kien_truc, tach_ky_hieu, danh_gia, ke_hoach):
        draw()
    for path in sorted(OUT.iterdir()):
        print(path.name, path.stat().st_size)
