"""Hình minh hoạ từ video VSL400: khung hình kèm khung xương 76 điểm.

Video lấy từ bản VSL400 đã cắt 224×224 trên Kaggle (thư mục test/<gloss>/<id>.mp4),
keypoint lấy từ vsl400-keypoint/test/<gloss>/<id>.npy, hai bên khớp từng khung.
Video không nằm trong kho mã (theo thoả thuận dữ liệu). Không truyền gì thì từng clip cần
được tải riêng từ Kaggle (assets.vsl400_video) vào data/vsl400-videos/; video QIPEDC lấy từ
data/signconnect/Videos (python -m preprocess.download_signconnect):

    .venv/bin/python docs/bao-cao-ket-qua/ve_khung_hinh.py
    .venv/bin/python docs/bao-cao-ket-qua/ve_khung_hinh.py /đường/dẫn/thư-mục-video   # đã có sẵn video
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import assets  # noqa: E402
from models.graph import build_edges  # noqa: E402
from preprocess.keypoints import LANDMARKS  # noqa: E402

OUT = HERE / "bieu-do"
KEYPOINTS = ROOT / "vsl400-keypoint" / "test"
VIDEOS = Path(sys.argv[1]) if len(sys.argv) > 1 else None

INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#ffffff"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
W = 16 / 2.54
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10, "figure.facecolor": SURFACE,
                     "savefig.dpi": 220})

NAMES, EDGES = build_edges(True)
IDX = {name: i for i, name in enumerate(LANDMARKS)}
EDGE_IDX = [(IDX[NAMES[a]], IDX[NAMES[b]]) for a, b in EDGES]
HAND0 = {i for i, n in enumerate(LANDMARKS) if n.endswith("_0")}
HAND1 = {i for i, n in enumerate(LANDMARKS) if n.endswith("_1")}

PAIRS = [
    ("Dơ", "001418", "Con heo", "000669"),
    ("Chị", "003417", "Cô", "000014"),
    ("Nhẹ", "001463", "Bây giờ", "000200"),
    ("Không cần", "000789", "Đắng", "000796"),
    ("Xôi", "004491", "Ướt", "000537"),
    ("Bánh tét", "001124", "Bánh chưng", "004510"),
]


def video_path(gloss: str, vid: str) -> Path:
    if VIDEOS is None:
        return assets.vsl400_video(gloss, vid)
    found = list(VIDEOS.rglob(f"{gloss}/{vid}.mp4"))
    if not found:
        raise FileNotFoundError(f"thiếu video {gloss}/{vid}.mp4 trong {VIDEOS}")
    return found[0]


def blur_face(frame: np.ndarray, pts: np.ndarray, size: int = 224) -> np.ndarray:
    """Làm mờ vùng mặt quanh 11 điểm mặt của Pose (mũi, mắt, tai, miệng)."""
    face = pts[:11, :2]
    face = face[(face != 0).any(1)] * size
    if not len(face):
        return frame
    cx, cy = face.mean(0)
    half = max(np.ptp(face[:, 0]) * 0.9, np.ptp(face[:, 1]) * 1.6, 16)
    x0, x1 = int(max(0, cx - half)), int(min(size, cx + half))
    y0, y1 = int(max(0, cy - half * 1.25)), int(min(size, cy + half * 1.05))
    if x1 > x0 and y1 > y0:
        frame[y0:y1, x0:x1] = cv2.GaussianBlur(frame[y0:y1, x0:x1], (0, 0), 6)
    return frame


def frames(gloss: str, vid: str) -> np.ndarray:
    """Khung hình VSL400, vùng mặt luôn được làm mờ (không dựa vào việc video đã khử nhận dạng)."""
    k = keypoints(gloss, vid)
    cap = cv2.VideoCapture(str(video_path(gloss, vid)))
    out = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t = len(out)
        if t < len(k):
            frame = blur_face(frame, k[t])
        out.append(frame)
    return np.stack(out)


def keypoints(gloss: str, vid: str) -> np.ndarray:
    path = KEYPOINTS / gloss / f"{vid}.npy"
    return np.load(path if path.is_file() else assets.vsl400_keypoint("test", gloss, vid))


def active_range(k: np.ndarray) -> tuple[int, int]:
    """Đoạn có ít nhất một bàn tay được nhận ra."""
    hands = [i for i in range(len(k)) if (k[i, sorted(HAND0 | HAND1), :2] != 0).any()]
    return (hands[0], hands[-1]) if hands else (0, len(k) - 1)


def draw(ax, image: np.ndarray, pts: np.ndarray, size: int = 224, body_color: str = "#f4f4f2") -> None:
    ax.imshow(image)
    xy = pts[:, :2] * size
    seen = (pts[:, :2] != 0).any(1)
    for a, b in EDGE_IDX:
        hand_edge = a in HAND0 | HAND1 or b in HAND0 | HAND1
        if seen[a] and seen[b] and (not hand_edge or np.linalg.norm(xy[a] - xy[b]) < 40):
            color = BLUE if a in HAND0 or b in HAND0 else ORANGE if a in HAND1 or b in HAND1 else body_color
            ax.plot([xy[a, 0], xy[b, 0]], [xy[a, 1], xy[b, 1]], color=color, lw=1.1, alpha=0.95)
    body = [i for i in range(34) if seen[i] and i not in (25, 26, 27, 28, 29, 30, 31, 32)]
    ax.scatter(xy[body, 0], xy[body, 1], s=4, color=body_color, edgecolor="#333", linewidth=0.3, zorder=3)
    ax.set_xlim(0, size)
    ax.set_ylim(size, 0)
    ax.axis("off")


def pick(k: np.ndarray, count: int) -> list[int]:
    start, end = active_range(k)
    return [int(round(x)) for x in np.linspace(start, end, count)]


def fig_pair(a, ida, b, idb, name: str, count: int = 6) -> None:
    fig, axes = plt.subplots(2, count, figsize=(W, W * 2 / count + 0.35))
    for row, (gloss, vid, color) in enumerate(((a, ida, BLUE), (b, idb, ORANGE))):
        imgs, k = frames(gloss, vid), keypoints(gloss, vid)
        for col, t in enumerate(pick(k, count)):
            draw(axes[row, col], imgs[t], k[t])
            axes[row, col].set_title(f"khung {t + 1}", fontsize=7.5, color=INK2, pad=2)
        axes[row, 0].text(-0.08, 0.5, gloss, transform=axes[row, 0].transAxes, rotation=90, ha="right",
                          va="center", fontsize=10.5, fontweight="bold", color=INK)
    fig.subplots_adjust(left=0.05, right=0.995, top=0.93, bottom=0.01, wspace=0.04, hspace=0.16)
    fig.savefig(OUT / name, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", name)


def normalized(k: np.ndarray) -> np.ndarray:
    """Toạ độ theo vai: gốc ở trung điểm hai vai, đơn vị là độ rộng vai (giống công thức 1)."""
    ls, rs = IDX["leftShoulder"], IDX["rightShoulder"]
    ok = (k[:, ls, :2] != 0).any(1) & (k[:, rs, :2] != 0).any(1)
    center = (k[:, ls, :2] + k[:, rs, :2]) / 2
    width = np.median(np.linalg.norm(k[ok, ls, :2] - k[ok, rs, :2], axis=1))
    out = (k[:, :, :2] - center[:, None]) / width
    out[(k[:, :, :2] == 0).all(-1)] = np.nan
    return out


def fig_trajectories(name: str) -> None:
    """Quỹ đạo cổ tay chuyển động nhiều nhất (theo toạ độ vai) của từng cặp, chồng lên nhau."""
    fig, axes = plt.subplots(2, 3, figsize=(W, W * 0.8))
    nose, ls, rs = IDX["nose"], IDX["leftShoulder"], IDX["rightShoulder"]
    wrists = (IDX["leftWrist"], IDX["rightWrist"])
    for ax, (a, ida, b, idb) in zip(axes.flat, PAIRS):
        ref = normalized(keypoints(a, ida))
        ax.scatter(np.nanmedian(ref[:, nose, 0]), np.nanmedian(ref[:, nose, 1]), s=70, marker="o",
                   facecolor="none", edgecolor=MUTED, lw=1)
        ax.plot([np.nanmedian(ref[:, ls, 0]), np.nanmedian(ref[:, rs, 0])],
                [np.nanmedian(ref[:, ls, 1]), np.nanmedian(ref[:, rs, 1])], color=MUTED, lw=1.5)
        for gloss, vid, color in ((a, ida, BLUE), (b, idb, ORANGE)):
            n = normalized(keypoints(gloss, vid))
            for w in wrists:
                path = n[:, w]
                if np.nanmax(path[:, 1]) - np.nanmin(path[:, 1]) < 0.6:
                    continue  # tay gần như đứng yên
                ax.plot(path[:, 0], path[:, 1], color=color, lw=1.6, alpha=0.9)
                first = np.where(np.isfinite(path[:, 0]))[0][0]
                ax.scatter(*path[first], color=color, s=14, zorder=3)
        ax.set_title(f"{a} (xanh) – {b} (cam)", fontsize=9)
        ax.set_xlim(-1.7, 1.7)
        ax.set_ylim(3.2, -2.0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#e4e3df")
    handles = [plt.Line2D([], [], color=BLUE, lw=2, label="từ thứ nhất"),
               plt.Line2D([], [], color=ORANGE, lw=2, label="từ thứ hai"),
               plt.Line2D([], [], color=INK2, marker="o", ls="", ms=4, label="điểm bắt đầu"),
               plt.Line2D([], [], color=MUTED, marker="o", ls="", markerfacecolor="none", label="mũi"),
               plt.Line2D([], [], color=MUTED, lw=1.5, label="hai vai")]
    fig.legend(handles=handles, ncol=5, fontsize=8, loc="lower center", frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.06, wspace=0.05, hspace=0.15)
    fig.savefig(OUT / name, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", name)


def fig_sequence(gloss: str, vid: str, name: str, count: int = 8) -> None:
    """Một ký hiệu là một chuỗi khung hình: ảnh gốc ở hàng trên, khung xương ở hàng dưới."""
    imgs, k = frames(gloss, vid), keypoints(gloss, vid)
    idx = [int(round(x)) for x in np.linspace(0, len(k) - 1, count)]
    fig, axes = plt.subplots(2, count, figsize=(W, W * 2 / count + 0.3))
    for col, t in enumerate(idx):
        axes[0, col].imshow(imgs[t])
        axes[0, col].axis("off")
        axes[0, col].set_title(f"t = {t / 25:.2f} s".replace(".", ","), fontsize=7.5, color=INK2, pad=2)
        draw(axes[1, col], np.full_like(imgs[t], 255), k[t], body_color="#8a8984")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.92, bottom=0.01, wspace=0.04, hspace=0.06)
    fig.savefig(OUT / name, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", name)


def fig_landmarks(gloss: str, vid: str, name: str) -> None:
    """Một khung hình với ba nhóm điểm: thân (34), bàn tay 1 (21), bàn tay 2 (21)."""
    imgs, k = frames(gloss, vid), keypoints(gloss, vid)
    start, end = active_range(k)
    both = [i for i in range(start, end + 1) if (k[i, sorted(HAND0 | HAND1), :2] != 0).all(1).sum() == 42]
    t = min(both or range(start, end + 1), key=lambda i: k[i, [IDX["leftWrist"], IDX["rightWrist"]], 1].mean())
    fig, axes = plt.subplots(1, 3, figsize=(W, W / 3 + 0.55))
    axes[0].imshow(imgs[t])
    axes[0].set_title("Khung hình gốc 224×224", fontsize=9)
    axes[0].axis("off")
    draw(axes[1], imgs[t], k[t])
    axes[1].set_title("Khung xương 76 điểm", fontsize=9)
    xy = k[t, :, :2] * 224
    seen = (k[t, :, :2] != 0).any(1)
    ax = axes[2]
    ax.set_facecolor(SURFACE)
    body = [i for i in range(34) if seen[i] and i not in range(23, 33)]
    ax.scatter(xy[body, 0], xy[body, 1], s=14, color=INK2, label="Thân, mặt (34 điểm)")
    h0 = [i for i in HAND0 if seen[i]]
    h1 = [i for i in HAND1 if seen[i]]
    ax.scatter(xy[h0, 0], xy[h0, 1], s=10, color=BLUE, label="Bàn tay 1 (21 điểm)")
    ax.scatter(xy[h1, 0], xy[h1, 1], s=10, color=ORANGE, label="Bàn tay 2 (21 điểm)")
    for a, b in EDGE_IDX:
        if seen[a] and seen[b]:
            ax.plot([xy[a, 0], xy[b, 0]], [xy[a, 1], xy[b, 1]], color="#c9c8c3", lw=0.7, zorder=0)
    ax.set_xlim(0, 224)
    ax.set_ylim(224, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Các nhóm điểm", fontsize=9)
    fig.legend(*ax.get_legend_handles_labels(), fontsize=8, loc="lower center", frameon=False, ncol=3)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.9, bottom=0.13, wspace=0.05)
    fig.savefig(OUT / name, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", name)


def moving_wrist(k: np.ndarray) -> int:
    """Cổ tay (theo khung thân) có biên độ dọc lớn nhất."""
    def span(w):
        y = k[:, w, 1][k[:, w, 1] != 0]
        return float(y.max() - y.min()) if len(y) else 0.0
    return max((IDX["leftWrist"], IDX["rightWrist"]), key=span)


def fig_normalize(gloss: str, vid: str, name: str) -> None:
    """Chuẩn hoá theo vai và nội suy về 64 khung, minh hoạ trên cổ tay chuyển động của một clip."""
    k = keypoints(gloss, vid)
    w = moving_wrist(k)
    raw = k[:, :, :2].copy()
    raw[(raw == 0).all(-1)] = np.nan
    n = normalized(k)
    mid = len(k) // 2
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.5))
    for ax, data, title in ((axes[0], raw, "Toạ độ ảnh (0–1)"), (axes[1], n, "Sau chuẩn hoá theo vai")):
        pts = data[mid]
        for a, b in EDGE_IDX:
            if a < 34 and b < 34 and np.isfinite(pts[[a, b]]).all() and a not in range(23, 33) and b not in range(23, 33):
                ax.plot(pts[[a, b], 0], pts[[a, b], 1], color="#c9c8c3", lw=1)
        ax.plot(data[:, w, 0], data[:, w, 1], color=ORANGE, lw=1.8, label="quỹ đạo cổ tay")
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal")
        ax.invert_yaxis()
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(1.05, 0)
    axes[1].scatter([0], [0], color=INK2, s=14, zorder=3)
    axes[1].annotate("gốc (0, 0)", (0, 0), xytext=(5, -9), textcoords="offset points", fontsize=7.5, color=INK2)
    axes[1].set_xlim(-2.2, 2.2)
    axes[1].set_ylim(3.6, -2.4)
    y = n[:, w, 1]
    t_raw = np.arange(len(y))
    ok = np.isfinite(y)
    t64 = np.linspace(0, len(y) - 1, 64)
    y64 = np.interp(t64, t_raw[ok], y[ok])
    axes[2].plot(t_raw, y, color=MUTED, lw=1, marker="o", ms=2.2, label=f"gốc ({len(y)} khung)")
    axes[2].plot(t64, y64, color=BLUE, lw=1.4, label="sau nội suy (64 khung)")
    axes[2].invert_yaxis()
    axes[2].set_title("Độ cao cổ tay theo thời gian", fontsize=9)
    axes[2].set_xlabel("khung", fontsize=8.5)
    axes[2].legend(fontsize=7, frameon=False, loc="lower center")
    for ax in axes:
        ax.tick_params(labelsize=7.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / name, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", name)


PRED = None


def predictions() -> dict:
    global PRED
    if PRED is None:
        import json
        PRED = json.loads((ROOT / "reports" / "ket-qua-2026-11-21" / "vi-du-du-doan.json").read_text())
    return PRED


def peak_frames(k: np.ndarray, count: int) -> list[int]:
    """Các khung lấy đều trong đoạn có tay, luôn gồm khung tay giơ cao nhất."""
    start, end = active_range(k)
    picks = [int(round(x)) for x in np.linspace(start, end, count + 2)[1:-1]]
    w = moving_wrist(k)
    ys = [(k[t, w, 1] if k[t, w, 1] != 0 else 9, t) for t in range(start, end + 1)]
    top = min(ys)[1]
    picks[min(range(count), key=lambda i: abs(picks[i] - top))] = top
    return sorted(set(picks))


def bars(ax, top: list, truth: str) -> None:
    names = [g for g, _ in top[:3]][::-1]
    vals = [100 * v for _, v in top[:3]][::-1]
    colors = [AQUA if g == truth else "#c9c8c3" for g in names]
    ax.barh(range(len(names)), vals, color=colors, height=0.55)
    for i, (g, v) in enumerate(zip(names, vals)):
        ax.text(1, i + 0.32, g, fontsize=7.5, color=INK, va="bottom", fontweight="bold" if g == truth else None)
        ax.text(99, i + 0.32, f"{v:.1f}%".replace(".", ","), fontsize=7.5, color=INK2, va="bottom", ha="right")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, len(names) - 0.1)
    ax.axis("off")


def fig_correct(items: list[tuple[str, str]], name: str, count: int = 4) -> None:
    """Mỗi hàng: các khung của một clip được nhận đúng, bên phải là 3 dự đoán đầu."""
    rows = len(items)
    fig = plt.figure(figsize=(W, rows * 1.32 + 0.1))
    grid = fig.add_gridspec(rows, count + 1, width_ratios=[1] * count + [1.35], wspace=0.04, hspace=0.12,
                            left=0.075, right=0.995, top=0.985, bottom=0.01)
    pred = predictions()["vsl"]
    for r, (gloss, vid) in enumerate(items):
        imgs, k = frames(gloss, vid), keypoints(gloss, vid)
        for c, t in enumerate(peak_frames(k, count)[:count]):
            ax = fig.add_subplot(grid[r, c])
            draw(ax, imgs[t], k[t])
        ax0 = fig.axes[-count] if count <= len(fig.axes) else fig.axes[0]
        ax0.text(-0.08, 0.5, gloss, transform=ax0.transAxes, rotation=90, ha="right", va="center",
                 fontsize=9.5, fontweight="bold", color=INK)
        bars(fig.add_subplot(grid[r, count]), pred[f"{gloss}/{vid}"], gloss)
    fig.savefig(OUT / name, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", name)


FACES = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def video_frames(path: Path, blur_faces: bool = True) -> np.ndarray:
    """Đọc video QIPEDC. Mặt được làm mờ để giống cách VSL400 khử nhận dạng."""
    cap = cv2.VideoCapture(str(path))
    out, box = [], None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if blur_faces:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found = FACES.detectMultiScale(gray, 1.1, 4, minSize=(18, 18))
            if len(found):
                box = max(found, key=lambda b: b[2] * b[3])
            if box is not None:
                x, y, w, h = box
                x0, y0 = max(0, x - w // 5), max(0, y - h // 4)
                x1, y1 = min(frame.shape[1], x + w + w // 5), min(frame.shape[0], y + h + h // 5)
                frame[y0:y1, x0:x1] = cv2.GaussianBlur(frame[y0:y1, x0:x1], (0, 0), 7)
        out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return np.stack(out)


def fig_qipedc(items: list[str], name: str, count: int = 4) -> None:
    """Video QIPEDC (khác nguồn): các khung giữa video và 3 dự đoán đầu."""
    rows = len(items)
    fig = plt.figure(figsize=(W, rows * 1.32 + 0.1))
    grid = fig.add_gridspec(rows, count + 1, width_ratios=[1] * count + [1.35], wspace=0.04, hspace=0.12,
                            left=0.075, right=0.995, top=0.985, bottom=0.01)
    pred = predictions()["qip"]
    for r, stem in enumerate(items):
        imgs = video_frames(ROOT / "data" / "signconnect" / "Videos" / f"{stem}.mp4")
        idx = [int(round(x)) for x in np.linspace(0.2 * len(imgs), 0.8 * len(imgs), count)]
        for c, t in enumerate(idx):
            ax = fig.add_subplot(grid[r, c])
            ax.imshow(imgs[t])
            ax.axis("off")
        gloss = pred[stem]["gloss"]
        fig.axes[-count].text(-0.08, 0.5, gloss, transform=fig.axes[-count].transAxes, rotation=90, ha="right",
                              va="center", fontsize=9.5, fontweight="bold", color=INK)
        bars(fig.add_subplot(grid[r, count]), pred[stem]["top5"], gloss)
    fig.savefig(OUT / name, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", name)


def fig_gallery(items: list[tuple[str, str]], name: str, cols: int = 4) -> None:
    """Thư viện ký hiệu: khung tay giơ cao nhất của mỗi ký hiệu, kèm khung xương."""
    rows = (len(items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(W, rows * W / cols * 1.08))
    for ax, (gloss, vid) in zip(axes.flat, items):
        imgs, k = frames(gloss, vid), keypoints(gloss, vid)
        start, end = active_range(k)
        w = moving_wrist(k)
        t = min(range(start, end + 1), key=lambda i: k[i, w, 1] if k[i, w, 1] != 0 else 9)
        draw(ax, imgs[t], k[t])
        ax.set_title(gloss, fontsize=9.5, color=INK, pad=3)
    for ax in list(axes.flat)[len(items):]:
        ax.axis("off")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.95, bottom=0.005, wspace=0.04, hspace=0.14)
    fig.savefig(OUT / name, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", name)


def fig_domain(items: list[tuple[str, str, str]], name: str, count: int = 4) -> None:
    """Cùng một từ: VSL400 (hàng trên) và QIPEDC (hàng dưới)."""
    fig, axes = plt.subplots(2 * len(items), count, figsize=(W, 2 * len(items) * W / count * 0.98))
    for b, (gloss, vid, stem) in enumerate(items):
        imgs, k = frames(gloss, vid), keypoints(gloss, vid)
        start, end = active_range(k)
        for c, t in enumerate(np.linspace(start, end, count + 2)[1:-1].round().astype(int)):
            axes[2 * b, c].imshow(imgs[t])
            axes[2 * b, c].axis("off")
        q = video_frames(ROOT / "data" / "signconnect" / "Videos" / f"{stem}.mp4")
        for c, t in enumerate(np.linspace(0.2 * len(q), 0.8 * len(q), count).round().astype(int)):
            axes[2 * b + 1, c].imshow(q[t])
            axes[2 * b + 1, c].axis("off")
        axes[2 * b, 0].text(-0.06, 0.5, f"{gloss}\nVSL400", transform=axes[2 * b, 0].transAxes, rotation=90,
                            ha="right", va="center", fontsize=8.5, color=INK)
        axes[2 * b + 1, 0].text(-0.06, 0.5, f"{gloss}\nQIPEDC", transform=axes[2 * b + 1, 0].transAxes,
                                rotation=90, ha="right", va="center", fontsize=8.5, color=INK)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.995, bottom=0.005, wspace=0.03, hspace=0.05)
    fig.savefig(OUT / name, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", name)


if __name__ == "__main__":
    fig_landmarks("Bánh chưng", "004510", "hinh-kn-khung-xuong-that.png")
    fig_sequence("Anh", "000000", "hinh-kn-chuoi-khung-hinh.png")
    fig_normalize("Con heo", "000669", "hinh-kn-chuan-hoa.png")
    for i, (a, ida, b, idb) in enumerate(PAIRS, 1):
        fig_pair(a, ida, b, idb, f"hinh-cap-{i}.png")
    fig_trajectories("hinh-cap-quy-dao.png")
    fig_correct([("Cảm ơn", "687080"), ("Xin lỗi", "003024"), ("Bác sĩ", "009110")], "hinh-dung-1.png")
    fig_correct([("Việt Nam", "022001"), ("Chụp hình", "002206"), ("Trà", "000275")], "hinh-dung-2.png")
    fig_correct([("Dơ", "001418"), ("Con heo", "023884"), ("Chị", "010427"), ("Cô", "023282")], "hinh-dung-cap-kho.png")
    fig_qipedc(["Trà", "Ô tô", "Khám bệnh", "Con mèo(2)"], "hinh-dung-qipedc.png")
    fig_gallery([("Cảm ơn", "687080"), ("Xin lỗi", "003024"), ("Bác sĩ", "009110"), ("Việt Nam", "022001"),
                 ("Chụp hình", "002206"), ("Trà", "000275"), ("Xe đạp", "014581"), ("Bóng đá", "001920"),
                 ("Con chó", "002347"), ("Giáng sinh", "003611"), ("Gội đầu", "018259"),
                 ("Mũ bảo hiểm", "001201")], "hinh-pl-thu-vien.png")
    fig_domain([("Ô tô", "018154", "Ô tô"), ("Chụp hình", "002206", "Chụp hình")], "hinh-pl-hai-nguon.png")
