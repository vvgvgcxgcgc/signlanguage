"""Khung hình từ video gốc: clip mà mô hình đề xuất đoán đúng còn mô hình gốc đoán sai.

Dùng lại cách cắt khung, làm mờ mặt và vẽ khung xương của ve_khung_hinh.py.
Dự đoán đọc từ reports/ket-qua-2026-11-21/de-xuat.json (mục "examples").
Video không nằm trong kho mã; không truyền thư mục thì từng clip được tải riêng từ Kaggle:

    .venv/bin/python docs/bao-cao-ket-qua/ve_khung_de_xuat.py [/đường/dẫn/thư-mục-video]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ve_khung_hinh as vk  # noqa: E402  (đọc thư mục video từ sys.argv[1])

EX = json.loads((vk.ROOT / "reports" / "ket-qua-2026-11-21" / "de-xuat.json").read_text())["examples"]
NAME = {"j76_ST-GCN": "ST-GCN", "j76_CTR-GCN": "CTR-GCN", "j76_MultiRate-Attn-STGCN": "MultiRate",
        "j76_AnchorGCN-A": "AnchorGCN-A", "j76_CTR-GCN-BT": "CTR-GCN-BT"}
RIGHT, WRONG, INK, INK2 = "#1baf7a", "#d64545", "#0b0b0b", "#52514e"


def verdict(ax, clip: str, truth: str, models: list[str]) -> None:
    """Mỗi dòng một mô hình: từ đoán ra, độ tin cậy, đúng (xanh) hay sai (đỏ)."""
    n = len(models)
    for r, m in enumerate(models):
        word, p = EX[clip][m][0]
        ok = word == truth
        y = n - 1 - r
        col = RIGHT if ok else WRONG
        ax.barh(y, 100 * p, color=col, alpha=0.9 if ok else 0.75, height=0.42)
        ax.text(0, y + 0.27, f"{NAME[m]}: {word}", fontsize=7.6, color=INK, va="bottom",
                fontweight="bold" if ok else None)
        ax.text(100 * p + 2, y, f"{100 * p:.0f}% · {'đúng' if ok else 'sai'}", fontsize=7.4,
                color=col, va="center", ha="left", fontweight="bold")
    ax.set_xlim(0, 135)
    ax.set_ylim(-0.4, n - 0.05)
    ax.axis("off")


def figure(rows: list[tuple[str, list[str]]], name: str, count: int = 4) -> None:
    fig = plt.figure(figsize=(vk.W, len(rows) * 1.36 + 0.1))
    grid = fig.add_gridspec(len(rows), count + 1, width_ratios=[1] * count + [1.45], wspace=0.04, hspace=0.12,
                            left=0.075, right=0.995, top=0.985, bottom=0.01)
    for r, (clip, models) in enumerate(rows):
        gloss, vid = clip.split("/")
        imgs, k = vk.frames(gloss, vid), vk.keypoints(gloss, vid)
        first = None
        for c, t in enumerate(vk.peak_frames(k, count)[:count]):
            ax = fig.add_subplot(grid[r, c])
            vk.draw(ax, imgs[t], k[t])
            first = first or ax
        first.text(-0.08, 0.5, gloss, transform=first.transAxes, rotation=90, ha="right", va="center",
                   fontsize=9.5, fontweight="bold", color=INK)
        verdict(fig.add_subplot(grid[r, count]), clip, gloss, models)
    fig.savefig(vk.OUT / name, facecolor=vk.SURFACE)
    plt.close(fig)
    print("wrote", name)


if __name__ == "__main__":
    base = ["j76_ST-GCN", "j76_CTR-GCN"]
    figure([("Nhẹ/024459", base + ["j76_MultiRate-Attn-STGCN"]),
            ("Con heo/016671", base + ["j76_MultiRate-Attn-STGCN"]),
            ("Cô/022498", base + ["j76_MultiRate-Attn-STGCN"])], "hinh-de-xuat-dung-1.png")
    figure([("Ướt/023723", base + ["j76_AnchorGCN-A"]),
            ("Bánh chưng/020844", base + ["j76_AnchorGCN-A", "j76_CTR-GCN-BT"]),
            ("Không cần/023058", base + ["j76_CTR-GCN-BT"])], "hinh-de-xuat-dung-2.png")
