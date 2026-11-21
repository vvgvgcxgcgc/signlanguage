"""Sơ đồ đơn giản giải thích mạng đồ thị (GCN) bằng hình người que.

    .venv/bin/python docs/bao-cao-ket-qua/ve_gcn.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "bieu-do"
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#ffffff"
BONE = "#b9b8b2"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
W = 16 / 2.54
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10, "figure.facecolor": SURFACE,
                     "savefig.dpi": 220})

# Người que đang ký: tay phải đưa lên cằm, tay trái buông.
P = {
    "mũi": (0.0, 1.62), "miệng": (0.0, 1.5), "cổ": (0.0, 1.25),
    "vai P": (-0.42, 1.2), "vai T": (0.42, 1.2),
    "khuỷu P": (-0.66, 0.86), "khuỷu T": (0.62, 0.74),
    "cổ tay P": (-0.36, 1.36), "cổ tay T": (0.68, 0.3),
    "ngón P1": (-0.2, 1.62), "ngón P2": (-0.12, 1.54), "ngón P3": (-0.14, 1.42),
    "ngón T1": (0.56, 0.04), "ngón T2": (0.7, -0.02), "ngón T3": (0.84, 0.06),
    "hông P": (-0.3, 0.2), "hông T": (0.3, 0.2),
}
BONES = [("mũi", "miệng"), ("miệng", "cổ"), ("cổ", "vai P"), ("cổ", "vai T"), ("vai P", "khuỷu P"),
         ("khuỷu P", "cổ tay P"), ("vai T", "khuỷu T"), ("khuỷu T", "cổ tay T"),
         ("cổ tay P", "ngón P1"), ("cổ tay P", "ngón P2"), ("cổ tay P", "ngón P3"),
         ("cổ tay T", "ngón T1"), ("cổ tay T", "ngón T2"), ("cổ tay T", "ngón T3"),
         ("vai P", "hông P"), ("vai T", "hông T"), ("hông P", "hông T")]
NEIGHBORS = {a: set() for a in P}
for a, b in BONES:
    NEIGHBORS[a].add(b)
    NEIGHBORS[b].add(a)


def person(ax, offset=(0, 0), scale=1.0, bone_color=BONE, bone_lw=2.2, joints=True, alpha=1.0,
           bone_styles=None):
    ox, oy = offset

    def at(n):
        x, y = P[n]
        return ox + x * scale, oy + y * scale

    ax.add_patch(Circle(at("mũi"), 0.2 * scale, fill=False, ec=bone_color, lw=bone_lw, alpha=alpha))
    for a, b in BONES:
        style = (bone_styles or {}).get((a, b)) or (bone_styles or {}).get((b, a)) or {}
        ax.plot(*zip(at(a), at(b)), color=style.get("color", bone_color), lw=style.get("lw", bone_lw),
                alpha=alpha, solid_capstyle="round", zorder=1)
    if joints:
        xs, ys = zip(*[at(n) for n in P])
        ax.scatter(xs, ys, s=16 * scale, color="#8f8e89", zorder=2, alpha=alpha)
    return at


def node(ax, xy, color, size=150, z=4):
    ax.scatter(*xy, s=size, color=color, edgecolor=INK, lw=0.8, zorder=z)


def arrow(ax, a, b, color, ls="-", rad=0.0, lw=1.8):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color=color, lw=lw, ls=ls,
                                 shrinkA=6, shrinkB=7, zorder=3, connectionstyle=f"arc3,rad={rad}"))


def frame(ax, title, xlim=(-1.05, 1.05), ylim=(-0.2, 1.95)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=9.5, pad=4)


def caption(ax, text):
    ax.text(0.5, -0.02, text, transform=ax.transAxes, ha="center", va="top", fontsize=8.3, color=INK2,
            linespacing=1.25)


def fig_principle(name: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(W, 2.75))
    # 1. Khung xương là đồ thị
    person(axes[0])
    frame(axes[0], "1. Khung xương là đồ thị")
    caption(axes[0], "Khớp là đỉnh,\nxương là cạnh")

    # 2. Gom thông tin từ hàng xóm
    at = person(axes[1])
    target = "cổ tay T"
    for n in NEIGHBORS[target]:
        node(axes[1], at(n), BLUE, 70)
        arrow(axes[1], at(n), at(target), BLUE)
    node(axes[1], at(target), ORANGE, 160, 5)
    frame(axes[1], "2. Gom thông tin hàng xóm")
    caption(axes[1], "Cổ tay (cam) cộng thông tin\ncủa khuỷu tay và các ngón (xanh)")

    # 3. Nhiều lớp: thông tin lan xa
    at = person(axes[2])
    hop1 = NEIGHBORS[target]
    hop2 = set().union(*(NEIGHBORS[n] for n in hop1)) - hop1 - {target}
    hop3 = set().union(*(NEIGHBORS[n] for n in hop2)) - hop2 - hop1 - {target}
    for group, color in ((hop3, "#cde2fb"), (hop2, "#86b6ef"), (hop1, BLUE)):
        for n in group:
            node(axes[2], at(n), color, 70)
    node(axes[2], at(target), ORANGE, 160, 5)
    frame(axes[2], "3. Xếp nhiều lớp")
    caption(axes[2], "Sau 1, 2, 3 lớp (xanh đậm → nhạt)\ncổ tay \"nhìn thấy\" tới vai, cổ")

    # 4. Theo thời gian
    ax = axes[3]
    xs = (-0.95, 0.0, 0.95)
    pts = []
    for i, x in enumerate(xs):
        a = person(ax, offset=(x, 0.45), scale=0.5, alpha=0.55 if i != 1 else 1.0, bone_lw=1.5)
        node(ax, a(target), ORANGE if i == 1 else YELLOW, 70, 5)
        pts.append(a(target))
        ax.text(x, 0.2, ["t − 1", "t", "t + 1"][i], ha="center", fontsize=8.5, color=INK2)
    arrow(ax, pts[0], pts[1], YELLOW, ls=(0, (3, 2)), rad=-0.35)
    arrow(ax, pts[2], pts[1], YELLOW, ls=(0, (3, 2)), rad=0.35)
    frame(ax, "4. Thêm chiều thời gian", xlim=(-1.35, 1.4))
    caption(ax, "Cổ tay ở khung t gom thêm\nchính nó ở các khung trước, sau")
    fig.subplots_adjust(left=0.0, right=1.0, top=0.9, bottom=0.2, wspace=0.05)
    fig.savefig(OUT / name, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def fig_compare(name: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(W, 3.05))
    # ST-GCN
    person(axes[0], bone_color="#6f6e69")
    frame(axes[0], "ST-GCN")
    caption(axes[0], "Chỉ dùng các xương có sẵn.\nCổ tay và miệng ở rất xa nhau\ntrên đồ thị, dù rất gần trong ảnh.")

    # AAGCN: thêm cạnh mới, thay đổi theo video
    at = person(axes[1], bone_color="#6f6e69")
    for a, b in (("cổ tay P", "miệng"), ("ngón P1", "mũi"), ("cổ tay P", "cổ tay T")):
        axes[1].plot(*zip(at(a), at(b)), color=ORANGE, lw=2.2, ls=(0, (3, 2)), zorder=3)
    frame(axes[1], "AAGCN")
    caption(axes[1], "Tự học thêm cạnh mới (cam, nét đứt),\nví dụ bàn tay ↔ miệng, tay ↔ tay.\nCác cạnh này thay đổi theo từng video.")

    # CTR-GCN: mỗi kênh một đồ thị
    ax = axes[2]
    hand = {b: {"color": BLUE, "lw": 3.4} for b in BONES if b[0].startswith("cổ tay")}
    arm = {b: {"color": ORANGE, "lw": 3.2} for b in BONES if b[0].startswith(("vai", "khuỷu", "cổ")) and not b[0].startswith("cổ tay")}
    person(ax, offset=(-0.62, 0.25), scale=0.62, bone_styles=hand, bone_lw=1.6)
    person(ax, offset=(0.62, 0.25), scale=0.62, bone_styles=arm, bone_lw=1.6)
    ax.text(-0.62, 0.0, "kênh A: ngón tay", ha="center", fontsize=8.5, color=BLUE)
    ax.text(0.62, 0.0, "kênh B: cánh tay", ha="center", fontsize=8.5, color=ORANGE)
    frame(ax, "CTR-GCN", xlim=(-1.25, 1.25))
    caption(ax, "Mỗi nhóm đặc trưng (kênh) dùng\nđồ thị riêng, nhấn mạnh các cạnh\nkhác nhau, mà tốn rất ít tham số.")
    fig.subplots_adjust(left=0.0, right=1.0, top=0.9, bottom=0.26, wspace=0.08)
    fig.savefig(OUT / name, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


if __name__ == "__main__":
    fig_principle("hinh-gcn-nguyen-ly.png")
    fig_compare("hinh-gcn-so-sanh.png")
