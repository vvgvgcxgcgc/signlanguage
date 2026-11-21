"""Vẽ các hình về mô hình đề xuất (Hình 25–27 của báo cáo).

Số liệu đọc từ reports/ket-qua-2026-11-21/de-xuat.json (do de_xuat.py tạo).

    .venv/bin/python docs/bao-cao-ket-qua/ve_de_xuat.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "reports" / "ket-qua-2026-11-21" / "de-xuat.json"
OUT = HERE / "bieu-do"
D = json.loads(DATA.read_text())

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ffffff"
BASE, PROP, PROP2 = "#2a78d6", "#eb6834", "#f3b293"
W = 16 / 2.54
for name in ("Times New Roman",):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.dpi": 220,
})
NAME = {
    "j76_MS-TCN": "MS-TCN", "j76_ST-GCN": "ST-GCN", "j76_CTR-GCN": "CTR-GCN", "j76_AAGCN": "AAGCN",
    "j68_AAGCN": "AAGCN J68", "j76_Transformer": "Transformer", "j76_MultiRate-Attn-STGCN": "MultiRate",
    "j76_CTR-GCN-BT": "CTR-GCN-BT", "j76_AnchorGCN-A": "AnchorGCN-A", "j76rest_AnchorGCN-AS": "AnchorGCN-AS",
    "j76rest_AnchorGCN-T32": "AnchorGCN-T32", "ENS4": "Tổ hợp 4 mô hình",
}
PROPOSED = {"j76_MultiRate-Attn-STGCN", "j76_CTR-GCN-BT", "j76_AnchorGCN-A", "j76rest_AnchorGCN-AS", "j76rest_AnchorGCN-T32"}


def comma(v, nd=1):
    return f"{v:.{nd}f}".replace(".", ",")


def color(m):
    return PROP if m in PROPOSED else BASE


HANDLES = [Patch(color=BASE, label="mô hình mốc"), Patch(color=PROP, label="đề xuất của nhóm")]


def legend(ax, loc="lower right"):
    ax.legend(handles=HANDLES, loc=loc, fontsize=9)


def fig_legend(fig, handles=HANDLES, ncol=2):
    fig.legend(handles=handles, loc="lower center", ncol=ncol, fontsize=9, bbox_to_anchor=(0.5, -0.06))


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", OUT / name)


def fig_hard_pairs():
    hp = D["hard_pairs"]
    order = sorted(D["models"], key=lambda m: hp["acc"][m])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, W * 0.5), gridspec_kw={"width_ratios": [1.05, 1]})
    yy = np.arange(len(order))
    a1.barh(yy, [hp["acc"][m] for m in order], color=[color(m) for m in order], height=0.68)
    for i, m in enumerate(order):
        a1.text(hp["acc"][m] + 0.4, i, f"{comma(hp['acc'][m])}  ({hp['confusions'][m]} lần nhầm)", va="center", fontsize=8, color=INK2)
    a1.set_yticks(yy, [NAME[m] for m in order], fontsize=9)
    a1.set_xlim(55, 82)
    a1.set_xlabel("Top-1 (%) trên 202 clip của các cặp khó")
    a1.set_title("Toàn bộ 14 chiều nhầm")
    a1.grid(axis="y", visible=False)

    show = ["j76_MS-TCN", "j76_ST-GCN", "j76_CTR-GCN", "j76_MultiRate-Attn-STGCN", "j76_AnchorGCN-A"]
    cols = ["#b9c7d8", "#6da7ec", BASE, PROP, PROP2]
    groups = list(hp["groups"])
    x = np.arange(len(groups))
    w = 0.16
    for k, (m, c) in enumerate(zip(show, cols)):
        vals = [hp["groups"][g]["acc"][m] for g in groups]
        a2.bar(x + (k - 2) * w, vals, w, color=c, label=NAME[m])
    short = {"nhịp, động tác ngón": "nhịp,\nđộng tác ngón", "điểm chạm trên mặt": "điểm chạm\ntrên mặt", "hình dạng bàn tay": "hình dạng\nbàn tay"}
    a2.set_xticks(x, [f"{short[g]}\n(n = {hp['groups'][g]['n']})" for g in groups], fontsize=8.5)
    a2.set_ylim(30, 92)
    a2.set_ylabel("Top-1 (%)")
    a2.set_title("Theo loại khác biệt")
    a2.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=1.2)
    handles = HANDLES + [Patch(color=c, label=NAME[m]) for m, c in zip(show, cols)]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.13))
    save(fig, "hinh-20-de-xuat-cap-kho.png")


def fig_complement():
    comp = D["complement"]
    fig, axes = plt.subplots(1, 2, figsize=(W, W * 0.42), sharex=True)
    for ax, base in zip(axes, ["j76_ST-GCN", "j76_CTR-GCN"]):
        c = comp[base]
        parts = sorted(c["partners"], key=lambda m: c["partners"][m]["fixed"])
        vals = [c["partners"][m]["fixed"] / c["errors"] * 100 for m in parts]
        yy = np.arange(len(parts))
        ax.barh(yy, vals, color=[color(m) for m in parts], height=0.66)
        for i, m in enumerate(parts):
            ax.text(vals[i] + 0.3, i, f"{comma(vals[i])}%", va="center", fontsize=8, color=INK2)
        ax.set_yticks(yy, [NAME[m] for m in parts], fontsize=8.5)
        ax.set_title(f"Lỗi của {NAME[base]} ({c['errors']} clip)")
        ax.set_xlim(15, 38)
        ax.set_xlabel("% lỗi được mô hình kia đoán đúng")
        ax.grid(axis="y", visible=False)
    fig.tight_layout(w_pad=1.5)
    fig_legend(fig)
    save(fig, "hinh-21-de-xuat-bo-sung.png")


OFFSET = {"j76_MS-TCN": (1.6, -0.7), "j76_ST-GCN": (-1.8, 0.5), "j68_AAGCN": (-1.8, 0.9), "j76_CTR-GCN-BT": (-1.8, -0.9), "j76_AnchorGCN-A": (1.8, 0.4)}


def fig_reliability():
    sel = D["selective_tau07"]
    fig, ax = plt.subplots(figsize=(W * 0.78, W * 0.48))
    for m, v in sel.items():
        c = PROP if m in PROPOSED else ("#0d366b" if m == "ENS4" else BASE)
        ax.scatter(v["qip_cov"], v["qip_acc"], s=40 + 9 * v["oov_accept"], color=c, alpha=0.9, edgecolor="white", linewidth=1, zorder=3)
        dx, dy = OFFSET.get(m, (1.6, 0))
        ax.annotate(f"{NAME[m]}\nnhầm từ lạ {comma(v['oov_accept'])}%", (v["qip_cov"], v["qip_acc"]),
                    xytext=(v["qip_cov"] + dx, v["qip_acc"] + dy), fontsize=8, color=INK2, va="center",
                    ha="left" if dx > 0 else "right")
    ax.set_xlabel("Độ phủ trên QIPEDC ở τ = 0,7 (%)")
    ax.set_ylabel("Độ chính xác khi trả lời (%)")
    ax.set_xlim(34, 74)
    ax.set_ylim(76, 94)
    ax.set_yticks(range(78, 94, 2))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: comma(v, 0)))
    ax.text(35, 93.3, "kích thước điểm: tỷ lệ chấp nhận nhầm từ chưa học", fontsize=8, color=MUTED)
    legend(ax, loc="lower left")
    save(fig, "hinh-22-de-xuat-tin-cay.png")


if __name__ == "__main__":
    fig_hard_pairs()
    fig_complement()
    fig_reliability()
