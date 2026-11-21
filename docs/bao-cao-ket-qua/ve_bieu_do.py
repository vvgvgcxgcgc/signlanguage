"""Vẽ các biểu đồ kết quả cho báo cáo.

Số liệu đọc từ reports/ket-qua-2026-11-21 (report.json, derived.json, curves.json, meta.json).
Ảnh PNG ghi vào bieu-do/, khổ vừa vùng chữ A4 (16 cm).

    .venv/bin/python docs/bao-cao-ket-qua/ve_bieu_do.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "reports" / "ket-qua-2026-11-21"
OUT = HERE / "bieu-do"
OUT.mkdir(exist_ok=True)

report = json.loads((DATA / "report.json").read_text())
derived = json.loads((DATA / "derived.json").read_text())
curves = json.loads((DATA / "curves.json").read_text())
EXCLUDE = {"j68_CTR-GCN", "j76_AnchorGCN", "j76rest_AnchorGCN"}  # lần chạy dở dang, không đưa vào báo cáo
M = {k: v for k, v in report["models"].items() if k not in EXCLUDE}
E = report["ensembles"]
ENS4 = report["best_ensemble"]

# Bảng màu tham chiếu (chế độ sáng), gán theo thứ tự cố định.
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ffffff"
W = 16 / 2.54

for name in ("Times New Roman",):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.dpi": 220, "lines.linewidth": 2,
})

NAME = {
    "j68_MS-TCN": "MS-TCN J68", "j76_MS-TCN": "MS-TCN J76", "j68_ST-GCN": "ST-GCN J68", "j76_ST-GCN": "ST-GCN J76",
    "j68_CTR-GCN": "CTR-GCN J68", "j76_CTR-GCN": "CTR-GCN J76", "j68_AAGCN": "AAGCN J68", "j76_AAGCN": "AAGCN J76",
    "j76_Transformer": "Transformer J76", "j68_MultiRate-Attn-STGCN": "MultiRate J68",
    "j76_MultiRate-Attn-STGCN": "MultiRate J76", "j76_CTR-GCN-BT": "CTR-GCN-BT", "j76_AnchorGCN-A": "AnchorGCN-A",
    "j76_AnchorGCN": "AnchorGCN (lần 1)", "j76rest_AnchorGCN": "AnchorGCN (lần 2)",
    "j76rest_AnchorGCN-AS": "AnchorGCN-AS", "j76rest_AnchorGCN-T32": "AnchorGCN-T32",
}
ENS_NAME = {
    "j76_MS-TCN + j76_ST-GCN": "Tổ hợp MS-TCN + ST-GCN",
    "j76_CTR-GCN + j76_Transformer": "Tổ hợp CTR-GCN + Transformer",
    "j76_CTR-GCN + j76_ST-GCN + j76_Transformer": "Tổ hợp 3 mô hình",
    ENS4: "Tổ hợp 4 mô hình",
}
INCOMPLETE: set[str] = set()


def pct(x):
    return 100 * x


def comma(v, nd=1):
    return f"{v:.{nd}f}".replace(".", ",")


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", name)


def ycomma(ax, nd=0, axis="y"):
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: comma(v, nd))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


# Hình 8. Macro-F1 kèm khoảng tin cậy, mọi mô hình và tổ hợp
def fig_f1_ci():
    rows = [(NAME[k], v["vsl"], k in INCOMPLETE, False) for k, v in M.items()]
    rows += [(ENS_NAME[k], v["vsl"], False, True) for k, v in E.items() if k in ENS_NAME]
    rows.sort(key=lambda r: r[1]["macro_f1"])
    fig, ax = plt.subplots(figsize=(W, 6.4))
    for i, (label, s, inc, ens) in enumerate(rows):
        c = S[1] if ens else S[0]
        lo, hi = pct(s["f1_ci"][0]), pct(s["f1_ci"][1])
        ax.plot([lo, hi], [i, i], color=c, lw=2, solid_capstyle="round")
        ax.scatter(pct(s["macro_f1"]), i, s=46, zorder=3, color=SURFACE if inc else c, edgecolor=c, linewidth=1.6)
        ax.text(hi + 0.15, i, comma(pct(s["macro_f1"]), 2), va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    base = pct(M["j76_MS-TCN"]["vsl"]["macro_f1"])
    ax.axvline(base, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.text(base, len(rows) - 0.4, " mốc MS-TCN J76", fontsize=8.5, color=INK2, va="bottom")
    ax.set_xlim(90, 96.6)
    ax.set_xlabel("Macro-F1 trên VSL400 test (%), đoạn ngang là khoảng tin cậy 95%")
    ycomma(ax, 0, "x")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[
        Line2D([], [], color=S[0], marker="o", lw=2, label="Mô hình đơn"),
        Line2D([], [], color=S[1], marker="o", lw=2, label="Tổ hợp"),
    ], loc="lower right", fontsize=8.5)
    save(fig, "hinh-08-macro-f1-ci.png")


# Hình 9. Đường học Macro-F1 validation
def fig_learning():
    runs = [("j76_CTR-GCN", "CTR-GCN"), ("j76_CTR-GCN-BT", "CTR-GCN-BT"), ("j76_AnchorGCN-A", "AnchorGCN-A"),
            ("j76rest_AnchorGCN-AS", "AnchorGCN-AS"), ("j76rest_AnchorGCN-T32", "AnchorGCN-T32")]
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.2), sharex=False)
    ends = []
    for (run, label), color in zip(runs, S):
        pts = curves[run]
        x = [p["epoch"] for p in pts]
        y = [pct(p["val_f1"]) for p in pts]
        axes[0].plot(x, y, color=color, label=label)
        zx = [a for a, b in zip(x, y) if a >= 15]
        zy = [b for a, b in zip(x, y) if a >= 15]
        if zx:
            axes[1].plot(zx, zy, color=color)
            ends.append([zy[-1], zx[-1], zy[-1], label])
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.5)
    for ly, ex, ey, label in ends:
        if ex >= 15 and ey >= 85:
            axes[1].annotate(label, (ex, ey), xytext=(67, ly), textcoords="data", fontsize=8, color=INK2,
                             va="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    axes[0].set_title("Toàn bộ quá trình")
    axes[0].set_ylim(0, 100)
    axes[1].set_title("Phóng to từ epoch 15")
    axes[1].set_ylim(85, 95.5)
    axes[1].set_xlim(15, 82)
    for ax in axes:
        ax.set_xlabel("Epoch")
        ycomma(ax)
    axes[0].set_ylabel("Macro-F1 validation (%)")
    axes[0].legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    save(fig, "hinh-09-duong-hoc.png")


# Hình 10. Khoảng cách train và validation của CTR-GCN
def fig_gap():
    pts = curves["j76_CTR-GCN"]
    x = [p["epoch"] for p in pts]
    tr = [pct(p["train_f1"]) for p in pts]
    va = [pct(p["val_f1"]) for p in pts]
    fig, ax = plt.subplots(figsize=(W, 2.9))
    ax.plot(x, tr, color=S[0], label="Train")
    ax.plot(x, va, color=S[1], label="Validation (VSL400 test)")
    ax.fill_between(x, va, tr, where=np.array(tr) > np.array(va), color=GRID, alpha=0.6, lw=0)
    ax.annotate(f"train {comma(tr[-1])}%", (x[-1], tr[-1]), xytext=(4, 4), textcoords="offset points", fontsize=8.5, color=INK2)
    ax.annotate(f"validation {comma(va[-1])}%", (x[-1], va[-1]), xytext=(4, -10), textcoords="offset points", fontsize=8.5, color=INK2)
    ax.set_ylim(60, 101)
    ax.set_xlim(1, 76)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro-F1 (%)")
    ycomma(ax)
    ax.legend(loc="lower right", fontsize=8.5)
    save(fig, "hinh-10-train-validation.png")


# Hình 11. Độ chính xác so với chi phí suy luận
def fig_cost():
    fig, ax = plt.subplots(figsize=(W, 3.9))
    labels = {"j76_MS-TCN", "j76_ST-GCN", "j76_CTR-GCN", "j76_AAGCN", "j76_Transformer", "j76_MultiRate-Attn-STGCN",
              "j76_CTR-GCN-BT", "j76_AnchorGCN-A", "j76rest_AnchorGCN-T32", "j76rest_AnchorGCN-AS", "j76_AnchorGCN"}
    for k, v in M.items():
        if not k.startswith("j76"):
            continue
        x, y = v["cpu_ms_median"], pct(v["vsl"]["macro_f1"])
        size = 25 + 22 * v["params"] / 1e6
        inc = k in INCOMPLETE
        ax.scatter(x, y, s=size, color=SURFACE if inc else S[0], edgecolor=S[0], linewidth=1.5, zorder=3)
        if k in labels:
            off = {"j76_AnchorGCN-A": (7, -10), "j76_CTR-GCN": (7, 4), "j76_CTR-GCN-BT": (-105, -1),
                   "j76_ST-GCN": (-40, 8), "j76_AAGCN": (7, -10), "j76rest_AnchorGCN-AS": (7, 3),
                   "j76rest_AnchorGCN-T32": (7, -10), "j76_MultiRate-Attn-STGCN": (-58, -4),
                   "j76_AnchorGCN": (8, -2)}.get(k, (7, 2))
            text = "AnchorGCN* (2 lần chạy, 14 epoch)" if k == "j76_AnchorGCN" else NAME[k].replace(" J76", "")
            arrow = dict(arrowstyle="-", color=MUTED, lw=0.6) if k == "j76_CTR-GCN-BT" else None
            ax.annotate(text, (x, y), xytext=off, textcoords="offset points", fontsize=8, color=INK2,
                        arrowprops=arrow)
    e = E[ENS4]
    ax.scatter(e["cpu_ms_median"], pct(e["vsl"]["macro_f1"]), s=25 + 22 * e["params"] / 1e6, color=S[1], zorder=3)
    ax.annotate("Tổ hợp 4 mô hình", (e["cpu_ms_median"], pct(e["vsl"]["macro_f1"])), xytext=(-40, 8),
                textcoords="offset points", fontsize=8, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100], ["1", "2", "5", "10", "20", "50", "100"])
    ax.set_xlabel("Thời gian phân loại một clip trên CPU Apple M4 (ms, thang log)")
    ax.set_ylabel("Macro-F1 VSL400 (%)")
    ax.set_ylim(90.5, 96.3)
    ycomma(ax)
    ax.legend(handles=[
        Line2D([], [], ls="", marker="o", color=S[0], label="Mô hình J76 (kích thước điểm theo số tham số)"),
        Line2D([], [], ls="", marker="o", color=S[1], label="Tổ hợp"),
    ], loc="upper left", fontsize=8)
    save(fig, "hinh-11-chi-phi-do-chinh-xac.png")


# Hình 12. Phân bố F1 theo gloss
def fig_f1_bins():
    rows = [("MS-TCN J76", M["j76_MS-TCN"]), ("AAGCN J76", M["j76_AAGCN"]), ("ST-GCN J76", M["j76_ST-GCN"]),
            ("CTR-GCN J76", M["j76_CTR-GCN"]), ("Tổ hợp 4 mô hình", E[ENS4])]
    names = ["< 0,50", "0,50–0,80", "0,80–0,90", "0,90–0,95", "≥ 0,95"]
    colors = [BLUE[0], BLUE[1], BLUE[2], BLUE[4], BLUE[6]]
    fig, ax = plt.subplots(figsize=(W, 2.7))
    for i, (label, v) in enumerate(rows):
        left = 0
        for j, n in enumerate(v["f1_bins"]):
            if n:
                ax.barh(i, n - 1.2, left=left + 0.6, color=colors[j], height=0.62)
                if n >= 18:
                    ax.text(left + n / 2, i, str(n), ha="center", va="center", fontsize=8,
                            color=SURFACE if j >= 3 else INK)
            left += n
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlim(0, 422)
    ax.set_xlabel("Số gloss (tổng 422) theo khoảng F1 của từng gloss")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(color=c, label=n) for c, n in zip(colors, names)], ncol=5, fontsize=8,
              loc="upper center", bbox_to_anchor=(0.45, 1.22))
    save(fig, "hinh-12-phan-bo-f1.png")


# Hình 13. Heatmap các cặp hay nhầm
def fig_heatmap():
    keys = ["Dơ→Con heo", "Con heo→Dơ", "Chị→Cô", "Cô→Chị", "Nhẹ→Bây giờ", "Không cần→Đắng", "Xôi→Ướt",
            "Thứ tư→Thứ ba", "Bánh tét→Bánh chưng", "Hát→Nhầm lẫn"]
    rows = [("MS-TCN", "j76_MS-TCN"), ("ST-GCN", "j76_ST-GCN"), ("CTR-GCN", "j76_CTR-GCN"),
            ("AAGCN", "j76_AAGCN"), ("CTR-GCN-BT", "j76_CTR-GCN-BT"), ("AnchorGCN-A", "j76_AnchorGCN-A"),
            ("MultiRate", "j76_MultiRate-Attn-STGCN"), ("Tổ hợp 4 mô hình", "ENS:" + ENS4)]
    H = report["hard_pairs"]
    rate = np.array([[H[k][p][0] / H[k][p][1] for p in keys] for _, k in rows])
    fig, ax = plt.subplots(figsize=(W, 3.6))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#f4f8fd"] + BLUE[1:])
    im = ax.imshow(rate, cmap=cmap, vmin=0, vmax=0.65, aspect="auto")
    for i, (_, k) in enumerate(rows):
        for j, p in enumerate(keys):
            n, d = H[k][p]
            ax.text(j, i, f"{n}/{d}", ha="center", va="center", fontsize=7.5,
                    color=SURFACE if rate[i, j] > 0.33 else INK)
    ax.set_xticks(range(len(keys)), [k.replace("→", " →\n") for k in keys], fontsize=7)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8.5)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(keys)), minor=True)
    ax.set_yticks(np.arange(-.5, len(rows)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("Tỷ lệ nhầm n/N", color=INK2)
    cb.outline.set_visible(False)
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: comma(v, 1)))
    save(fig, "hinh-13-cap-nham.png")


# Hình 14. J68 so với J76
def fig_legs():
    fams = [("MS-TCN", "MS-TCN"), ("ST-GCN", "ST-GCN"), ("AAGCN", "AAGCN"), ("MultiRate", "MultiRate-Attn-STGCN")]
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.1))
    for ax, split, title, lim in ((axes[0], "vsl", "VSL400 (cùng miền)", (92.5, 95.2)),
                                  (axes[1], "qip", "QIPEDC (khác miền)", (54, 70))):
        for (label, fam), color in zip(fams, S):
            a, b = pct(M[f"j68_{fam}"][split]["macro_f1"]), pct(M[f"j76_{fam}"][split]["macro_f1"])
            ax.plot([0, 1], [a, b], color=color, marker="o", ms=6)
            ax.annotate(f"{label} {comma(b, 1)}", (1, b), xytext=(6, 0), textcoords="offset points",
                        fontsize=8, color=INK2, va="center")
            ax.annotate(comma(a, 1), (0, a), xytext=(-6, 0), textcoords="offset points", fontsize=8,
                        color=INK2, va="center", ha="right")
        ax.set_xticks([0, 1], ["J = 68\n(bỏ chân)", "J = 76\n(giữ chân)"])
        ax.set_xlim(-0.35, 1.75)
        ax.set_ylim(*lim)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ycomma(ax)
    axes[0].set_ylabel("Macro-F1 (%)")
    fig.tight_layout()
    save(fig, "hinh-14-diem-chan.png")


# Hình 15. Suy giảm khác miền
def fig_domain():
    keys = ["j76_MS-TCN", "j68_MS-TCN", "j76_Transformer", "j76_MultiRate-Attn-STGCN", "j76_AnchorGCN-A",
            "j76_CTR-GCN-BT", "j76_ST-GCN", "j76_CTR-GCN", "j76rest_AnchorGCN-T32", "j68_ST-GCN", "j76_AAGCN",
            "j68_AAGCN"]
    rows = [(NAME[k], M[k]) for k in keys] + [("Tổ hợp 4 mô hình", E[ENS4])]
    rows.sort(key=lambda r: r[1]["qip"]["top1"])
    fig, ax = plt.subplots(figsize=(W, 4.6))
    for i, (label, v) in enumerate(rows):
        a, b = pct(v["vsl147"]["top1"]), pct(v["qip"]["top1"])
        ax.plot([b, a], [i, i], color=GRID, lw=3, solid_capstyle="round", zorder=1)
        ax.scatter(a, i, color=S[0], s=40, zorder=3)
        ax.scatter(b, i, color=S[1], s=40, zorder=3)
        ax.text(b - 1, i, comma(b), ha="right", va="center", fontsize=8, color=INK2)
        ax.text((a + b) / 2, i + 0.28, f"−{comma(pct(v['drop_top1']), 0)}%", ha="center", fontsize=7.5, color=MUTED)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlim(45, 100)
    ax.set_xlabel("Top-1 trên 147 gloss chung (%)")
    ycomma(ax, 0, "x")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Line2D([], [], ls="", marker="o", color=S[0], label="VSL400 test"),
                       Line2D([], [], ls="", marker="o", color=S[1], label="QIPEDC")],
              loc="upper left", fontsize=8.5)
    save(fig, "hinh-15-suy-giam-khac-mien.png")


# Hình 16. Top-k
def fig_topk():
    D = derived["models"]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.9), sharey=False)
    for ax, split, title, lim in ((axes[0], "vsl", "VSL400", (92, 99.5)), (axes[1], "qip", "QIPEDC", (50, 80))):
        for (k, label), c in zip((("j76_CTR-GCN", "CTR-GCN J76"), ("j68_AAGCN", "AAGCN J68"), ("ENS4", "Tổ hợp 4 mô hình")), S):
            y = [pct(v) for v in D[k][f"{split}_topk"]]
            ax.plot(range(1, 11), y, color=c, marker="o", ms=4, label=label)
        ax.axvline(5, color=MUTED, lw=1, ls=(0, (3, 3)))
        ax.set_xticks(range(1, 11))
        ax.set_ylim(*lim)
        ax.set_title(title)
        ax.set_xlabel("k (số từ gợi ý)")
        ycomma(ax)
    axes[0].set_ylabel("Top-k (%)")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, "hinh-16-top-k.png")


# Hình 17. Biểu đồ độ tin cậy
def fig_reliability():
    D = derived["models"]
    fig, axes = plt.subplots(1, 4, figsize=(W, 2.6), sharey=True)
    panels = [("j76_CTR-GCN", "vsl", "CTR-GCN · VSL400"), ("j76_CTR-GCN", "qip", "CTR-GCN · QIPEDC"),
              ("j76_CTR-GCN-BT", "vsl", "CTR-GCN-BT · VSL400"), ("j76_CTR-GCN-BT", "qip", "CTR-GCN-BT · QIPEDC")]
    for ax, (k, s, title) in zip(axes, panels):
        rel = D[k][f"{s}_rel"]
        centers = np.arange(0.05, 1.0, 0.1)
        acc = [r[1] if r[1] is not None else 0 for r in rel]
        color = S[0] if s == "vsl" else S[1]
        for cx, a, r in zip(centers, acc, rel):
            ax.bar(cx, a, width=0.092, color=color, alpha=1.0 if r[2] >= 10 else 0.3, zorder=2)
        ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls=(0, (3, 3)))
        ece = M[k][s]["ece"]
        ax.text(0.04, 0.9, f"ECE {comma(pct(ece))}%", fontsize=8, color=INK)
        ax.set_title(title, fontsize=8.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1], ["0", "0,5", "1"])
        ax.set_xlabel("Độ tin cậy", fontsize=8.5)
    axes[0].set_ylabel("Độ chính xác thực tế")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: comma(v, 1)))
    fig.tight_layout()
    save(fig, "hinh-17-do-tin-cay.png")


# Hình 18. Ngưỡng từ chối τ
def fig_threshold():
    D, taus = derived["models"], np.array(derived["taus"])
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.8))
    for (k, label), c in zip((("j76_CTR-GCN", "CTR-GCN J76"), ("j76_MS-TCN", "MS-TCN J76"), ("ENS4", "Tổ hợp 4")), S):
        for split, ls in (("vsl", "-"), ("qip", (0, (4, 2)))):
            acc = [pct(v) if v is not None and c >= 0.05 else np.nan
                   for v, c in zip(D[k][f"{split}_acc"], D[k][f"{split}_cov"])]
            axes[0].plot(taus, acc, color=c, ls=ls, lw=1.8)
            axes[1].plot(taus, [pct(v) for v in D[k][f"{split}_cov"]], color=c, ls=ls, lw=1.8)
        axes[2].plot(taus, [pct(v) for v in D[k]["oov_accept"]], color=c, lw=1.8, label=label)
    titles = ("Độ chính xác khi trả lời", "Độ phủ (tỷ lệ được trả lời)", "Chấp nhận nhầm từ chưa học")
    for ax, t in zip(axes, titles):
        ax.axvline(0.7, color=MUTED, lw=1, ls=(0, (3, 3)))
        ax.set_title(t, fontsize=9)
        ax.set_xlabel("Ngưỡng τ")
        ax.set_xlim(0, 0.98)
        ax.set_ylim(0, 102)
        ax.set_xticks([0, 0.3, 0.5, 0.7, 0.9], ["0", "0,3", "0,5", "0,7", "0,9"])
        ycomma(ax)
    axes[0].set_ylim(50, 101)
    axes[0].set_ylabel("%")
    handles = [Line2D([], [], color=c, label=l) for c, l in zip(S, ("CTR-GCN J76", "MS-TCN J76", "Tổ hợp 4 mô hình"))]
    handles += [Line2D([], [], color=INK2, label="VSL400"), Line2D([], [], color=INK2, ls=(0, (4, 2)), label="QIPEDC")]
    fig.legend(handles=handles, ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    save(fig, "hinh-18-nguong-tu-choi.png")


# Hình 19. Thời gian huấn luyện mỗi epoch
def fig_train_time():
    names = {"j76_ST-GCN": "ST-GCN J76", "j68_ST-GCN": "ST-GCN J68", "j68_MultiRate-Attn-STGCN": "MultiRate J68",
             "j76_MultiRate-Attn-STGCN": "MultiRate J76", "j68_AAGCN": "AAGCN J68", "j76_AAGCN": "AAGCN J76",
             "j76_CTR-GCN-BT": "CTR-GCN-BT", "j76_CTR-GCN": "CTR-GCN J76",
             "j76_AnchorGCN-A": "AnchorGCN-A", "j76rest_AnchorGCN-AS": "AnchorGCN-AS",
             "j76rest_AnchorGCN-T32": "AnchorGCN-T32"}
    rows = []
    for k, label in names.items():
        secs = sorted(p["sec"] for p in curves[k] if p.get("sec") and p["sec"] < 4 * 3600)
        if secs:
            rows.append((label, secs[len(secs) // 2] / 60))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(W, 3.6))
    for i, (label, m) in enumerate(rows):
        ax.barh(i, m, color=S[0], height=0.6)
        ax.text(m + 0.6, i, f"{comma(m, 0)} phút · ~{comma(m * 70 / 60, 0)} giờ / 70 epoch", va="center",
                fontsize=8, color=INK2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlim(0, 80)
    ax.set_xlabel("Trung vị thời gian một epoch trên 2 × T4 (phút)")
    ax.grid(axis="y", visible=False)
    save(fig, "hinh-19-thoi-gian-huan-luyen.png")


if __name__ == "__main__":
    for f in (fig_f1_ci, fig_learning, fig_gap, fig_cost, fig_f1_bins, fig_heatmap, fig_legs, fig_domain,
              fig_topk, fig_reliability, fig_threshold, fig_train_time):
        f()
