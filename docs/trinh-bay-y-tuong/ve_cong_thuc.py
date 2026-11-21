"""Vẽ các công thức thành ảnh PNG để hiển thị giống nhau trên Word, Pages, Google Docs và Markdown.

    .venv/bin/python docs/trinh-bay-y-tuong/ve_cong_thuc.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

OUT = Path(__file__).resolve().parent / "cong-thuc"
FONT_PT = 13
DPI = 300
WIDTH_IN = 16 / 2.54

plt.rcParams.update({
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "mathtext.fallback": "stix",
})

# Viết theo LaTeX. Số thứ tự của mỗi công thức là vị trí trong danh sách, tính từ 1,
# và được vẽ luôn vào ảnh, sát lề phải.
FORMULAS = [
    'c_t = \\frac{p_t^{\\text{vai trái}} + p_t^{\\text{vai phải}}}{2}, \\qquad s = \\operatorname{median}_t \\left\\| p_t^{\\text{vai trái}} - p_t^{\\text{vai phải}} \\right\\|_2, \\qquad \\hat{p}_{t,j} = \\frac{p_{t,j} - c_t}{s}',
    'f_{\\text{out}} = \\sum_{k=1}^{3} \\left( \\tilde{A}_k \\odot M_k \\right) f_{\\text{in}} W_k',
    '\\mathcal{L} = -\\sum_{c=1}^{N} \\left[ (1 - \\varepsilon)\\, y_c + \\frac{\\varepsilon}{N} \\right] \\log \\hat{y}_c',
    '\\hat{y} = \\frac{1}{|\\mathcal{M}|} \\sum_{m \\in \\mathcal{M}} \\operatorname{softmax}(z_m)',
    '\\frac{n_{\\text{loại}}}{n_{\\text{giữ}} + n_{\\text{loại}}} \\ge 0{,}5',
    '\\text{Top-}k = \\frac{1}{M} \\sum_{i=1}^{M} \\text{I}\\left( y_i \\in T_k(x_i) \\right)',
    'P_c = \\frac{TP_c}{TP_c + FP_c}, \\qquad R_c = \\frac{TP_c}{TP_c + FN_c}',
    '\\text{Macro-F1} = \\frac{1}{N} \\sum_{c=1}^{N} \\frac{2\\, P_c R_c}{P_c + R_c}',
    'C_{a \\to b} = \\frac{n_{a \\to b}}{n_a}',
    '\\Delta_{\\text{miền}} = \\text{Macro-F1}_{\\text{cùng miền}} - \\text{Macro-F1}_{\\text{khác miền}}',
    '\\text{ECE} = \\sum_{b=1}^{B} \\frac{|B_b|}{M} \\left| \\text{acc}(B_b) - \\text{conf}(B_b) \\right|',
    '\\text{Độ phủ}(\\tau) = \\frac{1}{M} \\sum_{i=1}^{M} \\text{I}\\left( \\text{conf}_i \\ge \\tau \\right)',
    '\\text{tIoU}(S, G) = \\frac{|S \\cap G|}{|S \\cup G|}',
    '\\text{SUS} = 2{,}5 \\left[ \\sum_{i \\text{ lẻ}} (x_i - 1) + \\sum_{i \\text{ chẵn}} (5 - x_i) \\right]',
]


def to_mathtext(tex: str) -> str:
    """Mathtext không có \\text{...} đúng nghĩa và đổi dấu gạch nối thành dấu trừ."""
    def word(match: re.Match) -> str:
        return r"\mathrm{" + match[1].replace(" ", r"\ ").replace("-", "\u2010") + "}"

    tex = re.sub(r"\\text\{([^}]*)\}", word, tex)
    return tex.replace(r"\|", r"\Vert ").replace(r"\ge ", r"\geq ").replace(r"\frac", r"\dfrac")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for number, tex in enumerate(FORMULAS, 1):
        figure = plt.figure(figsize=(WIDTH_IN, 3))
        body = figure.text(0.46, 0.5, f"${to_mathtext(tex)}$", fontsize=FONT_PT, ha="center", va="center", color="black")
        figure.text(0.995, 0.5, f"({number})", fontsize=FONT_PT, ha="right", va="center", color="black", family="Times New Roman")
        figure.canvas.draw()
        box = body.get_window_extent().transformed(figure.dpi_scale_trans.inverted())
        crop = Bbox([[0, box.y0 - 0.08], [WIDTH_IN, box.y1 + 0.08]])
        path = OUT / f"cong-thuc-{number:02d}.png"
        figure.savefig(path, dpi=DPI, bbox_inches=crop, facecolor="white")
        plt.close(figure)
        print(path.name, f"{box.width / WIDTH_IN:.2f}")


if __name__ == "__main__":
    main()
