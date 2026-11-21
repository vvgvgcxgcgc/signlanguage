"""Xuất bao-cao-ket-qua.md sang .docx theo khuôn báo cáo dự án (A4, Times New Roman 14).

Bìa lấy từ khối front matter ở đầu tệp md. Sau bìa là mục lục (Word cập nhật khi
mở tệp, hoặc nhấn F9). GIF lấy khung cuối vì khung đầu của sơ đồ gần như trống.

    .venv/bin/python docs/bao-cao-ket-qua/xuat_docx.py   # cần python-docx và pillow
"""

from __future__ import annotations

import importlib.util
import io
import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT, WD_TAB_LEADER
import json
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm, Pt, RGBColor
from PIL import Image

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "bao-cao-ket-qua.md"
TARGET = HERE / "bao-cao-ket-qua.docx"
FONT = "Times New Roman"
BODY_PT = 14
CAPTION_PT = 13
WIDTH_CM = 16.0
STAR = ""

# Dùng lại phần xử lý chữ và bảng của tài liệu ý tưởng.
_spec = importlib.util.spec_from_file_location("y_tuong", HERE.parent / "trinh-bay-y-tuong" / "xuat_docx.py")
yt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(yt)
yt.BODY_PT, yt.TABLE_PT, yt.WIDTH_CM = BODY_PT, 11, WIDTH_CM
add_inline, add_table, split_row, set_style_font = yt.add_inline, yt.add_table, yt.split_row, yt.set_style_font


def field(paragraph, code: str, placeholder: str = "") -> None:
    for kind, text in (("begin", None), (None, code), ("separate", None), ("text", placeholder), ("end", None)):
        run = paragraph.add_run()
        if kind == "text":
            run.text = text
        elif kind:
            mark = OxmlElement("w:fldChar")
            mark.set(qn("w:fldCharType"), kind)
            run._r.append(mark)
        else:
            instr = OxmlElement("w:instrText")
            instr.set(qn("xml:space"), "preserve")
            instr.text = text
            run._r.append(instr)


def setup(document) -> None:
    section = document.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.left_margin, section.right_margin = Cm(3), Cm(2)
    section.top_margin, section.bottom_margin = Cm(2), Cm(2)
    styles = document.styles
    set_style_font(styles["Normal"], BODY_PT)
    styles["Normal"].paragraph_format.space_after = Pt(6)
    styles["Normal"].paragraph_format.line_spacing = 1.3
    black = RGBColor(0, 0, 0)
    for name, size, italic in (("Heading 1", 15, False), ("Heading 2", 14, False), ("Heading 3", 14, False),
                               ("Heading 4", 14, True)):
        set_style_font(styles[name], size, True, black)
        styles[name].font.italic = italic
        styles[name].paragraph_format.space_before = Pt(16 if name == "Heading 1" else 10)
        styles[name].paragraph_format.space_after = Pt(6)
        styles[name].paragraph_format.keep_with_next = True
    styles["Heading 1"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_style_font(styles["List Bullet"], BODY_PT)
    names = [st.name for st in styles]
    for level in (1, 2, 3):
        style = styles[f"TOC {level}"] if f"TOC {level}" in names else styles.add_style(f"TOC {level}", 1)
        set_style_font(style, 13, level == 1)
        fmt = style.paragraph_format
        fmt.left_indent = Cm(0.6 * (level - 1))
        fmt.first_line_indent = Cm(0)
        fmt.space_before = Pt(6 if level == 1 else 0)
        fmt.space_after = Pt(2)
        fmt.line_spacing = 1.15
        fmt.tab_stops.add_tab_stop(Cm(16), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)


def cover(document, meta: dict[str, str]) -> None:
    """Bìa trong khung viền kép, giống bìa báo cáo dự án."""
    table = document.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    borders = "".join(
        f'<w:{side} w:val="double" w:sz="18" w:space="0" w:color="000000"/>' for side in ("top", "left", "bottom", "right")
    )
    table._tbl.tblPr.append(parse_xml(f'<w:tblBorders {nsdecls("w")}>{borders}</w:tblBorders>'))
    row = table.rows[0]
    row.height = Cm(24)
    row._tr.get_or_add_trPr().append(parse_xml(f'<w:trHeight {nsdecls("w")} w:val="{int(24 * 567)}" w:hRule="exact"/>'))
    cell = row.cells[0]
    cell.width = Cm(16)

    def line(text, size=14, bold=False, before=0, after=4, italic=False):
        paragraph = cell.add_paragraph() if cell.paragraphs[0].text or len(cell.paragraphs) > 1 else cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.first_line_indent = Cm(0)
        paragraph.paragraph_format.space_before = Pt(before)
        paragraph.paragraph_format.space_after = Pt(after)
        paragraph.paragraph_format.line_spacing = 1.15
        run = paragraph.add_run(text)
        run.bold, run.italic, run.font.size = bold or None, italic or None, Pt(size)
        return paragraph

    line(meta["co_quan"], 14, True, before=18)
    line(meta["truong"], 14, True)
    line("———————", 12, after=60)
    line(meta["loai"], 20, True, after=30)
    line("ĐỀ TÀI:", 15, True, after=6)
    line(f"“{meta['de_tai']}”", 17, True, after=70)
    authors = [a.strip() for a in meta["tac_gia"].split("|")]
    block = cell.add_paragraph()
    block.paragraph_format.left_indent = Cm(5.2)
    block.paragraph_format.space_after = Pt(2)
    block.add_run("Tác giả:").bold = True
    for k, name in enumerate(authors, 1):
        p = cell.add_paragraph(f"{k}. {name}")
        p.paragraph_format.left_indent = Cm(5.8)
        p.paragraph_format.space_after = Pt(2)
    p = cell.add_paragraph()
    p.paragraph_format.left_indent = Cm(5.2)
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(2)
    p.add_run("Người hướng dẫn:").bold = True
    p = cell.add_paragraph(meta["huong_dan"])
    p.paragraph_format.left_indent = Cm(5.8)
    line(meta["noi_ngay"], 14, italic=True, before=120)


PAGES = HERE / "so-trang.json"   # số trang của từng đề mục, do tinh_so_trang.py tạo


def toc(document):
    """Tiêu đề MỤC LỤC và một đoạn giữ chỗ. Các dòng mục lục được chèn sau, ở fill_toc."""
    heading = document.add_paragraph()
    heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = heading.add_run("MỤC LỤC")
    run.bold, run.font.size = True, Pt(15)
    holder = document.add_paragraph()
    holder.add_run().add_break(WD_BREAK.PAGE)
    return holder


def bookmark(paragraph, name: str, ident: int) -> None:
    start = OxmlElement("w:bookmarkStart")
    start.set(qn("w:id"), str(ident))
    start.set(qn("w:name"), name)
    end = OxmlElement("w:bookmarkEnd")
    end.set(qn("w:id"), str(ident))
    paragraph._p.insert(0, start)
    paragraph._p.append(end)


def _fld(paragraph_el, kind: str) -> None:
    run = OxmlElement("w:r")
    mark = OxmlElement("w:fldChar")
    mark.set(qn("w:fldCharType"), kind)
    run.append(mark)
    paragraph_el.append(run)


def _instr(paragraph_el, code: str) -> None:
    run = OxmlElement("w:r")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = code
    run.append(instr)
    paragraph_el.append(run)


def fill_toc(document, holder, entries: list[tuple[int, str, str]]) -> None:
    """Mục lục điền sẵn: tên mục, dòng chấm, số trang, bấm vào để nhảy tới mục.

    Vẫn là trường TOC của Word, nên Word cập nhật lại được (F9). Số trang lấy từ so-trang.json.
    """
    pages = json.loads(PAGES.read_text(encoding="utf8")) if PAGES.exists() else {}
    for i, (level, text, mark) in enumerate(entries):
        paragraph = holder.insert_paragraph_before(style=f"TOC {level}")
        p = paragraph._p
        if i == 0:
            _fld(p, "begin")
            _instr(p, 'TOC \\o "1-3" \\h \\z \\u')
            _fld(p, "separate")
        link = OxmlElement("w:hyperlink")
        link.set(qn("w:anchor"), mark)
        link.set(qn("w:history"), "1")
        for piece in (text, "\t"):
            r = OxmlElement("w:r")
            if piece == "\t":
                r.append(OxmlElement("w:tab"))
            else:
                t = OxmlElement("w:t")
                t.set(qn("xml:space"), "preserve")
                t.text = piece
                r.append(t)
            link.append(r)
        _fld(link, "begin")
        _instr(link, f" PAGEREF {mark} \\h ")
        _fld(link, "separate")
        r = OxmlElement("w:r")
        t = OxmlElement("w:t")
        t.text = str(pages.get(text, ""))
        r.append(t)
        link.append(r)
        _fld(link, "end")
        p.append(link)
        if i == len(entries) - 1:
            _fld(p, "end")


def picture(document, path: Path, alt: str) -> None:
    image = Image.open(path)
    if getattr(image, "n_frames", 1) > 1:
        image.seek(image.n_frames - 1)
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "PNG")
    buffer.seek(0)
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_with_next = True
    paragraph.paragraph_format.space_before = Pt(6)
    paragraph.add_run().add_picture(buffer, width=Cm(WIDTH_CM))._inline.docPr.set("descr", alt)


def equation(document, path: Path, number: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.line_spacing = 1.0
    paragraph.paragraph_format.space_before = Pt(4)
    paragraph.paragraph_format.space_after = Pt(8)
    paragraph.add_run().add_picture(str(path), width=Cm(WIDTH_CM))._inline.docPr.set("descr", f"Công thức ({number})")


def page_numbers(document) -> None:
    section = document.sections[0]
    section.different_first_page_header_footer = True
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    field(footer, "PAGE", "2")


def restore_stars(document) -> None:
    def fix(paragraphs):
        for paragraph in paragraphs:
            for run in paragraph.runs:
                if STAR in run.text:
                    run.text = run.text.replace(STAR, "*")
    fix(document.paragraphs)
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                fix(cell.paragraphs)


def convert() -> None:
    text = SOURCE.read_text(encoding="utf8").replace("\\*", STAR)
    front = re.match(r"---\n(.*?)\n---\n", text, re.S)
    meta = dict(line.split(": ", 1) for line in front[1].splitlines())
    lines = text[front.end():].splitlines()

    document = Document()
    setup(document)
    document.core_properties.title = "Báo cáo kết quả: Nhận diện ngôn ngữ ký hiệu tiếng Việt"
    cover(document, meta)
    document.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    holder = toc(document)
    entries: list[tuple[int, str, str]] = []
    page_numbers(document)

    i = 0
    in_references = False
    first_part = True
    while i < len(lines):
        line = lines[i]
        i += 1
        if not line.strip():
            continue
        if heading := re.match(r"(#{2,5}) (.+)", line):
            level = len(heading[1]) - 1
            if level == 1 and not first_part:
                document.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
            first_part = first_part and level != 1
            h = document.add_heading(heading[2], level)
            if level <= 3:
                mark = f"_Toc{len(entries) + 1:04d}"
                bookmark(h, mark, len(entries) + 1)
                entries.append((level, heading[2], mark))
            in_references = heading[2] == "TÀI LIỆU THAM KHẢO"
        elif formula := re.match(r"!\[Công thức \((\d+)\)\]\((.+?)\)", line):
            equation(document, HERE / formula[2], formula[1])
        elif image := re.match(r"!\[(.+?)\]\((.+?)\)", line):
            picture(document, HERE / image[2], image[1])
        elif re.match(r"\*(Hình|Bảng) ", line):
            paragraph = document.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.space_after = Pt(10)
            add_inline(paragraph, line.strip().strip("*"), CAPTION_PT, italic=True)
        elif line.startswith("|"):
            block = [line]
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            right = [cell.endswith(":") for cell in split_row(block[1])]
            add_table(document, [split_row(row) for k, row in enumerate(block) if k != 1], right)
            document.paragraphs[-1].paragraph_format.space_after = Pt(0)
        elif line.startswith("- "):
            add_inline(document.add_paragraph(style="List Bullet"), line[2:])
        elif number := re.match(r"(\d+)\. (.+)", line):
            paragraph = document.add_paragraph()
            paragraph.paragraph_format.left_indent = Cm(0.8)
            paragraph.paragraph_format.first_line_indent = Cm(-0.8)
            paragraph.paragraph_format.tab_stops.add_tab_stop(Cm(0.8))
            paragraph.add_run(f"{number[1]}.\t")
            add_inline(paragraph, number[2])
        else:
            paragraph = document.add_paragraph()
            if in_references:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
                paragraph.paragraph_format.left_indent = Cm(0.9)
                paragraph.paragraph_format.first_line_indent = Cm(-0.9)
                add_inline(paragraph, line, BODY_PT - 2)
            else:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                paragraph.paragraph_format.first_line_indent = Cm(1)
                add_inline(paragraph, line)
    fill_toc(document, holder, entries)
    restore_stars(document)
    settings = document.settings.element
    settings.append(parse_xml(f'<w:updateFields {nsdecls("w")} w:val="true"/>'))
    document.save(TARGET)
    print(TARGET, TARGET.stat().st_size)


if __name__ == "__main__":
    convert()
