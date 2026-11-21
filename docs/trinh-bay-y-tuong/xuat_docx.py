"""Xuất trinh-bay-y-tuong.md sang .docx.

Word in khung đầu của GIF, mà khung đầu của sơ đồ gần như trống, nên hình trong
.docx lấy khung cuối của từng GIF.

    python docs/trinh-bay-y-tuong/xuat_docx.py   # cần python-docx và pillow
"""

from __future__ import annotations

import io
import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm, Pt, RGBColor
from PIL import Image

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "trinh-bay-y-tuong.md"
TARGET = HERE / "trinh-bay-y-tuong.docx"
FONT = "Times New Roman"
BODY_PT = 13
TABLE_PT = 11
WIDTH_CM = 16.0
ACCENT = RGBColor(0, 0, 0)
W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'

def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def add_link(paragraph, url: str, text: str) -> None:
    rel = paragraph.part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), rel)
    link.append(parse_xml(
        f'<w:r {W}><w:rPr><w:color w:val="000000"/></w:rPr>'
        f'<w:t xml:space="preserve">{esc(text)}</w:t></w:r>'
    ))
    paragraph._p.append(link)


INLINE = re.compile(
    r"\*\*(?P<bold>.+?)\*\*|\*(?P<italic>.+?)\*|`(?P<code>[^`]+)`|<sub>(?P<sub>.+?)</sub>"
    r"|\[(?P<label>[^\]]+)\]\((?P<href>[^)]+)\)|<(?P<url>https?://[^>]+)>"
)


def add_inline(paragraph, text: str, size: float | None = None, bold=False, italic=False) -> None:
    def plain(chunk: str, **style) -> None:
        if not chunk:
            return
        run = paragraph.add_run(chunk)
        run.bold = style.get("bold", bold) or None
        run.italic = style.get("italic", italic) or None
        if size:
            run.font.size = Pt(size)

    position = 0
    for match in INLINE.finditer(text):
        plain(text[position:match.start()])
        position = match.end()
        if match["bold"]:
            add_inline(paragraph, match["bold"], size, True, italic)
        elif match["italic"]:
            add_inline(paragraph, match["italic"], size, bold, True)
        elif match["code"]:
            run = paragraph.add_run(match["code"])
            run.font.name = "Courier New"
            run.font.size = Pt((size or BODY_PT) - 1)
        elif match["sub"]:
            run = paragraph.add_run(match["sub"])
            run.font.subscript = True
            run.italic = italic or None
            if size:
                run.font.size = Pt(size)
        elif match["label"]:
            plain(match["label"])
        else:
            add_link(paragraph, match["url"], match["url"])
    plain(text[position:])


def add_equation(document, path: Path, number: str) -> None:
    """Ảnh rộng bằng vùng chữ, số thứ tự đã nằm trong ảnh."""
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.line_spacing = 1.0
    paragraph.paragraph_format.space_before = Pt(4)
    paragraph.paragraph_format.space_after = Pt(8)
    paragraph.add_run().add_picture(str(HERE / path), width=Cm(WIDTH_CM))._inline.docPr.set("descr", f"Công thức ({number})")


def add_figure(document, path: Path, alt: str) -> None:
    frames = Image.open(HERE / path)
    frames.seek(frames.n_frames - 1)
    buffer = io.BytesIO()
    frames.convert("RGB").save(buffer, "PNG")
    buffer.seek(0)
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.first_line_indent = Cm(0)
    paragraph.paragraph_format.keep_with_next = True
    inline = paragraph.add_run().add_picture(buffer, width=Cm(WIDTH_CM)).\
        _inline
    inline.docPr.set("descr", alt)


def add_table(document, rows: list[list[str]], right: list[bool]) -> None:
    if rows[0][-1] == "Tệp":
        rows, right = [row[:-1] for row in rows], right[:-1]
    count = len(rows[0])
    table = document.add_table(rows=len(rows), cols=count)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    weights = [max(max(len(re.sub(r"[*`$]", "", row[c])) for row in rows), 10) ** 0.5 for c in range(count)]
    widths = [Cm(WIDTH_CM * weight / sum(weights)) for weight in weights]
    for index, column in enumerate(table.columns):
        column.width = widths[index]
    for r, row in enumerate(rows):
        properties = table.rows[r]._tr.get_or_add_trPr()
        properties.append(OxmlElement("w:cantSplit"))
        if r == 0:
            properties.append(OxmlElement("w:tblHeader"))
        for c, text in enumerate(row):
            cell = table.cell(r, c)
            cell.width = widths[c]
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.first_line_indent = Cm(0)
            paragraph.paragraph_format.space_after = Pt(2)
            paragraph.paragraph_format.space_before = Pt(2)
            paragraph.paragraph_format.line_spacing = 1.0
            if right[c] and r:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            else:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
            add_inline(paragraph, text, TABLE_PT, bold=r == 0)
            if r == 0:
                cell._tc.get_or_add_tcPr().append(parse_xml(f'<w:shd {nsdecls("w")} w:val="clear" w:color="auto" w:fill="E7E6E6"/>'))
    spacer = document.add_paragraph()
    spacer.paragraph_format.space_after = Pt(4)


def split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def set_style_font(style, size: float, bold=None, color=None) -> None:
    style.font.name = FONT
    style.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), FONT)
    style.element.rPr.rFonts.set(qn("w:cs"), FONT)
    for attribute in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
        style.element.rPr.rFonts.attrib.pop(qn(f"w:{attribute}"), None)
    style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold
    if color is not None:
        style.font.color.rgb = color


def setup(document) -> None:
    section = document.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.left_margin, section.right_margin = Cm(3), Cm(2)
    section.top_margin, section.bottom_margin = Cm(2), Cm(2)
    styles = document.styles
    set_style_font(styles["Normal"], BODY_PT)
    styles["Normal"].paragraph_format.space_after = Pt(6)
    styles["Normal"].paragraph_format.line_spacing = 1.3
    for name, size in (("Heading 1", 14), ("Heading 2", 13), ("Heading 3", 13)):
        set_style_font(styles[name], size, True, ACCENT)
        styles[name].font.italic = name == "Heading 3"
        styles[name].paragraph_format.space_before = Pt(14 if name == "Heading 1" else 10)
        styles[name].paragraph_format.space_after = Pt(6)
    set_style_font(styles["List Bullet"], BODY_PT)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
        run = footer.add_run()
        if kind:
            mark = OxmlElement("w:fldChar")
            mark.set(qn("w:fldCharType"), kind)
            run._r.append(mark)
        else:
            code = OxmlElement("w:instrText")
            code.text = text
            run._r.append(code)


def convert() -> None:
    lines = SOURCE.read_text(encoding="utf8").splitlines()
    document = Document()
    setup(document)
    document.core_properties.title = "Trình bày ý tưởng: Nhận diện ngôn ngữ ký hiệu tiếng Việt"
    i = 0
    in_references = False
    while i < len(lines):
        line = lines[i]
        i += 1
        if not line.strip():
            continue
        if line.startswith("# "):
            paragraph = document.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = paragraph.add_run(line[2:])
            run.bold, run.font.size = True, Pt(16)
        elif heading := re.match(r"(#{2,4}) (.+)", line):
            level = len(heading[1]) - 1
            document.add_heading(heading[2], level)
            in_references = heading[2] == "Tài liệu tham khảo"
        elif line.startswith(">"):
            body = line[1:].strip()
            if body:
                paragraph = document.add_paragraph()
                paragraph.paragraph_format.left_indent = Cm(0.8)
                add_inline(paragraph, body, BODY_PT - 1, italic=True)
        elif formula := re.match(r"!\[Công thức \((\d+)\)\]\((.+?)\)", line):
            add_equation(document, Path(formula[2]), formula[1])
        elif image := re.match(r"!\[(.+?)\]\((.+?)\)", line):
            add_figure(document, Path(image[2]), image[1])
        elif line.startswith("*Hình"):
            caption = re.sub(r"\s*Bản in:.*$", "", line.strip("*"))
            paragraph = document.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_inline(paragraph, caption, BODY_PT - 1, italic=True)
        elif line.startswith("|"):
            block = [line]
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            right = [cell.endswith(":") for cell in split_row(block[1])]
            add_table(document, [split_row(row) for k, row in enumerate(block) if k != 1], right)
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
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT if in_references else WD_ALIGN_PARAGRAPH.JUSTIFY
            if in_references:
                paragraph.paragraph_format.left_indent = Cm(0.9)
                paragraph.paragraph_format.first_line_indent = Cm(-0.9)
            add_inline(paragraph, line, BODY_PT - 1 if in_references else None)
    document.settings.element.find(qn("w:zoom")).set(qn("w:percent"), "100")
    document.save(TARGET)
    print(TARGET, TARGET.stat().st_size)


if __name__ == "__main__":
    convert()
