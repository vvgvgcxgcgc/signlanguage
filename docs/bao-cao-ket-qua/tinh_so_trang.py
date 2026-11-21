"""Tính số trang của từng đề mục để điền vào mục lục.

Dàn trang bao-cao-ket-qua.docx bằng Pages (xuất PDF), tìm trang chứa từng đề mục,
ghi so-trang.json. Sau đó chạy lại xuat_docx.py để mục lục có số trang.

    .venv/bin/python docs/bao-cao-ket-qua/tinh_so_trang.py   # cần macOS có Pages và pypdf
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
DOCX = HERE / "bao-cao-ket-qua.docx"
SOURCE = HERE / "bao-cao-ket-qua.md"
OUT = HERE / "so-trang.json"


def to_pdf(docx: Path, pdf: Path) -> None:
    script = f'''
    tell application "Pages"
        set d to open POSIX file "{docx}"
        delay 2
        export d to POSIX file "{pdf}" as PDF
        close d saving no
    end tell
    '''
    subprocess.run(["osascript", "-e", script], check=True, timeout=300)


def squash(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def main() -> None:
    headings = [m[2] for m in re.finditer(r"^(#{2,4}) (.+)$", SOURCE.read_text(encoding="utf8"), re.M)]
    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "bao-cao.pdf"
        to_pdf(DOCX, pdf)
        pages = [squash(page.extract_text() or "") for page in PdfReader(str(pdf)).pages]
    # Bỏ qua bìa và mục lục: đề mục đầu tiên xuất hiện lần thứ hai là nơi nội dung bắt đầu.
    first = squash(headings[0])
    start = [i for i, text in enumerate(pages) if first in text]
    page = start[1] if len(start) > 1 else start[0]
    result, missing = {}, []
    for heading in headings:
        key = squash(heading)[:40]
        found = next((i for i in range(page, len(pages)) if key in pages[i]), None)
        if found is None:
            missing.append(heading)
            continue
        page = found
        result[heading] = found + 1   # số trang = thứ tự trang trong tệp, bìa là trang 1
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf8")
    print(f"{len(result)}/{len(headings)} đề mục, {len(pages)} trang -> {OUT}")
    if missing:
        print("không tìm thấy:", missing, file=sys.stderr)


if __name__ == "__main__":
    main()
