"""Stage 1-2: resume upload and text extraction for TXT, DOCX and PDF (no OCR; see Future Work)."""
from __future__ import annotations
from pathlib import Path


def read_txt(path) -> str:
    raw = Path(path).read_bytes()
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def read_docx(path) -> str:
    import docx
    d = docx.Document(str(path))
    lines = [p.text for p in d.paragraphs]
    for t in d.tables:                                   # tables are common in two-column templates
        for row in t.rows:
            for cell in row.cells:
                lines.extend(cell.text.split("\n"))
    return "\n".join(lines)


def read_pdf(path) -> str:
    try:
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            pages = [(pg.extract_text() or "") for pg in pdf.pages]
        text = "\n".join(pages)
        if text.strip():
            return text
    except Exception:
        pass
    from pypdf import PdfReader                          # fallback extractor
    return "\n".join((pg.extract_text() or "") for pg in PdfReader(str(path)).pages)


def parse_file(path) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".txt":
        return read_txt(path)
    if ext == ".docx":
        return read_docx(path)
    if ext == ".pdf":
        return read_pdf(path)
    raise ValueError(f"Unsupported resume format: {ext} (use .txt, .docx or .pdf)")


# ---------------------------------------------------------------------------------------------
# helpers to create test files in the three formats (used by the format-robustness experiment)
# ---------------------------------------------------------------------------------------------
def write_txt(text: str, path):
    Path(path).write_text(text, encoding="utf-8")


def write_docx(text: str, path):
    import docx
    d = docx.Document()
    for line in text.split("\n"):
        d.add_paragraph(line)
    d.save(str(path))


def write_pdf(text: str, path):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import simpleSplit
    c = canvas.Canvas(str(path), pagesize=A4)
    w, h = A4
    margin, y = 56, h - 56
    c.setFont("Helvetica", 10)
    for line in text.split("\n"):
        if line.strip() == "":
            y -= 7
        else:
            # '•' is not in the base-14 font set: map to a plain bullet glyph that Helvetica supports
            safe = line.replace("•", "-")
            for piece in simpleSplit(safe, "Helvetica", 10, w - 2 * margin):
                if y < margin:
                    c.showPage()
                    c.setFont("Helvetica", 10)
                    y = h - margin
                c.drawString(margin, y, piece)
                y -= 13
    c.save()
