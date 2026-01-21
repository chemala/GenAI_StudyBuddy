import io
import pymupdf
from PIL import Image
import pdfplumber
from pytesseract import pytesseract


def extract_pdf_with_ocr(path):
    pages = []
    doc = pymupdf.open(path)  # Use pymupdf for OCR approach

    for i, page in enumerate(doc):
        # Try regular text extraction first
        text = page.get_text()

        # If text is garbled or empty, use OCR
        if not text.strip() or has_garbled_text(text):
            # Render page as image
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes()))

            # Extract text with OCR
            text = pytesseract.image_to_string(img)

        if text.strip():
            pages.append(f"[PAGE {i + 1}]\n{text.strip()}")

    doc.close()
    return "\n\n".join(pages)


def extract_pdf_with_pdfplumber(path):
    """Alternative method using pdfplumber"""
    pages = []

    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append(f"[PAGE {i + 1}]\n{text.strip()}")

    return "\n\n".join(pages)


def has_garbled_text(text):
    # Check if text contains too many non-printable or unusual characters
    weird_chars = sum(1 for c in text if ord(c) > 127 or c in '�✓')
    return weird_chars / max(len(text), 1) > 0.3


def extract_text_or_pdf(path):
    if path.lower().endswith(".pdf"):
        # Try pdfplumber first (simpler, no OCR needed)
        try:
            return extract_pdf_with_pdfplumber(path)
        except Exception as e:
            print(f"pdfplumber failed: {e}, trying OCR...")
            # Fall back to OCR if pdfplumber fails
            return extract_pdf_with_ocr(path)

    if path.lower().endswith(".txt"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    raise ValueError("Unsupported file type")