import pdfplumber

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
        try:
            return extract_pdf_with_pdfplumber(path)
        except Exception as e:
            print(f"pdfplumber failed: {e}")

    if path.lower().endswith(".txt"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    raise ValueError("Unsupported file type")