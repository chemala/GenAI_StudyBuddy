import pymupdf


def extract_pdf_text_from_path(path):
    pages = []
    doc = pymupdf.open(path)

    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append(f"[PAGE {i + 1}]\n{text.strip()}")

    doc.close()
    return "\n\n".join(pages)

def extract_text_or_pdf(path):
    if path.lower().endswith(".pdf"):
        return extract_pdf_text_from_path(path)

    if path.lower().endswith(".txt"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    raise ValueError("Unsupported file type")