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