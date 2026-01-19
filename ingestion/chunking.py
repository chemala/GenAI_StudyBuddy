from config import CHUNK_SIZE, CHUNK_OVERLAP

import re
from typing import List


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text into overlapping chunks for better retrieval.

    Args:
        text: The text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Number of characters to overlap between chunks

    Returns:
        list: List of text chunks
    """
    # FIX: Handle empty or very short text
    if not text or len(text.strip()) == 0:
        print("⚠️ WARNING: Empty text provided to chunking function")
        return []

    # FIX: If text is smaller than chunk size, return it as a single chunk
    if len(text) <= chunk_size:
        print(f"ℹ️ Text ({len(text)} chars) is smaller than chunk size. Returning as single chunk.")
        return [text.strip()]

    chunks = []
    start = 0

    # OLD CODE (worked but no validation):
    # while start < len(text):
    #     end = start + chunk_size
    #     chunks.append(text[start:end])
    #     start = end - overlap

    # NEW CODE: Same logic but with filtering of empty chunks
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()  # FIX: Strip whitespace from chunks

        # FIX: Only add chunks with meaningful content (at least 10 characters)
        if len(chunk) >= 10:
            chunks.append(chunk)

        start = end - overlap

    print(f"✓ Created {len(chunks)} chunks from {len(text)} characters")

    return chunks


def chunk_text_smart(text: str, chunk_size: int = 1200, overlap: int = 240) -> List[str]:
    """
    Smart chunking that combines multiple strategies
    - Preserves page boundaries when possible
    - Uses semantic breaks (headers, bullet points)
    - Falls back to sentence-based chunking

    RECOMMENDED for most use cases
    """
    chunks = []

    # Split by pages
    page_pattern = r'\[PAGE (\d+)\]'
    pages = re.split(page_pattern, text)

    # Process each page
    for i in range(1, len(pages), 2):
        if i >= len(pages):
            break

        page_num = pages[i]
        page_content = pages[i + 1] if i + 1 < len(pages) else ""

        if not page_content.strip():
            continue

        # If page is small enough, keep it as one chunk
        if len(page_content) <= chunk_size:
            chunks.append(f"[PAGE {page_num}]\n{page_content.strip()}")
        else:
            # Split large pages by sections
            # Look for section markers (bullet points, headers, etc.)
            section_pattern = r'\n(?=■|\d+\.|[A-Z][a-z]+:|\n[A-Z])'
            sections = re.split(section_pattern, page_content)

            current_chunk = f"[PAGE {page_num}]\n"
            current_length = len(current_chunk)

            for section in sections:
                section = section.strip()
                if not section:
                    continue

                # If adding section would exceed size, save and start new chunk
                if current_length + len(section) > chunk_size and current_length > overlap:
                    chunks.append(current_chunk.strip())
                    # Keep page reference and some overlap
                    current_chunk = f"[PAGE {page_num}] (continued)\n"
                    current_length = len(current_chunk)

                current_chunk += section + "\n"
                current_length += len(section)

            if current_chunk.strip():
                chunks.append(current_chunk.strip())

    return [c for c in chunks if c.strip()]


"""
TOP_K = 10  # Increase from 5 to get more context
CHUNK_SIZE = 1000  # Larger chunks = more context per retrieval
CHUNK_OVERLAP = 150  # Overlap helps maintain continuity
"""


# Updated usage in your system:
def chunk_text(text: str) -> List[str]:
    """
    Main chunking function - uses smart chunking strategy

    Adjust these parameters based on your needs:
    - chunk_size: 500-1500 (larger = more context but less precision)
    - overlap: 100-200 (helps maintain context across chunks)
    """
    return chunk_text_smart(
        text,
        chunk_size=CHUNK_SIZE,  # INCREASED from typical 500
        overlap=CHUNK_OVERLAP
    )
