from config import CHUNK_SIZE, CHUNK_OVERLAP

import re
from typing import List


def chunk_text_simple(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Simple character-based chunking with overlap

    Args:
        text: Input text
        chunk_size: Target size of each chunk in characters
        overlap: Number of overlapping characters between chunks

    Returns:
        List of text chunks
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at sentence boundaries if possible
        if end < len(text):
            # Look for sentence endings near the chunk boundary
            chunk_text = text[start:end]
            last_period = chunk_text.rfind('.')
            last_newline = chunk_text.rfind('\n')

            # Break at the last sentence or paragraph if found
            break_point = max(last_period, last_newline)
            if break_point > chunk_size * 0.5:  # Only if we're past halfway
                end = start + break_point + 1

        chunks.append(text[start:end].strip())
        start = end - overlap

    return [c for c in chunks if c]  # Remove empty chunks


def chunk_text_semantic(text: str, max_chunk_size: int = 800, min_chunk_size: int = 200) -> List[str]:
    """
    Semantic chunking that preserves slide/section boundaries
    Better for structured documents like PDFs with clear sections

    Args:
        text: Input text
        max_chunk_size: Maximum chunk size
        min_chunk_size: Minimum chunk size

    Returns:
        List of text chunks
    """
    # Split by page markers first
    pages = re.split(r'\[PAGE \d+\]', text)

    chunks = []
    current_chunk = ""

    for page in pages:
        page = page.strip()
        if not page:
            continue

        # Split page into sections (based on headers or bullet points)
        sections = re.split(r'\n■\s+|\n#+\s+', page)

        for section in sections:
            section = section.strip()
            if not section:
                continue

            # If adding this section would exceed max size, save current chunk
            if len(current_chunk) + len(section) > max_chunk_size and len(current_chunk) >= min_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = section
            else:
                # Add to current chunk
                current_chunk += "\n" + section if current_chunk else section

    # Add final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def chunk_text_sliding_window(text: str, chunk_size: int = 1200, overlap: int = 240) -> List[str]:
    """
    Sliding window chunking with larger chunks for better context
    Recommended for RAG systems

    Args:
        text: Input text
        chunk_size: Size of each chunk
        overlap: Overlap between consecutive chunks

    Returns:
        List of text chunks
    """
    chunks = []

    # Split by sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)

    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        if current_length + sentence_length > chunk_size and current_chunk:
            # Save current chunk
            chunks.append(' '.join(current_chunk))

            # Start new chunk with overlap
            # Keep last few sentences for context
            overlap_sentences = []
            overlap_length = 0

            for sent in reversed(current_chunk):
                if overlap_length + len(sent) <= overlap:
                    overlap_sentences.insert(0, sent)
                    overlap_length += len(sent)
                else:
                    break

            current_chunk = overlap_sentences
            current_length = overlap_length

        current_chunk.append(sentence)
        current_length += sentence_length

    # Add final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

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
