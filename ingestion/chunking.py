from config import CHUNK_SIZE, CHUNK_OVERLAP


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