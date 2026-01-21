import faiss
import numpy as np
from embeddings.embedder import embed, EMB_DIM

def build_index(chunks):
    """
    Build a FAISS index from text chunks.
    
    Args:
        chunks: List of text chunks to index
        
    Returns:
        tuple: (FAISS index, chunks array)
    """
    # FIX: Validate that we have chunks to index
    if not chunks or len(chunks) == 0:
        print("❌ ERROR: Cannot build index with empty chunks")
        raise ValueError("No chunks provided to build index. Please ensure PDFs are properly loaded and chunked.")
    
    print(f"🔨 Building index for {len(chunks)} chunks...")
    
    # FIX: Add try-except for embedding failures
    try:
        # Create embeddings for all chunks
        vectors = embed(chunks)
        
        # FIX: Validate embedding dimensions
        if vectors.shape[0] != len(chunks):
            raise ValueError(f"Embedding count mismatch: got {vectors.shape[0]} embeddings for {len(chunks)} chunks")
        
        if vectors.shape[1] != EMB_DIM:
            raise ValueError(f"Embedding dimension mismatch: got {vectors.shape[1]}, expected {EMB_DIM}")
        
        print(f"   ✓ Created embeddings: shape={vectors.shape}, dtype={vectors.dtype}")
        
    except Exception as e:
        print(f"❌ ERROR: Failed to create embeddings: {e}")
        raise ValueError(f"Could not embed chunks: {e}")
    
    # OLD CODE (worked but no validation):
    # index = faiss.IndexFlatL2(EMB_DIM)
    # index.add(vectors)
    # return index, chunks
    
    # NEW CODE: Same logic but with validation
    try:
        # # Create FAISS index using L2 (Euclidean) distance
        # index = faiss.IndexFlatL2(EMB_DIM)
        index = faiss.IndexFlatIP(EMB_DIM)

        # Add vectors to index
        index.add(vectors)
        
        # FIX: Verify index was built correctly
        if index.ntotal != len(chunks):
            raise ValueError(f"Index build failed: expected {len(chunks)} vectors, got {index.ntotal}")
        
        print(f"   ✓ FAISS index built successfully with {index.ntotal} vectors")
        
    except Exception as e:
        print(f"❌ ERROR: Failed to build FAISS index: {e}")
        raise ValueError(f"Could not build search index: {e}")
    
    return index, chunks