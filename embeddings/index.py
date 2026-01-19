import faiss
import numpy as np

from embeddings.embedder import embed, EMB_DIM

# def build_index(chunks):
#     vectors = embed(chunks)
#     index = faiss.IndexFlatL2(EMB_DIM)
#     index.add(vectors)
#     return index, chunks


def build_index(chunks):
    vectors = embed(chunks)

    # DEBUG: Check vectors before adding to index
    norms = np.linalg.norm(vectors, axis=1)
    print(f"DEBUG build_index - Vector norms: min={norms.min():.6f}, max={norms.max():.6f}, mean={norms.mean():.6f}")
    print(f"DEBUG build_index - Vector shape: {vectors.shape}")
    print(f"DEBUG build_index - Index type: IndexFlatIP")

    index = faiss.IndexFlatIP(EMB_DIM)
    index.add(vectors)

    print(f"DEBUG build_index - Vectors added to index: {index.ntotal}")

    return index, chunks