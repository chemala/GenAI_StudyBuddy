from sentence_transformers import SentenceTransformer
import numpy as np

embedder = SentenceTransformer("all-mpnet-base-v2")
EMB_DIM = embedder.get_sentence_embedding_dimension()

def embed(texts):
    print(f"EMB_DIM = {EMB_DIM}")
    return embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype("float32")

# def embed(texts):
#     vecs = embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
#
#     # DEBUG: Check norms
#     norms = np.linalg.norm(vecs, axis=1)
#     print(
#         f"DEBUG - Vector norms (should be ~1.0): min={norms.min():.3f}, max={norms.max():.3f}, mean={norms.mean():.3f}")
#
#     return vecs