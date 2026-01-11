from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-mpnet-base-v2")
EMB_DIM = embedder.get_sentence_embedding_dimension()

def embed(texts):
    return embedder.encode(texts, convert_to_numpy=True).astype("float32")