import faiss
from embeddings.embedder import embed, EMB_DIM

def build_index(chunks):
    vectors = embed(chunks)
    index = faiss.IndexFlatL2(EMB_DIM)
    index.add(vectors)
    return index, chunks