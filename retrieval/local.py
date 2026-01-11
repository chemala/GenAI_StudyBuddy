from embeddings.embedder import embed

def retrieve_local(query, index, chunks, k=5):
    qv = embed([query])
    _, I = index.search(qv, k)
    return [chunks[i] for i in I[0]]