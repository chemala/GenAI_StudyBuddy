import numpy as np
from embeddings.embedder import embed
from sentence_transformers.util import cos_sim
from config import TOP_K

def mmr(query_embedding, doc_embeddings, k=TOP_K, lambda_param=0.7):
    doc_embeddings = np.array(doc_embeddings)
    query_embedding = np.array(query_embedding).reshape(1, -1)

    sim_to_query = cos_sim(doc_embeddings, query_embedding).reshape(-1)
    selected = []
    candidates = list(range(len(doc_embeddings)))

    selected.append(np.argmax(sim_to_query))
    candidates.remove(selected[0])

    for _ in range(k - 1):
        candidate_scores = []
        for c in candidates:
            sim_to_selected = max(cos_sim(doc_embeddings[c], doc_embeddings[s]) for s in selected)
            score = lambda_param * sim_to_query[c] - (1 - lambda_param) * sim_to_selected
            candidate_scores.append(score)

        selected_candidate = candidates[np.argmax(candidate_scores)]
        selected.append(selected_candidate)
        candidates.remove(selected_candidate)

    return selected

def retrieve_local(query, index, chunks, k=5):
    q_emb = embed([query])[0]
    # Get more candidates for MMR
    num_candidates = min(k * 4, len(chunks))  # up to 4x or total chunks
    D, I = index.search(q_emb.reshape(1, -1), num_candidates)
    candidate_indices = I[0]
    # Embed candidate chunks
    candidate_texts = [chunks[i] for i in candidate_indices]
    candidate_embs = embed(candidate_texts)
    # Select with MMR
    selected = mmr(q_emb, candidate_embs, k=k)
    return [chunks[candidate_indices[i]] for i in selected]
