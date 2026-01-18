from tavily import TavilyClient
import numpy as np

def extract_keywords(text, top_n=5):
    words = text.lower().split()
    word_count = {}
    stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"}  # basic stopwords
    for word in words:
        word = word.strip(".,!?;:").lower()
        if len(word) > 3 and word not in stopwords:
            word_count[word] = word_count.get(word, 0) + 1
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    return [w for w, c in sorted_words[:top_n]]

def filter_relevant_web_results(query_embedding, web_results, embedder, top_n=3):
    if not web_results:
        return []
    web_embeddings = embedder.encode(web_results, normalize_embeddings=True).astype("float32")
    similarities = np.dot(web_embeddings, query_embedding.T).flatten()
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    return [web_results[i] for i in top_indices]

def tavily_search(query, api_key, k=3):
    client = TavilyClient(api_key=api_key)
    res = client.search(query, limit=k)
    return [r.get("content", "") for r in res["results"]]
