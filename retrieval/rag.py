from retrieval.local import retrieve_local
from retrieval.web import tavily_search

def rag(query, mode, index, chunks, api_key):
    context = []

    if mode in ["local", "hybrid"]:
        context += retrieve_local(query, index, chunks)

    if mode in ["web", "hybrid"] and api_key:
        context += tavily_search(query, api_key)

    return "\n\n---\n\n".join(context)