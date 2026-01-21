from retrieval.local import retrieve_local
from retrieval.web import tavily_search, extract_keywords

def rag(query, mode, index, chunks, api_key):
    context = []

    local_chunks = []
    if mode in ["local", "hybrid"]:
        local_chunks = retrieve_local(query, index, chunks)
        context += local_chunks

    if mode in ["web", "hybrid"] and api_key:
        local_context = "\n".join(local_chunks) if local_chunks else ""
        if local_context:
            keywords = extract_keywords(local_context)
            enhanced_query = query + " " + " ".join(keywords)
        else:
            enhanced_query = query
        context += tavily_search(enhanced_query, api_key)

    return "\n\n---\n\n".join(context)