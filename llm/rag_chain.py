from tavily import TavilyClient
from config import TOP_K
from embeddings.embedder import embed
from llm.model import get_llm

#SIMILARITY_THRESHOLD = 0.22 #  (0.2 - permissive | 0.4 - strict)

def retrieve_relevant_chunks(question, index, chunks, top_k=TOP_K, gap_threshold=0.1):

    q_emb = embed([question])
    d, i = index.search(q_emb, top_k)   # vector database

    # DEBUG
    print(f"DEBUG - Raw distances: {d[0]}")
    print(f"DEBUG - Max possible cosine sim should be ~1.0")

    # Filtering chunks by relevance
    relevant_chunks = []
    chunk_scores = []

    top_score = d[0][0] if len(d[0]) > 0 else 0
    print(f"Top score: {top_score:.3f}, Gap threshold: {gap_threshold}")

    for ci, similarity in zip(i[0], d[0]):
        score_gap = top_score - similarity
        if score_gap <= gap_threshold:
            relevant_chunks.append(chunks[ci])
            chunk_scores.append(similarity)

        print(f"Chunk {ci}: similarity = {similarity:.3f} (gap: {score_gap:.3f})")

    avg_relevance = sum(chunk_scores) / len(chunk_scores) if chunk_scores else 0.0
    print(f"Average chunk relevance: {avg_relevance:.3f}")

    # Sort relevant chunks, more relevant first
    sorted_chunks_scores = sorted(zip(relevant_chunks, chunk_scores), key=lambda x: x[1], reverse=True)
    relevant_chunks = [chunk for chunk, score in sorted_chunks_scores]

    return relevant_chunks, chunk_scores, avg_relevance


def rag_with_llm(question, mode, index, chunks, tavily_api_key, chat_history, top_k=TOP_K):

    relevant_chunks, chunk_scores, avg_relevance = retrieve_relevant_chunks(
        question, index, chunks, top_k
    )

    if not relevant_chunks:
        return ("I don't have enough relevant information in the lecture notes to answer this question. "
                "Could you rephrase or ask something more specific?"), chat_history

    local_context = "\n".join(relevant_chunks)

    # Optional web search
    web_context = ""
    if mode in ["web", "hybrid"]:
        tavily = TavilyClient(api_key=tavily_api_key)
        web = tavily.search(question, max_results=3)
        web_context = "\n".join(r["content"] for r in web['results'])

    # Build prompt
    context = local_context
    if web_context:
        context += "\n\nWeb context:\n" + web_context

    # Include chat history in the prompt (FIXED format)
    history_str = ""
    if chat_history:
        history_str = "\n\nPrevious conversation:\n" + "\n".join(
            [f"{h['role'].capitalize()}: {h['content']}" for h in chat_history]
        )

    prompt = f"""<s>[INST]
Based *strictly* on the following context and previous conversation, formulate a helpful response to the query. 
Provide information or tips *only as directly relevant* to the question and found within the context. 
Do not ask clarifying questions, introduce new topics, or discuss linguistic nuances unless explicitly asked 
about them in the question.

Context:
{context}
{history_str}

Question:
{question}
[/INST]
"""

    # Generate answer using lazy-loaded LLM
    llm = get_llm()
    result = llm.chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful study assistant that creates clear explanations and flashcards."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=600, # todo: can be in config / UI
        temperature=0.3, # todo: can be in config / UI
    )

    answer = result.choices[0].message.content

    if avg_relevance < 0.3:
        answer = f"[Low confidence - limited relevant information found]\n\n{answer}"

    print(f"Retrieved {len(relevant_chunks)} relevant chunks (avg similarity: {avg_relevance:.3f})")
    print(answer)

    # Update chat history - append QUESTION and ANSWER
    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": answer})

    return answer, chat_history
