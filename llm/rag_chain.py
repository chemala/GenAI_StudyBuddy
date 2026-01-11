from tavily import TavilyClient
from config import TOP_K
from embeddings.embedder import embedder
from llm.model import get_llm


def rag_with_llm(question, mode, index, chunks, tavily_api_key, chat_history, top_k=TOP_K):
    # Embed question
    q_emb = embedder.encode(
        [question],
        normalize_embeddings=True
    ).astype("float32")

    # Retrieve from FAISS
    D, I = index.search(q_emb, top_k)
    local_context = "\n".join(chunks[i] for i in I[0])

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
Based *strictly* on the following context and previous conversation, formulate a helpful response to the query. Provide information or tips *only as directly relevant* to the question and found within the context. Do not ask clarifying questions, introduce new topics, or discuss linguistic nuances unless explicitly asked about them in the question.

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
        max_tokens=600,
        temperature=0.3,
    )

    answer = result.choices[0].message.content
    print(answer)

    # Update chat history - append QUESTION and ANSWER
    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": answer})

    return answer, chat_history
