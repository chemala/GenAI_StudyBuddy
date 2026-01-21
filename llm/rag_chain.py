from tavily import TavilyClient
from config import TOP_K
from embeddings.embedder import embed
from llm.model import get_llm
from retrieval.web import extract_keywords, tavily_search, filter_relevant_web_results
from sentence_transformers.util import cos_sim
import numpy as np


def retrieve_relevant_chunks(question, index, chunks, top_k=TOP_K, gap_threshold=0.15, use_mmr=True, lambda_param=0.8):
    """
    Dynamic retrieval with optional MMR for diversity.
    Filter by gap threshold FIRST, then apply MMR to filtered candidates.
    """
    q_emb = embed([question])

    # Step 1: Retrieve more candidates if using MMR (need pool for diversity)
    #search_k = top_k * 3 if use_mmr else top_k
    #search_k = min(search_k, len(chunks))

    d, i = index.search(q_emb, top_k)

    top_score = d[0][0] if len(d[0]) > 0 else 0

    # Dynamic expansion for low scores
    # if top_score < 0.3:
    #     search_k = min(top_k, len(chunks))
    #     print(f"Low top score detected. Expanding search from {top_k * 3 if use_mmr else top_k} to {search_k} chunks")
    #     d, i = index.search(q_emb, search_k)
    #     top_score = d[0][0]

    print(f"Top score: {top_score:.3f}, Gap threshold: {gap_threshold}, MMR: {use_mmr}")

    # Step 2: Filter by gap threshold FIRST
    filtered_indices = []
    filtered_chunk_ids = []
    filtered_scores = []

    for idx, (ci, similarity) in enumerate(zip(i[0], d[0])):
        score_gap = top_score - similarity

        if score_gap <= gap_threshold:
            filtered_indices.append(idx)  # Index in search results
            filtered_chunk_ids.append(ci)  # Original chunk ID
            filtered_scores.append(similarity)
            print(f"Chunk {ci}: similarity = {similarity:.3f} (gap: {score_gap:.3f}) ✓")
        else:
            print(f"Chunk {ci}: similarity = {similarity:.3f} (gap: {score_gap:.3f}) ✗")

    if not filtered_indices:
        print("No chunks passed gap threshold")
        return [], [], 0.0

    print(f"Filtered to {len(filtered_indices)} chunks passing gap threshold")

    # Step 3: Apply MMR to filtered candidates (if enabled and we have enough)
    if use_mmr and len(filtered_indices) > top_k:
        print(f"Applying MMR to select {top_k} diverse chunks from {len(filtered_indices)} filtered candidates")

        # Get embeddings for FILTERED chunks only
        candidate_chunks = [chunks[ci] for ci in filtered_chunk_ids]
        candidate_embeddings = embed(candidate_chunks)

        # Get query embedding (already have it)
        q_emb_for_mmr = q_emb[0]

        # Apply MMR on filtered candidates
        mmr_indices = mmr(q_emb_for_mmr, candidate_embeddings, k=top_k, lambda_param=lambda_param)

        # Select final chunks based on MMR
        final_chunk_ids = [filtered_chunk_ids[mmr_idx] for mmr_idx in mmr_indices]
        final_scores = [filtered_scores[mmr_idx] for mmr_idx in mmr_indices]

        print(f"MMR selected chunks: {final_chunk_ids}")
    else:
        # Use all filtered chunks (already sorted by relevance from FAISS)
        final_chunk_ids = filtered_chunk_ids[:top_k]
        final_scores = filtered_scores[:top_k]
        print(f"Using top {len(final_chunk_ids)} filtered chunks (no MMR needed)")

    # Build final results
    relevant_chunks = [chunks[ci] for ci in final_chunk_ids]
    chunk_scores = final_scores

    avg_relevance = sum(chunk_scores) / len(chunk_scores) if chunk_scores else 0.0
    print(f"Average chunk relevance: {avg_relevance:.3f}")
    print(f"Retrieved {len(relevant_chunks)} final chunks")

    return relevant_chunks, chunk_scores, avg_relevance


def mmr(query_embedding, doc_embeddings, k=TOP_K, lambda_param=0.7):
    """
    Maximal Marginal Relevance for diverse chunk selection.
    """
    doc_embeddings = np.array(doc_embeddings)
    query_embedding = np.array(query_embedding).reshape(1, -1)

    sim_to_query = cos_sim(doc_embeddings, query_embedding).reshape(-1)
    selected = []
    candidates = list(range(len(doc_embeddings)))

    # Select first chunk (most relevant)
    selected.append(np.argmax(sim_to_query))
    candidates.remove(selected[0])

    # Select remaining k-1 chunks balancing relevance and diversity
    for _ in range(min(k - 1, len(candidates))):
        candidate_scores = []
        for c in candidates:
            sim_to_selected = max(cos_sim(doc_embeddings[c], doc_embeddings[s]) for s in selected)
            score = lambda_param * sim_to_query[c] - (1 - lambda_param) * sim_to_selected
            candidate_scores.append(score)

        selected_candidate = candidates[np.argmax(candidate_scores)]
        selected.append(selected_candidate)
        candidates.remove(selected_candidate)

    return selected


def reformulate_query_for_web(question, local_context, llm):
    """
    Use LLM to create a better web search query based on question + document context
    """
    reformulation_prompt = f"""Given this question and document context, create a specific web search query that would find relevant supplementary information.

Question: {question}

Document context (first 500 chars):
{local_context[:500]}

Generate ONLY a concise search query (5-10 words) that:
1. Includes key technical terms from the document
2. Focuses on finding current/additional information
3. Avoids vague phrases like "what is this about"

Search query:"""

    result = llm.chat_completion(
        messages=[{"role": "user", "content": reformulation_prompt}],
        max_tokens=50,
        temperature=0.3,
    )

    return result.choices[0].message.content.strip()


def build_local_prompt(local_context, question, allow_inference=False):
    """
    Local-only mode: Uses ONLY document context.
    """
    if allow_inference:
        prompt = f"""You are a helpful study assistant. Answer the question based on the document context, making reasonable inferences when needed.

INFERENCE MODE INSTRUCTIONS:
- Start with EXPLICIT information from the context
- Look for REPEATED concepts - repetition indicates core content
- Make REASONABLE INFERENCES based on:
  * Emphasis and space devoted to topics (more coverage = likely more important)
  * Order of presentation
  * Specific examples and detailed explanations given
  * Connections between concepts
- When making inferences:
  * Explain your reasoning
  * Cite specific evidence
  * Acknowledge if multiple interpretations are possible
- If even reasonable inference isn't possible, acknowledge the limitation

FORMATTING REQUIREMENTS:
- Use clear section headers with markdown (##)
- Break content into SHORT paragraphs (2-3 sentences max)
- Use bullet points (-) for lists
- Add blank lines between sections
- Use **bold** for key terms and concepts
- Keep answers well-structured and scannable

DOCUMENT CONTEXT:
{local_context}

QUESTION: {question}

Provide a well-formatted, easy-to-read answer:"""
    else:
        prompt = f"""You are a helpful study assistant. Answer the question based STRICTLY on the provided document context.

STRICT MODE INSTRUCTIONS:
- Only provide information that is EXPLICITLY stated in the context
- Do NOT make inferences, interpretations, or draw conclusions beyond what's directly written
- Do NOT speculate about relationships, importance, or rankings unless explicitly stated
- If the context doesn't directly answer the question, clearly state what IS available and what is MISSING
- Be precise and literal in your interpretation

FORMATTING REQUIREMENTS:
- Use clear section headers with markdown (##)
- Break content into SHORT paragraphs (2-3 sentences max)
- Use bullet points (-) for lists
- Add blank lines between sections
- Use **bold** for key terms and concepts
- Keep answers well-structured and scannable

DOCUMENT CONTEXT:
{local_context}

QUESTION: {question}

Provide a well-formatted, easy-to-read answer:"""

    return prompt


def build_web_prompt(local_context, web_context, question, allow_inference=False):
    """
    Web mode (formerly Hybrid): Combines document context with web results.
    """
    context_parts = []
    if local_context:
        context_parts.append(f"DOCUMENT CONTEXT (PRIMARY):\n{local_context}")
    if web_context:
        context_parts.append(f"\nADDITIONAL WEB SOURCES (SUPPLEMENTARY):\n{web_context}")

    context = "\n\n".join(context_parts)

    if allow_inference:
        prompt = f"""You are a helpful study assistant. Answer the question using both the document context and web sources, making reasonable inferences when needed.

INFERENCE MODE INSTRUCTIONS:
- Prioritize information from the DOCUMENT CONTEXT (PRIMARY SOURCE)
- Use WEB SOURCES to supplement, verify, or provide additional current information
- Make REASONABLE INFERENCES based on:
  * Emphasis and coverage in the document
  * Patterns and repeated concepts
  * Connections between document and web information
  * Consensus across multiple sources
- When making inferences:
  * Explain your reasoning
  * Cite evidence from both sources
  * Acknowledge different perspectives
- If sources conflict, analyze which is more authoritative or recent

FORMATTING REQUIREMENTS:
- Use clear section headers with markdown (##)
- Break content into SHORT paragraphs (2-3 sentences max)
- Use bullet points (-) for lists
- Add blank lines between sections
- Use **bold** for key terms and concepts
- Cite sources clearly (e.g., [From document] or [From web])
- Keep answers well-structured and scannable

{context}

QUESTION: {question}

Provide a well-formatted, easy-to-read answer:"""
    else:
        prompt = f"""You are a helpful study assistant. Answer the question using both the document context and web sources.

STRICT MODE INSTRUCTIONS:
- Prioritize information from the DOCUMENT CONTEXT (PRIMARY SOURCE)
- Use WEB SOURCES ONLY to supplement or verify information
- Only state what is EXPLICITLY mentioned in the sources
- Do NOT make inferences or draw conclusions beyond what's directly stated
- If sources conflict, present both views without choosing
- Acknowledge if neither source fully answers the question

FORMATTING REQUIREMENTS:
- Use clear section headers with markdown (##)
- Break content into SHORT paragraphs (2-3 sentences max)
- Use bullet points (-) for lists
- Add blank lines between sections
- Use **bold** for key terms and concepts
- Cite sources clearly (e.g., [From document] or [From web])
- Keep answers well-structured and scannable

{context}

QUESTION: {question}

Provide a well-formatted, easy-to-read answer:"""

    return prompt


def rag_with_llm(
        question,
        mode,
        index,
        chunks,
        tavily_api_key,
        chat_history,
        top_k=TOP_K,
        allow_inference=False,
):
    """
    Unified RAG function with dynamic retrieval + MMR.

    - Dynamic retrieval with gap threshold
    - MMR for diversity (applied after filtering)
    - Local / Web modes
    - Validation & error handling
    - Safe execution (no crashes on empty index, chunks, or web failure)
    """

    # ======================= VALIDATION & SAFETY =======================

    if not chunks or len(chunks) == 0:
        error_msg = "No documents have been indexed yet. Please upload documents and build the index."
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": error_msg})
        return error_msg, chat_history

    if index is not None and getattr(index, "ntotal", 0) == 0:
        error_msg = "The index is empty. Please rebuild the index with valid documents."
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": error_msg})
        return error_msg, chat_history

    # ==================================================================

    llm = get_llm()

    # ======================= RETRIEVAL WITH DYNAMIC + MMR =======================

    try:
        retrieved_chunks, chunk_scores, avg_relevance = retrieve_relevant_chunks(
            question,
            index,
            chunks,
            top_k=top_k,
            gap_threshold=0.1,
            use_mmr=True,
            lambda_param=0.8
        )
    except Exception as e:
        error_msg = f"Retrieval failed: {e}"
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": error_msg})
        return error_msg, chat_history

    if not retrieved_chunks:
        no_context_msg = (
            "I couldn't find relevant information in the uploaded documents "
            "to answer your question. Try rephrasing or check if the document contains this topic."
        )
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": no_context_msg})
        return no_context_msg, chat_history

    local_context = "\n\n---\n\n".join(retrieved_chunks)

    # ======================= MODE HANDLING =======================

    if mode == "local":
        prompt = build_local_prompt(local_context, question, allow_inference)
        mode_display = f"LOCAL ({'INFERENCE' if allow_inference else 'STRICT'})"

    elif mode == "web":
        # Reformulate query using document context
        try:
            search_query = reformulate_query_for_web(question, local_context, llm)
            print(f"Original question: {question}")
            print(f"Reformulated search: {search_query}")
        except Exception as e:
            print(f"Query reformulation failed: {e}")
            search_query = question

        # Web search (safe)
        web_context = ""
        try:
            q_emb = embed([question])[0]
            web_results = tavily_search(search_query, tavily_api_key, k=5)
            web_results = filter_relevant_web_results(
                q_emb.flatten(),
                web_results,
                top_n=3,
            )
            web_context = "\n".join(web_results) if web_results else ""
        except Exception as e:
            print(f"Web search failed: {e}")
            web_context = ""

        prompt = build_web_prompt(
            local_context,
            web_context,
            question,
            allow_inference
        )
        mode_display = f"WEB ({'INFERENCE' if allow_inference else 'STRICT'})"

    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'local' or 'web'")

    # ======================= DEBUG OUTPUT =======================

    print(f"\n{'=' * 60}")
    print(f"MODE: {mode_display}")
    print(f"Retrieved {len(retrieved_chunks)} chunks (avg similarity: {avg_relevance:.3f})")
    print(f"{'=' * 60}")
    print(prompt)
    print(f"{'=' * 60}\n")

    # ======================= LLM CALL =======================

    try:
        result = llm.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful study assistant that answers questions "
                        "accurately and strictly based on the provided context."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=800,
            temperature=0.2 if not allow_inference else 0.3,
        )

        answer = result.choices[0].message.content
        print(answer)
    except Exception as e:
        error_msg = f"LLM generation failed: {e}"
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": error_msg})
        return error_msg, chat_history

    # ======================= CHAT HISTORY =======================

    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": answer})

    return answer, chat_history