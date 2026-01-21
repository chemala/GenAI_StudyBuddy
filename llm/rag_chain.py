from tavily import TavilyClient
from config import TOP_K
from embeddings.embedder import embedder
from llm.model import get_llm
from retrieval.web import extract_keywords, tavily_search, filter_relevant_web_results
from sentence_transformers.util import cos_sim
import numpy as np


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

def build_strict_prompt(local_context, web_context, question):
    """
    Strict prompt: Only answers with explicitly stated information.
    Best for factual verification and direct information retrieval.
    """
    context_parts = []
    if local_context:
        context_parts.append(f"DOCUMENT CONTEXT:\n{local_context}")
    if web_context:
        context_parts.append(f"\nADDITIONAL WEB SOURCES:\n{web_context}")

    context = "\n\n".join(context_parts)

    prompt = f"""You are a helpful study assistant. Answer the question based STRICTLY on the provided context.

STRICT MODE INSTRUCTIONS:
- Only provide information that is EXPLICITLY stated in the context
- Do NOT make inferences, interpretations, or draw conclusions beyond what's directly written
- Do NOT speculate about relationships, importance, or rankings unless explicitly stated
- If the context doesn't directly answer the question, clearly state what IS available and what is MISSING
- Be precise and literal in your interpretation

{context}

QUESTION: {question}

ANSWER:"""

    return prompt


def build_inference_prompt(local_context, web_context, question):
    """
    Inference prompt: Allows reasonable conclusions based on emphasis, patterns, and context.
    Best for analytical questions and deeper understanding.
    """
    context_parts = []
    if local_context:
        context_parts.append(f"DOCUMENT CONTEXT:\n{local_context}")
    if web_context:
        context_parts.append(f"\nADDITIONAL WEB SOURCES:\n{web_context}")

    context = "\n\n".join(context_parts)

    prompt = f"""You are a helpful study assistant. Answer the question based on the provided context, using both explicit information and reasonable inferences.

INFERENCE MODE INSTRUCTIONS:
- Start with EXPLICIT information from the context
- Look for REPEATED concepts across multiple excerpts - that indicates core content
- When explicit information is insufficient, make REASONABLE INFERENCES based on:
  * Emphasis and space devoted to topics (more coverage = likely more important)
  * Order of presentation (items listed first may indicate priority)
  * Specific examples and detailed explanations given
  * Causal relationships and connections described
  * Context from surrounding material
  * Repeated themes or concepts

- When making inferences:
  * Explain your reasoning
  * Cite specific evidence (slide numbers, amount of coverage, positioning)
  * Acknowledge if multiple interpretations are possible
  * Don't overreach - stay grounded in what's actually there

- If even reasonable inference isn't possible, acknowledge the limitation

{context}

QUESTION: {question}

ANSWER:"""

    return prompt



def rag_with_llm(question, mode, index, chunks, tavily_api_key, chat_history,
                 top_k=TOP_K, allow_inference=False):
    """
    Enhanced RAG with strict vs inference mode.

    Args:
        allow_inference: If True, uses inference prompt. If False, uses strict prompt.
    """
    # Compute embeddings for query and chunks
    q_emb = embedder.encode([question], normalize_embeddings=True).astype("float32")[0]
    chunk_embs = embedder.encode(chunks, normalize_embeddings=True).astype("float32")

    # Use MMR to select diverse but relevant chunks
    selected = mmr(q_emb, chunk_embs, k=top_k)
    local_context = "\n".join(chunks[i] for i in selected)

    # Smarter web search logic
    web_context = ""
    if mode in ["web", "hybrid"]:
        if should_use_web_search(question, local_context):
            web_results = tavily_search(question, tavily_api_key, k=5)
            web_results = filter_relevant_web_results(
                q_emb.flatten(),
                web_results,
                embedder,
                top_n=3,
            )
            if web_results:
                web_context = "\n".join(web_results)

    # Choose prompt based on mode
    if allow_inference:
        prompt = build_inference_prompt(local_context, web_context, question)
    else:
        prompt = build_strict_prompt(local_context, web_context, question)

    print(f"\n{'=' * 60}")
    print(f"MODE: {'INFERENCE' if allow_inference else 'STRICT'}")
    print(f"{'=' * 60}")
    print(prompt)
    print(f"{'=' * 60}\n")

    # Generate answer
    llm = get_llm()
    result = llm.chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful study assistant that answers questions accurately based on provided context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=800,
        temperature=0.2,
    )

    answer = result.choices[0].message.content
    print(answer)

    # Update chat history
    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": answer})

    return answer, chat_history



def rag_with_llm_aykut(question, mode, index, chunks, tavily_api_key, chat_history, top_k=TOP_K):
    # ============== NEW: Detect document summary queries and adjust retrieval ==============
    # For broad queries like "What does this PDF talk about?", we need MORE chunks
    # to get better topic coverage and avoid over-emphasizing minor sections
    is_summary_query = None  # is_document_summary_query(question)

    if is_summary_query:
        # OLD APPROACH (used fixed top_k=5):
        # This retrieved too few chunks, causing topic bias toward semantically rich sections

        # NEW APPROACH: Retrieve more chunks for better topic coverage
        # Use 20 chunks (or all available if less) to capture dominant themes
        adaptive_top_k = min(20, len(chunks))
        print(f"\n{'=' * 60}")
        print(f"📋 DOCUMENT SUMMARY QUERY DETECTED")
        print(f"   Using {adaptive_top_k} chunks (instead of {top_k}) for better topic coverage")
        print(f"   This helps identify DOMINANT vs MINOR themes")
        print(f"{'=' * 60}\n")
    else:
        # For specific questions, use normal top_k
        adaptive_top_k = top_k
    # =======================================================================================

    # Embed question
    q_emb = embedder.encode(
        [question],
        normalize_embeddings=True
    ).astype("float32")

    # ============== FIX: Added validation for index and chunks ==============
    # Check if we have chunks to search
    if not chunks or len(chunks) == 0:
        # CRITICAL: If no chunks available, return error message instead of crashing
        error_msg = "⚠️ No documents have been indexed yet. Please upload and build the index first."
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": error_msg})
        return error_msg, chat_history

    # Check if index has been built with content
    if index.ntotal == 0:
        # CRITICAL: If index is empty, we can't retrieve anything
        error_msg = "⚠️ The index is empty. Please re-upload your documents and rebuild the index."
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": error_msg})
        return error_msg, chat_history
    # =========================================================================

    # Retrieve from FAISS - vector database
    # NEW: Use adaptive_top_k instead of fixed top_k
    D, I = index.search(q_emb, adaptive_top_k)

    # ============== FIX: Validate retrieved indices before accessing chunks ==============
    # OLD CODE (BUGGY - could access invalid indices):
    # local_context = "\n".join(chunks[i] for i in I[0])

    # NEW CODE: Filter out invalid indices (FAISS returns -1 for no results)
    # and ensure indices are within bounds of chunks array
    valid_indices = []
    for idx in I[0]:
        # Check: index must be >= 0 and < total number of chunks
        if 0 <= idx < len(chunks):
            valid_indices.append(idx)

    # Build context from valid retrieved chunks
    retrieved_chunks = [chunks[i] for i in valid_indices]
    local_context = "\n\n---\n\n".join(retrieved_chunks)  # Improved separator for readability

    # Debug logging: Show what was retrieved
    print(f"\n{'=' * 60}")
    print(f"RETRIEVAL DEBUG INFO:")
    print(f"Question: {question}")
    print(f"Retrieved {len(valid_indices)} chunks from index")
    print(f"Chunk indices: {valid_indices}")
    print(f"First chunk preview: {retrieved_chunks[0][:200] if retrieved_chunks else 'NONE'}...")
    print(f"{'=' * 60}\n")
    # =====================================================================================

    # Optional web search
    web_context = ""
    if mode in ["web", "hybrid"]:
        # FIX: Add try-except for web search failures
        try:
            tavily = TavilyClient(api_key=tavily_api_key)
            web = tavily.search(question, max_results=3)
            web_context = "\n".join(r["content"] for r in web['results'])
        except Exception as e:
            # Log web search error but don't crash - continue with local context
            print(f"⚠️ Web search failed: {e}")
            web_context = ""

    # Build prompt
    # FIX: Check if we have any context at all
    if not local_context and not web_context:
        # No context retrieved - inform user
        no_context_msg = "I couldn't find relevant information in the uploaded documents to answer your question. Please try rephrasing or check if the document contains this information."
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": no_context_msg})
        return no_context_msg, chat_history

    context = local_context
    if web_context:
        context += "\n\n=== Additional Web Context ===\n" + web_context

    # Include chat history in the prompt (FIXED format)
    history_str = ""
    if chat_history:
        history_str = "\n\nPrevious conversation:\n" + "\n".join(
            [f"{h['role'].capitalize()}: {h['content']}" for h in chat_history[-6:]]
            # FIX: Only last 3 exchanges to avoid token limits
        )

    # ============== NEW: Conditional prompt based on query type ==============
    # Use specialized prompt for document summaries, standard prompt for specific questions

    if is_summary_query:
        # DOCUMENT SUMMARY QUERY - use specialized prompt
        # This prompt emphasizes proportional representation and guards against topic inversion
        prompt = None  # build_summary_prompt(question, context, retrieved_chunks, history_str)
        print(f"   Using SPECIALIZED SUMMARY PROMPT (with keyword frequency analysis)")
    else:
        # SPECIFIC QUESTION - use standard prompt
        # OLD APPROACH (single prompt for all queries):
        # This didn't distinguish between "What is bias?" and "What does this PDF talk about?"
        # causing the same retrieval and prompting strategy for very different query types

        # NEW APPROACH: Keep the improved specific-question prompt
        prompt = f"""<s>[INST]
You are a Study Buddy assistant helping students learn from their documents.

INSTRUCTIONS:
1. Answer the question using ONLY information from the context below
2. If the answer is in the context, provide a clear and detailed explanation
3. If the context doesn't contain enough information, say so honestly
4. Use examples from the context when possible
5. Be concise but thorough

Context from your documents:
{context}
{history_str}

Student's Question:
{question}

Remember: Base your answer strictly on the context provided. If you're not sure, say so.
[/INST]
"""
        print(f"   Using STANDARD Q&A PROMPT")

    # ===========================================================================
    print(f"{'=' * 60}\n")

    # Generate answer using lazy-loaded LLM
    llm = get_llm()
    result = llm.chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful study assistant that creates clear explanations based on provided documents. Always cite information from the context."
                # FIX: More specific system prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=600,  # todo: can be in config / UI
        temperature=0.3,  # todo: can be in config / UI
    )

    answer = result.choices[0].message.content
    print(f"\n{'=' * 60}")
    print(f"FINAL ANSWER:")
    print(answer)
    print(f"{'=' * 60}\n")

    # Update chat history - append QUESTION and ANSWER
    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": answer})

    return answer, chat_history


# Helper function for web search decision (from previous artifact)
def should_use_web_search(question, local_context):
    """
    Determine if web search is needed based on question type and local context quality.
    """
    local_only_patterns = [
        "what is this document about",
        "summarize this",
        "what does this say",
        "according to the document",
        "in this document",
        "what are the main points"
    ]

    question_lower = question.lower()
    if any(pattern in question_lower for pattern in local_only_patterns):
        return False

    web_needed_patterns = [
        "latest",
        "current",
        "recent",
        "today",
        "what happened",
        "compare to",
        "vs",
        "more information about",
        "explain more about"
    ]

    if any(pattern in question_lower for pattern in web_needed_patterns):
        return True

    if len(local_context) < 200:
        return True

    return False