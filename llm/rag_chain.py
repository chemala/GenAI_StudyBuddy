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


# Integration into your RAG function
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