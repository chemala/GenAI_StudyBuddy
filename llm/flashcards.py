import json
from config import TOP_K
from embeddings.embedder import embedder
from llm.model import get_llm
from retrieval.web import tavily_search, filter_relevant_web_results
from llm.rag_chain import mmr
import numpy as np


def generate_flashcards(flashcard_topic, num_flashcards, mode, index, chunks, tavily_api_key, top_k=TOP_K):
    """
    Generate flashcards using the same RAG approach as rag_chain.py

    Args:
        flashcard_topic: Topic to generate flashcards about
        num_flashcards: Number of flashcards to generate
        mode: "local", "web", or "hybrid"
        index: Vector index (not used with MMR, kept for compatibility)
        chunks: Document chunks
        tavily_api_key: API key for Tavily search
        top_k: Number of chunks to retrieve
    """
    # Construct query for context retrieval
    query = f"Information about {flashcard_topic}"

    # Compute embeddings for query and chunks
    q_emb = embedder.encode([query], normalize_embeddings=True).astype("float32")[0]
    chunk_embs = embedder.encode(chunks, normalize_embeddings=True).astype("float32")

    # Use MMR to select diverse but relevant chunks
    selected = mmr(q_emb, chunk_embs, k=top_k)
    local_context = "\n".join(chunks[i] for i in selected)

    # Gather web context if needed
    web_context = ""
    if mode in ["web", "hybrid"] and tavily_api_key:
        if should_use_web_for_flashcards(flashcard_topic, local_context):
            web_results = tavily_search(query, tavily_api_key, k=5)
            web_results = filter_relevant_web_results(
                q_emb.flatten(),
                web_results,
                embedder,
                top_n=3,
                similarity_threshold=0.5
            )
            if web_results:
                web_context = "\n".join(web_results)

    # Combine contexts
    context_parts = []
    if local_context:
        context_parts.append(f"DOCUMENT CONTEXT:\n{local_context}")
    if web_context:
        context_parts.append(f"\nADDITIONAL WEB SOURCES:\n{web_context}")

    full_context = "\n\n".join(context_parts)

    # Build flashcard generation prompt
    prompt = f"""Based on the following context, generate {num_flashcards} flashcards about "{flashcard_topic}".

INSTRUCTIONS:
- Each flashcard should have a clear "question" and a concise "answer"
- Use ONLY information explicitly stated in the context
- Questions should test key concepts, definitions, and important facts
- Answers should be accurate and directly supported by the context
- Return output as a JSON array of objects
- If the context doesn't contain enough information for {num_flashcards} flashcards, generate fewer flashcards and acknowledge this limitation

{full_context}

Return ONLY a JSON array in this exact format:
[
  {{"question": "What is X?", "answer": "Y."}},
  {{"question": "What is A?", "answer": "B."}}
]

Do not include any text before or after the JSON array."""

    # Generate flashcards using LLM
    llm = get_llm()
    result = llm.chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful study assistant that creates clear, accurate flashcards based on provided context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=1500,
        temperature=0.3,  # Slightly higher for more varied question formats
    )

    raw_response = result.choices[0].message.content

    # Parse JSON safely
    try:
        # Try to extract JSON array from response
        json_start = raw_response.find("[")
        json_end = raw_response.rfind("]") + 1

        if json_start != -1 and json_end > json_start:
            flashcards = json.loads(raw_response[json_start:json_end])
        else:
            flashcards = json.loads(raw_response)

        # Validate flashcard structure
        valid_flashcards = []
        for card in flashcards:
            if isinstance(card, dict) and "question" in card and "answer" in card:
                valid_flashcards.append(card)

        return valid_flashcards

    except json.JSONDecodeError as e:
        print("Error parsing JSON:", e)
        print("Raw response:", raw_response)
        return []


def should_use_web_for_flashcards(topic, local_context):
    """
    Determine if web search would be helpful for flashcard generation.
    Generally more conservative than question answering - prefer local context.
    """
    # If we have substantial local context, prefer it
    if len(local_context) > 500:
        return False

    # Web-oriented topics
    web_needed_patterns = [
        "latest",
        "current",
        "recent",
        "today",
        "modern",
        "contemporary"
    ]

    topic_lower = topic.lower()
    if any(pattern in topic_lower for pattern in web_needed_patterns):
        return True

    # If local context is too sparse, web search might help
    if len(local_context) < 200:
        return True

    return False