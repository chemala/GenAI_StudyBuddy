"""
Flashcard Generation Module - Merged & Optimized Version

This module generates HIGH-QUALITY conceptual flashcards using RAG retrieval
with MMR (Maximal Marginal Relevance) for diverse, relevant chunk selection.

Key Features:
- MMR-based retrieval for diverse but relevant context
- Enhanced prompt engineering for conceptual (not trivia) flashcards
- Optional web search augmentation for hybrid mode
- Comprehensive logging and validation
"""

import json
from typing import List, Dict, Optional

from config import TOP_K
from embeddings.embedder import embed
from llm.model import get_llm
from retrieval.web import tavily_search, filter_relevant_web_results
from llm.rag_chain import mmr, reformulate_query_for_web


def retrieve_context_rag_style(
    query: str,
    mode: str,
    chunks: List[str],
    tavily_api_key: Optional[str],
    top_k: int,
    llm,
):
    """
    Shared RAG-style retrieval used by both Q&A and flashcard generation.
    """

    if not chunks:
        return ""

    # --- Embeddings ---
    q_emb = embed([query])[0]

    chunk_embs = embed(chunks)

    # --- MMR retrieval ---
    selected = mmr(q_emb, chunk_embs, k=min(top_k, len(chunks)))
    selected = [i for i in selected if 0 <= i < len(chunks)]

    local_context = "\n\n---\n\n".join(chunks[i] for i in selected)

    # --- Web augmentation (same as rag_with_llm) ---
    web_context = ""
    if mode in ["web"] and tavily_api_key:
        try:
            search_query = reformulate_query_for_web(query, local_context, llm)
            print(f"Flashcard web query: {search_query}")

            web_results = tavily_search(search_query, tavily_api_key, k=5)
            web_results = filter_relevant_web_results(
                q_emb.flatten(),
                web_results,
                top_n=3,
            )

            if web_results:
                web_context = "\n\n".join(web_results)

        except Exception as e:
            print(f"Web augmentation failed: {e}")

    # --- Merge context ---
    parts = []
    if local_context:
        parts.append(f"DOCUMENT CONTEXT:\n{local_context}")
    if web_context:
        parts.append(f"ADDITIONAL WEB SOURCES:\n{web_context}")

    return "\n\n".join(parts)

def generate_flashcards(
    flashcard_topic: str,
    num_flashcards: int,
    mode: str,
    index,
    chunks: List[str],
    tavily_api_key: Optional[str] = None,
    top_k: int = None
) -> List[Dict[str, str]]:

    if not chunks:
        print("No document chunks available")
        return []

    if top_k is None:
        top_k = min(15, len(chunks))
    else:
        top_k = min(top_k, len(chunks))

    print(f"\n{'=' * 60}")
    print("   GENERATING FLASHCARDS (RAG-CHAIN STYLE)")
    print(f"   Topic: {flashcard_topic}")
    print(f"   Mode: {mode}")
    print(f"   top_k (MMR): {top_k}")
    print(f"{'=' * 60}\n")

    llm = get_llm()

    # --- RAG-style query (same philosophy as rag_with_llm) ---
    query = (
        f"Comprehensive explanation of {flashcard_topic}, "
        f"including mechanisms, trade-offs, implications, limitations, and examples"
    )

    # --- SHARED RAG RETRIEVAL ---
    full_context = retrieve_context_rag_style(
        query=query,
        mode=mode,
        chunks=chunks,
        tavily_api_key=tavily_api_key,
        top_k=top_k,
        llm=llm,
    )

    if not full_context.strip():
        print("No relevant context retrieved")
        return []

    # --- Prompt stays unchanged (this is your strength) ---
    prompt = _build_flashcard_prompt(
        flashcard_topic,
        num_flashcards,
        full_context
    )

    # --- LLM generation ---
    result = llm.chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert educational content creator specializing in "
                    "conceptual university-level flashcards."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=2000,
        temperature=0.4,
    )

    raw_response = result.choices[0].message.content

    flashcards = _parse_and_validate_flashcards(raw_response)
    _log_flashcard_summary(flashcards)

    return flashcards


def _build_flashcard_prompt(topic: str, num_flashcards: int, context: str) -> str:
    """
    Build the enhanced flashcard generation prompt with quality rules.

    This prompt enforces:
    - No trivia/metadata questions
    - Conceptual understanding focus
    - Proper question types (why, how, consequences)
    - Appropriate answer length and style
    """
    return f"""<s>[INST]
You are creating study flashcards for a university student preparing for an exam.

TASK: Generate {num_flashcards} HIGH-QUALITY flashcards about: "{topic}"

══════════════════════════════════════════════════════════════
CRITICAL RULES - MUST FOLLOW
══════════════════════════════════════════════════════════════

1. NO TRIVIA OR METADATA QUESTIONS:
      Do NOT ask about: titles, dates, names, slide numbers, section headings, authors
      FORBIDDEN question patterns:
      - "What is the name of..."
      - "When was..."
      - "Who created/developed..."
      - "What is the title of..."
      - "How many types/steps are mentioned..."
      - "According to the document..."

2. TEST UNDERSTANDING, NOT RECALL:
      MECHANISMS: How things work, cause → effect relationships
      TRADE-OFFS: Advantages vs disadvantages, when to use what
      COMPARISONS: Technique A vs B, approach X vs Y
      IMPLICATIONS: Consequences for systems, society, practice
      LIMITATIONS: What doesn't work and why
      PROCESSES: Step-by-step how things happen and why each step matters

3. PREFERRED QUESTION TYPES (in order):
   1st: "Why..." (tests reasoning about causes/reasons)
   2nd: "How..." (tests understanding of mechanisms/processes)
   3rd: "What is the consequence/implication of..." (tests effect reasoning)
   4th: "Compare..." / "What is the trade-off between..." (tests analysis)
   5th: "What..." ONLY if requiring multi-sentence conceptual explanation

4. ANSWER REQUIREMENTS:
   - Length: 2-4 sentences (concise but complete)
   - Style: Conceptual explanation in your own words
   - Tone: Like a knowledgeable student explaining to a peer
   - Self-contained: Never say "the document states..." or "as mentioned..."

5. QUALITY SELF-CHECK (apply to each flashcard):
   ✓ "Would answering this require actual understanding?"
   ✓ "Could this appear on an oral exam or theory test?"
   ✓ "Is this more than just recalling a definition or fact?"

══════════════════════════════════════════════════════════════
EXAMPLES
══════════════════════════════════════════════════════════════

   GOOD FLASHCARDS:

Q: Why is demographic bias in training data particularly difficult to mitigate?
A: Demographic bias is often embedded in complex correlations throughout the dataset, not just explicit labels. Removing it requires understanding intricate statistical relationships between features, and aggressive mitigation can inadvertently reduce model performance on legitimate tasks that correlate with demographics.

Q: What is the fundamental trade-off between bias mitigation and model accuracy?
A: Enforcing fairness constraints typically reduces overall accuracy because it limits the model's ability to exploit patterns that may be statistically valid but socially problematic. Teams must balance fairness goals with performance requirements, as perfect fairness often comes at a measurable cost.

Q: How does "fairness through unawareness" fail in practice?
A: Simply removing protected attributes doesn't prevent bias because ML models can infer these attributes from correlated features. For example, zip codes correlate strongly with race and income, so models can still learn discriminatory patterns without explicit demographic data.

   BAD FLASHCARDS (DO NOT CREATE):

Q: What is bias?
A: Unfairness in AI systems.
(Too simple, no reasoning required)

Q: What is the title of this section?
A: Introduction to Fairness.
(Metadata trivia)

Q: How many types of bias are discussed?
A: Four types.
(Counting trivia)

Q: Who developed this framework?
A: Researchers at MIT.
(Name trivia)

══════════════════════════════════════════════════════════════
CONTEXT FROM SOURCES
══════════════════════════════════════════════════════════════

{context}

══════════════════════════════════════════════════════════════
OUTPUT FORMAT
══════════════════════════════════════════════════════════════

Generate exactly {num_flashcards} flashcards following ALL rules above.
Return ONLY a valid JSON array with no additional text:

[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]
[/INST]"""


def _parse_and_validate_flashcards(raw_response: str) -> List[Dict[str, str]]:
    """
    Parse LLM response and validate flashcard structure.

    Args:
        raw_response: Raw text response from LLM

    Returns:
        List of validated flashcard dictionaries
    """
    try:
        # Extract JSON array from response
        json_start = raw_response.find("[")
        json_end = raw_response.rfind("]") + 1

        if json_start != -1 and json_end > json_start:
            flashcards = json.loads(raw_response[json_start:json_end])
        else:
            flashcards = json.loads(raw_response)

        # Validate structure and filter invalid cards
        valid_flashcards = []
        for card in flashcards:
            if isinstance(card, dict) and "question" in card and "answer" in card:
                # Ensure both fields are non-empty strings
                if card["question"].strip() and card["answer"].strip():
                    valid_flashcards.append({
                        "question": card["question"].strip(),
                        "answer": card["answer"].strip()
                    })

        return valid_flashcards

    except json.JSONDecodeError as e:
        print(f"   Error parsing flashcard JSON: {e}")
        print(f"   Raw response preview: {raw_response[:200]}...")
        return []


def _log_flashcard_summary(flashcards: List[Dict[str, str]]) -> None:
    """Log a summary of generated flashcards."""
    print(f"{'=' * 60}")

    if not flashcards:
        print("   No valid flashcards generated")
        print(f"{'=' * 60}\n")
        return

    print(f"   Successfully generated {len(flashcards)} flashcards")
    print(f"\n Preview (first 2 flashcards):\n")

    for i, card in enumerate(flashcards[:2], 1):
        q_preview = card['question'][:75] + "..." if len(card['question']) > 75 else card['question']
        a_preview = card['answer'][:75] + "..." if len(card['answer']) > 75 else card['answer']
        print(f"   [{i}] Q: {q_preview}")
        print(f"       A: {a_preview}")
        print()

    if len(flashcards) > 2:
        print(f"   ... and {len(flashcards) - 2} more flashcards")

    print(f"{'=' * 60}\n")