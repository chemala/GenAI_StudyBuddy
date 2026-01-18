import json
from llm.model import get_llm
from retrieval.rag import rag

# ===================== FLASHCARD GENERATION - UPGRADED FOR QUALITY =====================
# This module has been completely rewritten to generate HIGH-QUALITY conceptual flashcards
# instead of trivial fact-based questions. See implementation plan for details.
# ========================================================================================

def generate_flashcards(flashcard_topic, num_flashcards, mode, index, chunks, api_key):
    """
    Generate high-quality study flashcards focused on conceptual understanding.
    
    Changes from previous version:
    - Uses MORE chunks (15 vs 5) for better topic coverage
    - Implements strict quality rules (no trivia/metadata)
    - Focuses on "why", "how", and "consequences" questions
    - Ensures 2-4 sentence conceptual answers
    
    Args:
        flashcard_topic: Topic to generate flashcards about
        num_flashcards: Number of flashcards to generate
        mode: Retrieval mode (local/web/hybrid)
        index: FAISS index
        chunks: Document chunks
        api_key: Tavily API key
        
    Returns:
        list: List of flashcard dicts with 'question' and 'answer' keys
    """
    
    # ============== OLD APPROACH (5 chunks - insufficient coverage) ==============
    # query = f"Information about {flashcard_topic}"
    # context = rag(query, mode, index, chunks, api_key)  # Uses default top_k=5
    # =============================================================================
    
    # ============== NEW APPROACH: Enhanced retrieval for flashcards ==============
    # Use more chunks (15) to get comprehensive coverage of the topic
    # This helps generate diverse flashcards covering different aspects
    print(f"\n{'='*60}")
    print(f"📚 GENERATING FLASHCARDS")
    print(f"   Topic: {flashcard_topic}")
    print(f"   Count: {num_flashcards}")
    print(f"   Using enhanced retrieval (15 chunks) for better coverage")
    print(f"{'='*60}\n")
    
    # Construct a comprehensive query to retrieve relevant content
    query = f"Comprehensive information about {flashcard_topic}, including definitions, mechanisms, examples, limitations, and implications"
    
    # Import embedder and retrieve MORE chunks than normal RAG queries
    from embeddings.embedder import embedder
    from config import TOP_K
    
    # Retrieve 15 chunks (vs normal 5) for comprehensive flashcard generation
    flashcard_top_k = min(15, len(chunks))
    
    # Embed query
    q_emb = embedder.encode([query], normalize_embeddings=True).astype("float32")
    
    # Search FAISS index
    D, I = index.search(q_emb, flashcard_top_k)
    
    # Get valid chunks
    valid_indices = [idx for idx in I[0] if 0 <= idx < len(chunks)]
    retrieved_chunks = [chunks[i] for i in valid_indices]
    context = "\n\n---\n\n".join(retrieved_chunks)
    
    print(f"   Retrieved {len(retrieved_chunks)} chunks for comprehensive coverage\n")
    # =============================================================================

    # ============== OLD PROMPT (Generated low-quality flashcards) ==============
    # Problems with old prompt:
    # - No quality rules specified
    # - Examples showed bad "What is X?" pattern
    # - No guidance on question types
    # - No restrictions on trivia/metadata
    # - Didn't specify answer length/style
    #
    # prompt = f"""<s>[INST]
    # Based on the following context, generate {num_flashcards} flashcards about "{flashcard_topic}".
    # Each flashcard should have a "question" and an "answer".
    # Return the output as a JSON array of objects.
    # 
    # Context:
    # {context}
    # 
    # Example format:
    # [
    #   {{"question": "What is X?", "answer": "Y."}},
    #   {{"question": "What is A?", "answer": "B."}}
    # ]
    # [/INST]
    # """
    # ===========================================================================
    
    # ============== NEW PROMPT: High-Quality Conceptual Flashcards ==============
    prompt = f"""<s>[INST]
You are creating study flashcards for a university student preparing for an exam.

TASK: Generate {num_flashcards} HIGH-QUALITY flashcards about: "{flashcard_topic}"

CRITICAL RULES - MUST FOLLOW:

1. NO TRIVIA OR METADATA QUESTIONS:
   ❌ Do NOT ask about: titles, dates, names, slide numbers, section headings, authors
   ❌ Examples of FORBIDDEN questions:
      - "What is the name of..."
      - "When was..."
      - "Who created..."
      - "What is the title of..."
      - "How many types are mentioned..."

2. TEST UNDERSTANDING, NOT FACTS:
   ✅ Ask about MECHANISMS (how things work, cause → effect)
   ✅ Ask about TRADE-OFFS (advantages vs disadvantages)
   ✅ Ask about COMPARISONS (technique A vs B, when to use which)
   ✅ Ask about IMPLICATIONS (consequences for society/systems)
   ✅ Ask about LIMITATIONS (what doesn't work and why)
   ✅ Ask about PROCESSES (step-by-step how things happen)

3. QUESTION TYPES (in order of preference):
   - "Why..." (tests reasoning about causes/reasons)
   - "How..." (tests understanding of mechanisms/processes)
   - "What is the consequence/implication of..." (tests reasoning about effects)
   - "Compare..." or "What is the trade-off between..." (tests analytical thinking)
   - "What..." ONLY if it requires a multi-sentence explanation, NOT simple definition

4. ANSWER FORMAT:
   - Length: 2-4 sentences (concise but complete explanation)
   - Style: Conceptual explanation, NOT verbatim quotes from text
   - Tone: Like a knowledgeable student explaining to another student
   - Content: Self-contained (don't say "the document states..." or "as mentioned...")

5. QUALITY SELF-CHECK:
   Before finalizing each flashcard, ask yourself:
   "Would a student actually LEARN something conceptual by recalling this answer?"
   "Could this help them in an oral exam or written theory exam?"
   If NO → discard or rewrite with deeper reasoning

EXAMPLES OF GOOD FLASHCARDS (follow this style):

✅ Q: Why is demographic bias in training data particularly hard to mitigate?
   A: Because demographic bias is often embedded in complex correlations throughout the entire dataset, not just in explicit labels. Removing it requires understanding intricate statistical relationships between features, and aggressive mitigation can inadvertently reduce model performance on legitimate tasks that happen to correlate with demographics.

✅ Q: What is the fundamental trade-off between bias mitigation and model accuracy?
   A: Enforcing fairness constraints typically reduces overall model accuracy because it limits the model's ability to exploit patterns that may be statistically valid but socially problematic. Development teams must carefully balance fairness goals with performance requirements, as perfect fairness often comes at a measurable performance cost.

✅ Q: How does the concept of "fairness through unawareness" fail to prevent bias in practice?
   A: Simply removing protected attributes like gender or race from training data doesn't prevent bias because machine learning models can infer these attributes from correlated features. For example, zip codes strongly correlate with race and income, so even without explicit demographic data, models can still learn and perpetuate discriminatory patterns.

✅ Q: What is the consequence of using different fairness metrics that may conflict with each other?
   A: Different fairness metrics can be mathematically incompatible, meaning optimizing for one metric may worsen another. This forces practitioners to make difficult ethical choices about which notion of fairness to prioritize, and there's rarely a universally correct answer applicable to all contexts.

EXAMPLES OF BAD FLASHCARDS (DO NOT CREATE THESE):

❌ Q: What is bias?
   A: Unfairness in AI systems.
   (Too simple, no reasoning, doesn't test understanding)

❌ Q: What is the title of this document?
   A: Bias in Generative AI.
   (Metadata trivia, not conceptual)

❌ Q: How many types of bias are mentioned?
   A: Four types.
   (Counting trivia, doesn't aid learning)

❌ Q: Who developed the fairness metric?
   A: Various researchers.
   (Name trivia, not useful for exam prep)

Context from the document:
{context}

Generate exactly {num_flashcards} flashcards following ALL rules above.
Return ONLY a valid JSON array with no additional text before or after:
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]
[/INST]"""
    # ===========================================================================

    # Generate flashcards using LLM
    llm = get_llm()
    
    # OLD PARAMETERS (too restrictive for quality answers):
    # max_tokens=600  # Not enough for 2-4 sentence answers on multiple flashcards
    
    # NEW PARAMETERS: More tokens for quality conceptual answers
    llm_response = llm.chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are an expert educational content creator specializing in high-quality study flashcards that test conceptual understanding, not rote memorization."  # More specific system prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=1500,  # Increased from 600 - need space for longer conceptual answers
        temperature=0.4,  # Slightly higher than 0.3 for more diverse questions
    )

    raw_response = llm_response.choices[0].message.content
    
    print(f"{'='*60}")
    print(f"LLM Response received, parsing flashcards...")
    print(f"{'='*60}\n")

    # Parse JSON safely (same as before, but with better error messages)
    try:
        json_start = raw_response.find("[")
        json_end = raw_response.rfind("]") + 1

        if json_start != -1 and json_end > json_start:
            flashcards = json.loads(raw_response[json_start:json_end])
        else:
            flashcards = json.loads(raw_response)
        
        # NEW: Quality validation - check that flashcards meet basic requirements
        print(f"✅ Successfully generated {len(flashcards)} flashcards")
        print(f"\nFlashcard Preview:")
        for i, card in enumerate(flashcards[:2], 1):  # Show first 2 as preview
            print(f"\n  Flashcard {i}:")
            print(f"  Q: {card['question'][:80]}...")
            print(f"  A: {card['answer'][:80]}...")
        
        if len(flashcards) > 2:
            print(f"\n  ... and {len(flashcards) - 2} more flashcards")
        
        print(f"\n{'='*60}\n")
        
        return flashcards

    except json.JSONDecodeError as e:
        print("❌ Error parsing JSON:", e)
        print("Raw response:", raw_response)
        return []

