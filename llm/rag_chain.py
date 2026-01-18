from tavily import TavilyClient
from config import TOP_K
from embeddings.embedder import embedder
from llm.model import get_llm

# ===================== NEW: Document Summarization Helpers =====================
# Added to fix topic emphasis bias when asking "What does this PDF talk about?"

def is_document_summary_query(question: str) -> bool:
    """
    Detect if user is asking for a document-level summary.
    
    This helps identify broad queries that need more chunks for better topic coverage,
    as opposed to specific questions that can be answered with fewer chunks.
    
    Args:
        question: The user's question
        
    Returns:
        bool: True if this appears to be a summary/overview query
    """
    summary_patterns = [
        "what does this", "what is this", "summarize", 
        "main topic", "about what", "overview", 
        "key themes", "talk about", "main focus",
        "what are the", "primary topic", "main idea",
        "content of", "subject of", "focus on"
    ]
    q_lower = question.lower().strip()
    return any(pattern in q_lower for pattern in summary_patterns)


def extract_topic_keywords(chunks, top_n=15):
    """
    Extract frequent important terms from chunks to identify dominant topics.
    
    This provides quantitative evidence of which topics appear most frequently,
    helping to combat recency bias where later sections get over-emphasized.
    
    Args:
        chunks: List of text chunks
        top_n: Number of top keywords to return
        
    Returns:
        list: List of (keyword, frequency) tuples
    """
    from collections import Counter
    import re
    
    # Combine all chunks
    all_text = " ".join(chunks).lower()
    
    # Simple tokenization - extract words 4+ characters
    words = re.findall(r'\b[a-z]{4,}\b', all_text)
    
    # Count frequency
    word_freq = Counter(words)
    
    # Filter common stopwords that don't indicate topics
    stopwords = {
        'this', 'that', 'with', 'from', 'have', 'will', 'also', 
        'can', 'are', 'for', 'the', 'and', 'but', 'not', 'you',
        'all', 'were', 'been', 'has', 'had', 'when', 'more',
        'which', 'their', 'there', 'than', 'into', 'these',
        'such', 'what', 'some', 'other', 'only', 'then'
    }
    filtered = {w: c for w, c in word_freq.items() if w not in stopwords}
    
    # Get top N keywords
    top_keywords = Counter(filtered).most_common(top_n)
    
    return top_keywords


def build_summary_prompt(question, context, chunks, history_str):
    """
    Build a specialized prompt for document summarization queries.
    
    This prompt explicitly instructs the LLM to:
    1. Identify proportional topic coverage (prevents topic inversion)
    2. Prioritize by frequency, not semantic richness (prevents recency bias)
    3. Distinguish core content from contextual framing
    
    Args:
        question: User's question
        context: Retrieved context text
        chunks: List of retrieved chunks
        history_str: Chat history string
        
    Returns:
        str: Formatted prompt for the LLM
    """
    # Extract and format keyword frequencies as evidence
    topic_keywords = extract_topic_keywords(chunks, top_n=10)
    keywords_str = ", ".join([f"{word}({count})" for word, count in topic_keywords])
    
    prompt = f"""<s>[INST]
You are helping a student understand the MAIN FOCUS of their document.

CRITICAL CONTEXT:
- The student asked: "{question}"
- You have {len(chunks)} excerpts from their document
- Your job is to identify what the document PRIMARILY covers

RULES FOR DOCUMENT SUMMARIZATION:
1. Identify ALL major topics mentioned across the excerpts
2. Estimate rough proportions (e.g., "80% covers topic A, 15% topic B, 5% topic C")
3. PRIORITIZE topics by COVERAGE and FREQUENCY, not by semantic richness or philosophical depth
4. A topic appearing in only 2-3 out of 20 excerpts is MINOR, even if conceptually interesting
5. Topics at the END of documents are often conclusions or contextual framing, NOT the main content
6. Look for REPEATED concepts across multiple excerpts - that indicates core content
7. Use quantitative language: "primarily", "mainly", "briefly introduces", "most of the content"

KEYWORD FREQUENCY ANALYSIS (shows what terms appear most often):
{keywords_str}

Document excerpts:
{context}
{history_str}

Provide a summary that ACCURATELY reflects the PROPORTIONAL emphasis in the original document.
Start with the DOMINANT topic (appears in >50% of content) and clearly structure your answer to show relative importance.
[/INST]"""
    
    return prompt

# ================================================================================


def rag_with_llm(question, mode, index, chunks, tavily_api_key, chat_history, top_k=TOP_K):
    
    # ============== NEW: Detect document summary queries and adjust retrieval ==============
    # For broad queries like "What does this PDF talk about?", we need MORE chunks
    # to get better topic coverage and avoid over-emphasizing minor sections
    is_summary_query = is_document_summary_query(question)
    
    if is_summary_query:
        # OLD APPROACH (used fixed top_k=5):
        # This retrieved too few chunks, causing topic bias toward semantically rich sections
        
        # NEW APPROACH: Retrieve more chunks for better topic coverage
        # Use 20 chunks (or all available if less) to capture dominant themes
        adaptive_top_k = min(20, len(chunks))
        print(f"\n{'='*60}")
        print(f"📋 DOCUMENT SUMMARY QUERY DETECTED")
        print(f"   Using {adaptive_top_k} chunks (instead of {top_k}) for better topic coverage")
        print(f"   This helps identify DOMINANT vs MINOR themes")
        print(f"{'='*60}\n")
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
    print(f"\n{'='*60}")
    print(f"RETRIEVAL DEBUG INFO:")
    print(f"Question: {question}")
    print(f"Retrieved {len(valid_indices)} chunks from index")
    print(f"Chunk indices: {valid_indices}")
    print(f"First chunk preview: {retrieved_chunks[0][:200] if retrieved_chunks else 'NONE'}...")
    print(f"{'='*60}\n")
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
            [f"{h['role'].capitalize()}: {h['content']}" for h in chat_history[-6:]]  # FIX: Only last 3 exchanges to avoid token limits
        )

    # ============== NEW: Conditional prompt based on query type ==============
    # Use specialized prompt for document summaries, standard prompt for specific questions
    
    if is_summary_query:
        # DOCUMENT SUMMARY QUERY - use specialized prompt
        # This prompt emphasizes proportional representation and guards against topic inversion
        prompt = build_summary_prompt(question, context, retrieved_chunks, history_str)
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
    print(f"{'='*60}\n")

    # Generate answer using lazy-loaded LLM
    llm = get_llm()
    result = llm.chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful study assistant that creates clear explanations based on provided documents. Always cite information from the context."  # FIX: More specific system prompt
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
    print(f"\n{'='*60}")
    print(f"FINAL ANSWER:")
    print(answer)
    print(f"{'='*60}\n")

    # Update chat history - append QUESTION and ANSWER
    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": answer})

    return answer, chat_history
