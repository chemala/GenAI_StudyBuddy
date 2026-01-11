import json
from llm.model import get_llm
from retrieval.rag import rag


def generate_flashcards(flashcard_topic, num_flashcards, mode, index, chunks, api_key):
    # 1. Construct a query string
    query = f"Information about {flashcard_topic}"

    # 2. Retrieve context via RAG
    context = rag(query, mode, index, chunks, api_key)

    # 3. Build prompt
    prompt = f"""<s>[INST]
Based on the following context, generate {num_flashcards} flashcards about "{flashcard_topic}".
Each flashcard should have a "question" and an "answer".
Return the output as a JSON array of objects.

Context:
{context}

Example format:
[
  {{"question": "What is X?", "answer": "Y."}},
  {{"question": "What is A?", "answer": "B."}}
]
[/INST]
"""

    # 4. Lazy-load LLM and generate
    llm = get_llm()
    # llm_response = llm(prompt) # bug :( todo: adjust for model Qwen2.5-7B-Instruct
    llm_response = llm.chat_completion(
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

    # raw_response = llm_response[0]["generated_text"].split("[/INST]")[-1].strip()
    raw_response = llm_response.choices[0].message.content

    # 5. Parse JSON safely
    try:
        json_start = raw_response.find("[")
        json_end = raw_response.rfind("]") + 1

        if json_start != -1 and json_end > json_start:
            return json.loads(raw_response[json_start:json_end])

        return json.loads(raw_response)

    except json.JSONDecodeError as e:
        print("Error parsing JSON:", e)
        print("Raw response:", raw_response)
        return []
