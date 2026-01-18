import gradio as gr
from config import TAVILY_API_KEY
from embeddings.index import build_index
from ingestion.chunking import chunk_text
from ingestion.pdf_loader import extract_pdf_text_from_path
from llm.flashcards import generate_flashcards
from llm.rag_chain import rag_with_llm
from state import state


def build(files):
    if not files:
        return "No files uploaded."

    texts = []
    for f in files:
        text = extract_pdf_text_from_path(f)
        texts += chunk_text(text)

    idx, ch = build_index(texts)
    state["index"] = idx
    state["chunks"] = ch

    return f"Indexed {len(ch)} chunks."


def format_chat_history_for_display():
    """Convert messages format to old Gradio tuple format"""
    formatted_history = []
    for i in range(0, len(state['chat_history']), 2):
        if i + 1 < len(state['chat_history']):
            user_msg = state['chat_history'][i]['content']
            assistant_msg = state['chat_history'][i + 1]['content']
            formatted_history.append([user_msg, assistant_msg])
    return formatted_history


def ask(q, mode, api_key, inference_mode):
    """
    Updated ask function with inference mode parameter
    """
    if state["index"] is None:
        return "Build the index first.", [], gr.update(value=q)

    # Convert checkbox value to boolean for allow_inference parameter
    allow_inference = (inference_mode == "Inference Mode")

    answer, updated_chat_history = rag_with_llm(
        q,
        mode,
        state["index"],
        state["chunks"],
        api_key,
        state["chat_history"],
        allow_inference=allow_inference  # Pass the inference mode
    )

    state["chat_history"] = updated_chat_history

    return answer, state['chat_history'], gr.update(value="")


def call_generate_flashcards(flashcard_topic, num_flashcards, mode, api_key):
    if state["index"] is None:
        return "Build the index first before generating flashcards.", []

    flashcards_data = generate_flashcards(
        flashcard_topic,
        int(num_flashcards),
        mode,
        state["index"],
        state["chunks"],
        api_key
    )

    # Format for gr.DataFrame
    formatted_flashcards = [[card['question'], card['answer']] for card in flashcards_data]
    return formatted_flashcards


def launch_ui():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📚 RAG Study Assistant")
        gr.Markdown("Upload PDFs, build an index, and ask questions with optional web search.")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📄 Document Management")
                files = gr.File(
                    file_types=[".pdf"],
                    file_count="multiple",
                    label="Upload PDFs"
                )

                api = gr.Textbox(
                    label="Tavily API Key",
                    value=TAVILY_API_KEY,
                    type="password"
                )

                build_btn = gr.Button("🔨 Build Index", variant="primary")
                status = gr.Textbox(label="Status", interactive=False)

            with gr.Column(scale=2):
                gr.Markdown("### ⚙️ Settings")

                with gr.Row():
                    mode = gr.Radio(
                        ["local", "web", "hybrid"],
                        value="hybrid",
                        label="Search Mode",
                        info="Local: PDF only | Web: Online only | Hybrid: Both sources"
                    )

                    inference_mode = gr.Radio(
                        ["Strict Mode", "Inference Mode"],
                        value="Strict Mode",
                        label="Answer Mode",
                        info="Strict: Only explicit facts | Inference: Allows reasoning from context"
                    )

                with gr.Accordion("ℹ️ Mode Explanations", open=False):
                    gr.Markdown("""
                    **Search Modes:**
                    - **Local**: Answers only from uploaded PDFs
                    - **Web**: Searches the internet for current information
                    - **Hybrid**: Combines PDF content with web search when needed

                    **Answer Modes:**
                    - **Strict Mode**: Only provides information explicitly stated in the documents. Best for factual verification.
                    - **Inference Mode**: Can make reasonable conclusions based on emphasis, patterns, and context. Best for analytical questions.
                    """)

        gr.Markdown("---")
        gr.Markdown("### 💬 Chat Interface")

        chatbot = gr.Chatbot(
            label="Conversation History",
            height=400
        )

        question = gr.Textbox(
            label="Ask a Question",
            placeholder="e.g., What is the biggest source of bias in GenAI?",
            lines=2
        )

        with gr.Row():
            ask_btn = gr.Button("🔍 Ask Question", variant="primary", scale=2)
            clear_btn = gr.Button("🗑️ Clear Chat", scale=1)

        answer = gr.Textbox(
            label="Latest Answer",
            lines=8,
            interactive=False
        )

        gr.Markdown("---")
        gr.Markdown("### 🃏 Flashcard Generator")

        with gr.Row():
            flashcard_topic = gr.Textbox(
                label="Flashcard Topic",
                placeholder="e.g., Key concepts from the document",
                scale=2
            )
            num_flashcards = gr.Number(
                label="Number of Cards",
                value=5,
                minimum=1,
                maximum=20,
                step=1,
                scale=1
            )

        generate_flashcards_btn = gr.Button("✨ Generate Flashcards", variant="secondary")

        flashcard_display = gr.DataFrame(
            headers=["Question", "Answer"],
            value=[],
            label="Generated Flashcards",
            wrap=True
        )

        # Event handlers
        build_btn.click(build, inputs=files, outputs=status)

        ask_btn.click(
            fn=ask,
            inputs=[question, mode, api, inference_mode],  # Added inference_mode
            outputs=[answer, chatbot, question]
        )

        # Clear chat function
        def clear_chat():
            state["chat_history"] = []
            return [], "", []

        clear_btn.click(
            fn=clear_chat,
            inputs=None,
            outputs=[chatbot, answer, chatbot]
        )

        generate_flashcards_btn.click(
            fn=call_generate_flashcards,
            inputs=[flashcard_topic, num_flashcards, mode, api],
            outputs=flashcard_display
        )

        # Update chatbot display on load
        def update_chatbot_display():
            return format_chat_history_for_display()

        demo.load(update_chatbot_display, inputs=None, outputs=chatbot)

    demo.launch(share=True, debug=True)


if __name__ == "__main__":
    print("Tavily key set:", bool(TAVILY_API_KEY))
    launch_ui()