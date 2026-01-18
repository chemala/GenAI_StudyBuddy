import gradio as gr
from config import TAVILY_API_KEY
from embeddings.index import build_index
from ingestion.chunking import chunk_text
from ingestion.pdf_loader import extract_pdf_text_from_path
from llm.flashcards import generate_flashcards
from llm.rag_chain import rag_with_llm
from state import state


def build(files):
    """
    Build the document index from uploaded PDF files.
    
    Args:
        files: List of uploaded PDF file paths
        
    Returns:
        str: Status message for the user
    """
    if not files:
        return "No files uploaded."
    
    # FIX: Add comprehensive error handling for the entire build process
    try:
        texts = []
        total_chars = 0
        
        # OLD CODE (no error handling):
        # for f in files:
        #     text = extract_pdf_text_from_path(f)
        #     texts += chunk_text(text)
        
        # NEW CODE: Process each file with individual error handling
        for f in files:
            try:
                # Extract text from PDF (with validation)
                text = extract_pdf_text_from_path(f)
                total_chars += len(text)
                
                # Chunk the text (with validation)
                chunks = chunk_text(text)
                texts += chunks
                
                print(f"✓ Processed {f.name}: {len(chunks)} chunks created")
                
            except ValueError as ve:
                # Specific error (empty PDF, corrupted file, etc.)
                return f"❌ Error processing {f.name}: {ve}"
            except Exception as e:
                # Unexpected error
                return f"❌ Unexpected error processing {f.name}: {e}"
        
        # FIX: Validate that we have chunks before building index
        if not texts or len(texts) == 0:
            return "❌ No text could be extracted from the uploaded PDFs. Please check if they contain readable text."
        
        print(f"\n📊 Total: {len(texts)} chunks from {total_chars} characters")
        
        # Build the index (with validation)
        idx, ch = build_index(texts)
        
        # Store in global state
        state["index"] = idx
        state["chunks"] = ch
        
        # FIX: More informative success message
        return f"✅ Success! Indexed {len(ch)} chunks from {len(files)} PDF(s). You can now ask questions about your documents."
    
    except ValueError as ve:
        # Specific errors from our validation
        return f"❌ Indexing failed: {ve}"
    except Exception as e:
        # Catch-all for unexpected errors
        print(f"❌ Unexpected error during index build: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ An unexpected error occurred: {e}. Please check the console for details."

def format_chat_history_for_display():
    """Convert messages format to old Gradio tuple format"""
    formatted_history = []
    for i in range(0, len(state['chat_history']), 2):
        if i + 1 < len(state['chat_history']):
            user_msg = state['chat_history'][i]['content']
            assistant_msg = state['chat_history'][i + 1]['content']
            formatted_history.append([user_msg, assistant_msg])
    return formatted_history


def ask(q, mode, api_key):
    if state["index"] is None:
        return "Build the index first.", [], gr.update(value=q)

    answer, updated_chat_history = rag_with_llm(
        q,
        mode,
        state["index"],
        state["chunks"],
        api_key,
        state["chat_history"]
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
    with gr.Blocks() as demo:
        gr.Markdown("## RAG Study Assistant")

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

        build_btn = gr.Button("Build Index")
        status = gr.Textbox(label="Status")

        mode = gr.Radio(
            ["local", "web", "hybrid"],
            value="hybrid",
            label="Search mode"
        )

        question = gr.Textbox(label="Question", lines=2)

        # Add Chatbot component for displaying history
        chatbot = gr.Chatbot(
            label="Chat History",
            height=300,
        )

        answer = gr.Textbox(label="Answer", lines=12)

        # New components for flashcard generation
        flashcard_topic = gr.Textbox(
            label="Flashcard Topic",
            placeholder="e.g., Key concepts from the document"
        )
        num_flashcards = gr.Number(
            label="Number of Flashcards",
            value=5,
            minimum=1,
            maximum=20,
            step=1
        )

        # Add gr.DataFrame for displaying generated flashcards
        flashcard_display = gr.DataFrame(
            headers=["Question", "Answer"],
            value=[],
            label="Generated Flashcards"
        )

        ask_btn = gr.Button("Ask")
        generate_flashcards_btn = gr.Button("Generate Flashcards") # New button

        build_btn.click(build, inputs=files, outputs=status)

        # Update ask_btn.click to return the answer, chat history, and clear the question textbox
        ask_btn.click(
            fn=ask,
            inputs=[question, mode, api],
            outputs=[answer, chatbot, question] # Now outputs both the answer, chat history, and the question component for clearing
        )

        # Attach the call_generate_flashcards function to the new button's click event
        generate_flashcards_btn.click(
            fn=call_generate_flashcards,
            inputs=[flashcard_topic, num_flashcards, mode, api],
            outputs=flashcard_display
        )

        # Function to update the chatbot display from the state's chat_history
        def update_chatbot_display():
            return format_chat_history_for_display()

        # Initial load and subsequent updates for the chatbot
        demo.load(update_chatbot_display, inputs=None, outputs=chatbot)
        # Removed the problematic line with _js

    demo.launch(share=True, debug=True)

if __name__ == "__main__":
    print("Tavily key set:", bool(TAVILY_API_KEY))
    launch_ui()