import gradio as gr
import time
from datetime import datetime
from config import TAVILY_API_KEY, HF_TOKEN, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
from embeddings.index import build_index
from ingestion.chunking import chunk_text
from ingestion.file_loader import extract_text_or_pdf
from llm.flashcards import generate_flashcards
from llm.rag_chain import rag_with_llm
from state import state
import base64
import markdown


def get_logo_base64():
    with open("logo.png", "rb") as f:
        return base64.b64encode(f.read()).decode()

def get_logo_tab_base64():
    with open("logo_tab.png", "rb") as f:
        return base64.b64encode(f.read()).decode()

# Enhanced CSS with active nav state
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --primary: #3b82f6;
    --primary-dark: #2563eb;
    --primary-light: #60a5fa;
    --secondary: #8b5cf6;
    --accent: #f59e0b;
    --success: #10b981;
    --danger: #ef4444;
    
    --bg-main: #0a0f1e;
    --bg-secondary: #111827;
    --bg-tertiary: #1f2937;
    --bg-card: #1f2937;
    --bg-hover: #374151;
    
    --text-primary: #f9fafb;
    --text-secondary: #e5e7eb;
    --text-muted: #9ca3af;
    
    --border: #374151;
    --border-light: #4b5563;
}

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

body {
    background: var(--bg-main) !important;
    color: var(--text-primary) !important;
}

.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
    background: var(--bg-main) !important;
}

#component-0, #component-1, .contain {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

.app-wrapper {
    display: flex !important;
    min-height: 100vh !important;
    background: var(--bg-main) !important;
}

.sidebar {
    width: 280px !important;
    min-width: 280px !important;
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
    padding: 1.5rem !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 0.5rem !important;
}

.main-content {
    flex: 1 !important;
    background: var(--bg-main) !important;
    padding: 2rem !important;
    overflow-y: auto !important;
}

/* Logo section - minimal spacing */
.logo-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 0.25rem 0 0.25rem 0;
    margin-bottom: 0.75rem;
    border-bottom: 1px solid var(--border);
    background: transparent !important;
}

.logo-container img {
    width: 300px;
    height: auto;
    filter: drop-shadow(0 0 20px rgba(59, 130, 246, 0.3));
}

/* Navigation buttons with active state and smooth transitions */
.nav-section button {
    width: 100% !important;
    margin: 0 0 0.5rem 0 !important;
    padding: 0.875rem 1rem !important;
    background: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    text-align: left !important;
    transition: all 0.15s ease-in-out !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
}

.nav-section button:hover {
    background: var(--bg-hover) !important;
    color: var(--text-primary) !important;
}

/* Active navigation state with smooth transition */
.nav-section button.nav-active {
    background: var(--primary) !important;
    color: white !important;
    transition: all 0.15s ease-in-out !important;
}

.settings-section {
    margin-top: auto;
    padding-top: 1rem;
}

.settings-section button {
    width: 100% !important;
    padding: 0.875rem 1rem !important;
    background: var(--bg-tertiary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    transition: all 0.15s ease-in-out !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
}

.settings-section button:hover {
    background: var(--bg-hover) !important;
    color: var(--text-primary) !important;
    border-color: var(--border-light) !important;
}

.settings-section button.nav-active {
    background: var(--primary) !important;
    color: white !important;
    border-color: var(--primary) !important;
    transition: all 0.15s ease-in-out !important;
}

.page-content {
    max-width: 1200px;
}

.page-header {
    margin-bottom: 2rem;
}

.page-title {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 0.5rem 0;
}

.page-description {
    font-size: 1rem;
    color: var(--text-muted);
    margin: 0;
}

.gr-form, .gr-box {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
}

input[type="text"],
input[type="number"],
input[type="password"],
textarea,
select {
    background: var(--bg-tertiary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    padding: 0.875rem 1rem !important;
    font-size: 0.95rem !important;
}

input:focus,
textarea:focus,
select:focus {
    outline: none !important;
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
}

label, .gr-text-label {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    margin-bottom: 0.5rem !important;
}

/* File upload styling */
.gr-file-upload {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

.gr-file {
    background: var(--bg-tertiary) !important;
    border: 2px dashed var(--border-light) !important;
    border-radius: 8px !important;
}

.gr-file:hover {
    border-color: var(--primary) !important;
}

/* Remove orange warning styling */
.gr-file.pending, .gr-file-upload.pending {
    border-color: var(--border-light) !important;
    background: var(--bg-tertiary) !important;
}

.warning, .gr-form.pending {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
}

button {
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 0.875rem 1.5rem !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}

.primary-btn, button.primary, button[class*="primary"] {
    background: var(--primary) !important;
    color: white !important;
    border: none !important;
}

.primary-btn:hover {
    background: var(--primary-dark) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.5) !important;
}

.secondary-btn, button.secondary, button[class*="secondary"] {
    background: var(--bg-tertiary) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
}

.secondary-btn:hover {
    background: var(--bg-hover) !important;
}

.status-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0;
}

.status-item {
    background: var(--bg-tertiary);
    padding: 1.25rem;
    border-radius: 8px;
    border-left: 3px solid var(--primary);
}

.status-label {
    font-size: 0.875rem;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
}

.status-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text-primary);
}

.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 12px;
    font-size: 0.875rem;
    font-weight: 600;
}

.badge-success {
    background: var(--success);
    color: white;
}

.badge-warning {
    background: var(--accent);
    color: white;
}

.mode-selector {
    display: flex;
    gap: 1rem;
    margin: 1.5rem 0;
}

.mode-btn {
    flex: 1;
    padding: 1rem;
    background: var(--bg-tertiary);
    border: 2px solid var(--border);
    border-radius: 8px;
    color: var(--text-secondary);
    font-weight: 600;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s ease;
    user-select: none;
}

.mode-btn:hover {
    background: var(--bg-hover);
    border-color: var(--border-light);
}

.mode-btn.active {
    background: var(--primary) !important;
    border-color: var(--primary) !important;
    color: white !important;
}

.answer-box {
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-left: 3px solid var(--primary);
    border-radius: 8px;
    padding: 1.5rem;
    margin-top: 1.5rem;
}

.answer-box p {
    color: var(--text-primary);
    line-height: 1.8;
    margin: 0;
}

.history-list {
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

.history-item {
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-left: 3px solid var(--primary);
    border-radius: 8px;
    padding: 1.25rem;
}

.history-q {
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.75rem;
    font-size: 1.05rem;
}

.history-a {
    color: var(--text-secondary);
    line-height: 1.6;
    padding-left: 1.5rem;
}

.history-time {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 0.75rem;
    padding-left: 1.5rem;
}

/* Flashcard section header */
.flashcard-section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    padding: 1rem;
    background: var(--bg-tertiary);
    border-radius: 8px;
    border: 1px solid var(--border);
}

.flashcard-instruction {
    color: var(--text-muted);
    font-size: 0.95rem;
    font-style: italic;
}

.flashcard-container {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 1.5rem;
}

.flashcard {
    background: var(--bg-tertiary);
    border: 2px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    min-height: 160px;
    cursor: pointer;
    transition: all 0.3s ease;
    display: flex;
    flex-direction: column;
}

.flashcard:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.6);
    border-color: var(--primary);
}

.flashcard-q {
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 1rem;
    flex: 1;
    font-size: 1.05rem;
}

.flashcard-a {
    color: var(--text-secondary);
    padding-top: 1rem;
    border-top: 1px solid var(--border);
    line-height: 1.6;
    display: none;
}

.flashcard.revealed .flashcard-a {
    display: block;
}

.empty-state {
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 3rem 2rem;
    text-align: center;
}

.empty-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--text-secondary);
    margin-bottom: 0.5rem;
}

.empty-subtitle {
    font-size: 0.875rem;
    color: var(--text-muted);
}

/* Improved settings group styling with padding */
.setting-group {
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

.nav-section button[variant="primary"],
.settings-section button[variant="primary"] {
    background: var(--primary) !important;
    color: white !important;
    border-color: var(--primary) !important;
}

.setting-header {
    font-size: 1.125rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 1.25rem;
    padding-left: 0.25rem;
}

.setting-desc {
    font-size: 0.875rem;
    color: var(--text-muted);
    margin-bottom: 1.25rem;
}

footer {
    display: none !important;
}

@media (max-width: 768px) {
    .sidebar {
        width: 100% !important;
        border-right: none !important;
        border-bottom: 1px solid var(--border) !important;
    }
    
    .flashcard-container {
        grid-template-columns: 1fr;
    }
    
    .mode-selector {
        flex-direction: column;
    }
}
"""

# JavaScript for navigation active states with smoother transitions
nav_js = """
<script>
// Add data-page attributes to buttons after load
window.addEventListener('load', () => {
    const buttons = {
        'nav-upload-btn': 'upload',
        'nav-questions-btn': 'questions',
        'nav-flashcards-btn': 'flashcards',
        'nav-history-btn': 'history',
        'nav-library-btn': 'library',
        'nav-settings-btn': 'settings'
    };
    
    for (const [btnId, page] of Object.entries(buttons)) {
        const btn = document.getElementById(btnId);
        if (btn) {
            btn.setAttribute('data-page', page);
            btn.addEventListener('click', () => {
                // Use requestAnimationFrame for smoother transition
                requestAnimationFrame(() => {
                    // Remove active from all
                    document.querySelectorAll('[data-page]').forEach(b => {
                        b.classList.remove('nav-active');
                    });
                    // Add to clicked with slight delay for visual smoothness
                    setTimeout(() => {
                        btn.classList.add('nav-active');
                    }, 10);
                });
            });
        }
    }
    
    // Set initial active
    const uploadBtn = document.getElementById('nav-upload-btn');
    if (uploadBtn) {
        uploadBtn.classList.add('nav-active');
    }
});
</script>
"""


# Helper functions
def build_index_handler(files, settings):
    """Build index from uploaded files (PDFs/TXTs)"""
    if not files:
        return "⚠️ Please upload at least one file first.", None, [], ""
    
    try:
        start_time = time.time()
        
        texts = []
        total_chars = 0
        file_count = 0
        
        # robust loop from main.py
        for f in files:
            try:
                # Extract text
                text = extract_text_or_pdf(f)
                total_chars += len(text)
                
                # Chunk text
                chunk_size = int(settings.get("chunk_size", CHUNK_SIZE))
                chunk_overlap = int(settings.get("chunk_overlap", CHUNK_OVERLAP))
                chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
                texts += chunks
                file_count += 1
                
            except ValueError as ve:
                print(f"Error processing {f.name}: {ve}")
                continue # skip bad file
            except Exception as e:
                print(f"Unexpected error processing {f.name}: {e}")
                continue

        if not texts:
             return "❌ No valid text extracted from uploaded files.", None, [], ""

        index, chunk_list = build_index(texts)
        process_time = time.time() - start_time
        
        state["index"] = index
        state["chunks"] = chunk_list
        
        status_html = f"""
        <div class="status-grid">
            <div class="status-item">
                <div class="status-label">Files Processed</div>
                <div class="status-value">{file_count}</div>
            </div>
            <div class="status-item">
                <div class="status-label">Total Chunks</div>
                <div class="status-value">{len(chunk_list)}</div>
            </div>
            <div class="status-item">
                <div class="status-label">Processing Time</div>
                <div class="status-value">{process_time:.1f}s</div>
            </div>
            <div class="status-item">
                <div class="status-label">Status</div>
                <div class="status-value"><span class="badge badge-success">Ready</span></div>
            </div>
        </div>
        """
        
        return f"✅ Successfully indexed {len(chunk_list)} chunks from {file_count} files", index, chunk_list, status_html
        
    except Exception as e:
        return f"❌ Error: {str(e)}", None, [], ""


def ask_question_handler(question, mode, inference_mode, index, chunks, settings, chat_history):
    """Handle question asking"""
    if not question or not question.strip():
        return "", chat_history, """
        <div class="empty-state">
            <div class="empty-title">No answer yet</div>
            <div class="empty-subtitle">Ask a question to get started</div>
        </div>
        """
    
    if index is None:
        return "", chat_history, """
        <div class="empty-state">
            <div class="empty-title">⚠️ Index not built</div>
            <div class="empty-subtitle">Please upload and index a PDF first</div>
        </div>
        """
    
    try:
        tavily_key = settings.get("tavily_api_key", TAVILY_API_KEY)
        top_k = int(settings.get("top_k", TOP_K))
        
        # Convert inference mode string to boolean
        allow_inference = (inference_mode.lower() == "inference")
        
        # Map modes if necessary, but UI should now pass "local" or "web"
        # Backend "web" is the new hybrid
        
        answer, updated_history = rag_with_llm(
            question, mode, index, chunks, tavily_key, chat_history, top_k=top_k, allow_inference=allow_inference
        )
        
        # Convert Markdown to HTML to preserve formatting
        answer_html_content = markdown.markdown(answer)
        
        answer_html = f"""
        <div class="answer-box">
            {answer_html_content}
        </div>
        """
        
        return "", updated_history, answer_html
        
    except Exception as e:
        return "", chat_history, f"""
        <div class="empty-state">
            <div class="empty-title">❌ Error</div>
            <div class="empty-subtitle">{str(e)}</div>
        </div>
        """


def format_chat_history_html(chat_history):
    """Format chat history as HTML"""
    if not chat_history or len(chat_history) == 0:
        return """
        <div class="empty-state">
            <div class="empty-title">No chat history yet</div>
            <div class="empty-subtitle">Ask questions to build your history</div>
        </div>
        """
    
    html = '<div class="history-list">'
    for i in range(len(chat_history) - 1, 0, -2):
        if i >= 1:
            user_msg = chat_history[i-1]["content"]
            assistant_msg = chat_history[i]["content"]
            
            # Convert Markdown to HTML for history display
            if user_msg:
                user_msg = markdown.markdown(user_msg)
            if assistant_msg:
                assistant_msg = markdown.markdown(assistant_msg)
                
            timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
            
            html += f"""
            <div class="history-item">
                <div class="history-q"><strong>Q:</strong> {user_msg}</div>
                <div class="history-a">{assistant_msg}</div>
                <div class="history-time">{timestamp}</div>
            </div>
            """
    html += '</div>'
    
    return html if len(chat_history) > 0 else """
    <div class="empty-state">
        <div class="empty-title">No chat history yet</div>
        <div class="empty-subtitle">Ask questions to build your history</div>
    </div>
    """


def generate_flashcards_handler(topic, num, mode, index, chunks, settings, library):
    """Generate flashcards - topic is optional"""
    # Allow empty topic
    if index is None:
        return """
        <div class="empty-state">
            <div class="empty-title">⚠️ Index not built</div>
            <div class="empty-subtitle">Please upload and index a PDF first</div>
        </div>
        """, library, format_flashcard_library_html(library)
    
    try:
        tavily_key = settings.get("tavily_api_key", TAVILY_API_KEY)
        
        # Use topic if provided, otherwise use generic prompt
        topic_text = topic.strip() if topic and topic.strip() else "Generate flashcards from the document"
        
        flashcards_data = generate_flashcards(topic_text, int(num), mode, index, chunks, tavily_key)
        
        if not flashcards_data:
            return """
            <div class="empty-state">
                <div class="empty-title">No flashcards generated</div>
                <div class="empty-subtitle">Try adjusting your settings or check your document</div>
            </div>
            """, library, format_flashcard_library_html(library)
        
        for card in flashcards_data:
            if card not in library:
                library.append(card)
        
        state["flashcard_library"] = library
        
        # Preview with instruction header
        preview_html = '''
        <div class="flashcard-section-header">
            <div class="flashcard-instruction">💡 Click on any flashcard to reveal its answer</div>
        </div>
        <div class="flashcard-container">
        '''
        
        for card in flashcards_data:
            q = card.get("question", "")
            a = card.get("answer", "")
            preview_html += f"""
            <div class="flashcard" onclick="this.classList.toggle('revealed')">
                <div class="flashcard-q">{q}</div>
                <div class="flashcard-a">{a}</div>
            </div>
            """
        preview_html += '</div>'
        
        return preview_html, library, format_flashcard_library_html(library)
        
    except Exception as e:
        return f"""
        <div class="empty-state">
            <div class="empty-title">❌ Error</div>
            <div class="empty-subtitle">{str(e)}</div>
        </div>
        """, library, format_flashcard_library_html(library)


def format_flashcard_library_html(library):
    """Format flashcard library as HTML"""
    if not library or len(library) == 0:
        return """
        <div class="empty-state">
            <div class="empty-title">No flashcards in library</div>
            <div class="empty-subtitle">Generate flashcards to add them here</div>
        </div>
        """
    
    html = '''
    <div class="flashcard-section-header">
        <div class="flashcard-instruction">💡 Click on any flashcard to reveal its answer</div>
    </div>
    <div class="flashcard-container">
    '''
    
    for card in library:
        q = card.get("question", "")
        a = card.get("answer", "")
        html += f"""
        <div class="flashcard" onclick="this.classList.toggle('revealed')">
            <div class="flashcard-q">{q}</div>
            <div class="flashcard-a">{a}</div>
        </div>
        """
    html += '</div>'
    return html


def save_settings_handler(tavily_key, hf_token, chunk_size, chunk_overlap, top_k):
    """Save settings - with confirmation"""
    settings = {
        "tavily_api_key": tavily_key,
        "hf_token": hf_token,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "top_k": top_k
    }
    gr.Info("✅ Settings saved successfully!")
    return settings, "✅ Settings saved successfully!"


def reset_settings_handler():
    """Reset settings to defaults - removed theme"""
    settings = {
        "tavily_api_key": TAVILY_API_KEY,
        "hf_token": HF_TOKEN,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k": TOP_K
    }
    return (settings, TAVILY_API_KEY, HF_TOKEN, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, "✅ Reset to defaults!")


def create_app():
    logo_base64 = get_logo_base64()
    logo_tab_base64 = get_logo_tab_base64()
    
    with gr.Blocks(title="Study Buddy", css=custom_css, head=f"""
    <link rel="icon" type="image/png" href="data:image/png;base64,{logo_tab_base64}">
    """) as app:
        # Add navigation JavaScript
        gr.HTML(nav_js)
        
        # State
        chat_history_state = gr.State([])
        flashcard_library_state = gr.State([])
        index_state = gr.State(None)
        chunks_state = gr.State([])
        settings_state = gr.State({
            "tavily_api_key": TAVILY_API_KEY,
            "hf_token": HF_TOKEN,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "top_k": TOP_K
        })
        current_mode = gr.State("local")
        current_inference_mode = gr.State("strict")
        
        with gr.Row(elem_classes="app-wrapper"):
            # Sidebar
            with gr.Column(scale=0, min_width=280, elem_classes="sidebar"):
                gr.HTML(f"""
                    <div class="logo-container">
                        <img src="data:image/png;base64,{logo_base64}" />
                    </div>
                """)
                
                with gr.Group(elem_classes="nav-section"):
                    upload_btn = gr.Button("📤 Upload PDF", elem_id="nav-upload-btn", variant="primary")
                    questions_btn = gr.Button("💬 Ask Questions", elem_id="nav-questions-btn")
                    flashcards_btn = gr.Button("🎴 Flashcards", elem_id="nav-flashcards-btn")
                    history_btn = gr.Button("📋 Chat History", elem_id="nav-history-btn")
                    library_btn = gr.Button("📚 Flashcard Library", elem_id="nav-library-btn")
                
                with gr.Group(elem_classes="settings-section"):
                    settings_btn = gr.Button("⚙️ Settings", elem_id="nav-settings-btn")
            
            # Main content
            with gr.Column(scale=1, elem_classes="main-content"):
                # Upload page
                with gr.Column(visible=True, elem_classes="page-content") as upload_page:
                    gr.HTML("""
                    <div class="page-header">
                        <h1 class="page-title">Upload Your Study Material</h1>
                        <p class="page-description">Upload PDF documents to build a searchable knowledge base</p>
                    </div>
                    """)
                    
                    pdf_file = gr.File(label="PDF Documents", file_types=[".pdf"], file_count="multiple")
                    build_index_btn = gr.Button("Build Index", variant="primary", size="lg")
                    status_text = gr.Textbox(label="Status", interactive=False)
                    status_display = gr.HTML("")
                
                # Questions page
                with gr.Column(visible=False, elem_classes="page-content") as questions_page:
                    gr.HTML("""
                    <div class="page-header">
                        <h1 class="page-title">Ask Questions</h1>
                        <p class="page-description">Query your indexed documents using AI-powered search</p>
                    </div>
                    """)
                    
                    gr.HTML("""
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-bottom: 1rem;">
                        <div>
                            <label style="display: block; margin-bottom: 0.75rem; font-weight: 600;">Search Mode</label>
                            <div class="mode-selector">
                                <div class="mode-btn active" data-group="search" data-value="local" onclick="
                                    document.querySelectorAll('[data-group=\\'search\\']').forEach(b => b.classList.remove('active'));
                                    this.classList.add('active');
                                    document.querySelector('#current-mode textarea').value = 'local';
                                    document.querySelector('#current-mode textarea').dispatchEvent(new Event('input'));
                                ">Local</div>
                                <div class="mode-btn" data-group="search" data-value="web" onclick="
                                    document.querySelectorAll('[data-group=\\'search\\']').forEach(b => b.classList.remove('active'));
                                    this.classList.add('active');
                                    document.querySelector('#current-mode textarea').value = 'web';
                                    document.querySelector('#current-mode textarea').dispatchEvent(new Event('input'));
                                ">Web</div>
                            </div>
                        </div>
                        <div>
                            <label style="display: block; margin-bottom: 0.75rem; font-weight: 600;">Answer Mode</label>
                            <div class="mode-selector">
                                <div class="mode-btn active" data-group="inference" data-value="strict" onclick="
                                    document.querySelectorAll('[data-group=\\'inference\\']').forEach(b => b.classList.remove('active'));
                                    this.classList.add('active');
                                    document.querySelector('#current-inference-mode textarea').value = 'strict';
                                    document.querySelector('#current-inference-mode textarea').dispatchEvent(new Event('input'));
                                ">Strict</div>
                                <div class="mode-btn" data-group="inference" data-value="inference" onclick="
                                    document.querySelectorAll('[data-group=\\'inference\\']').forEach(b => b.classList.remove('active'));
                                    this.classList.add('active');
                                    document.querySelector('#current-inference-mode textarea').value = 'inference';
                                    document.querySelector('#current-inference-mode textarea').dispatchEvent(new Event('input'));
                                ">Inference</div>
                            </div>
                        </div>
                    </div>
                    """)
                    
                    mode_hidden = gr.Textbox(value="local", visible=False, elem_id="current-mode")
                    inference_mode_hidden = gr.Textbox(value="strict", visible=False, elem_id="current-inference-mode")
                    question_input = gr.Textbox(label="Your Question", lines=4, placeholder="What would you like to know?")
                    ask_btn = gr.Button("Ask Question", variant="primary", size="lg")
                    answer_output = gr.HTML("""
                    <div class="empty-state">
                        <div class="empty-title">No answer yet</div>
                        <div class="empty-subtitle">Ask a question to get started</div>
                    </div>
                    """)
                
                # Flashcards page
                with gr.Column(visible=False, elem_classes="page-content") as flashcards_page:
                    gr.HTML("""
                    <div class="page-header">
                        <h1 class="page-title">Generate Flashcards</h1>
                        <p class="page-description">Create study flashcards from your documents</p>
                    </div>
                    """)
                    
                    flashcard_topic = gr.Textbox(
                        label="Topic or Context (Optional)", 
                        lines=3,
                        placeholder="Leave empty to generate flashcards from the entire document..."
                    )
                    num_flashcards = gr.Number(label="Number of Flashcards", value=5, minimum=1, maximum=50)
                    generate_btn = gr.Button("Generate Flashcards", variant="primary", size="lg")
                    flashcard_preview = gr.HTML("""
                    <div class="empty-state">
                        <div class="empty-title">No flashcards yet</div>
                        <div class="empty-subtitle">Generate flashcards to see them here</div>
                    </div>
                    """)
                
                # History page
                with gr.Column(visible=False, elem_classes="page-content") as history_page:
                    gr.HTML("""
                    <div class="page-header">
                        <h1 class="page-title">Chat History</h1>
                        <p class="page-description">Review your past questions and answers</p>
                    </div>
                    """)
                    
                    history_output = gr.HTML("""
                    <div class="empty-state">
                        <div class="empty-title">No chat history yet</div>
                        <div class="empty-subtitle">Ask questions to build your history</div>
                    </div>
                    """)
                
                # Library page
                with gr.Column(visible=False, elem_classes="page-content") as library_page:
                    gr.HTML("""
                    <div class="page-header">
                        <h1 class="page-title">Flashcard Library</h1>
                        <p class="page-description">All your generated flashcards in one place</p>
                    </div>
                    """)
                    
                    library_output = gr.HTML("""
                    <div class="empty-state">
                        <div class="empty-title">No flashcards in library</div>
                        <div class="empty-subtitle">Generate flashcards to add them here</div>
                    </div>
                    """)
                
                # Settings page - cleaned up descriptions
                with gr.Column(visible=False, elem_classes="page-content") as settings_page:
                    gr.HTML("""
                    <div class="page-header">
                        <h1 class="page-title">Settings</h1>
                        <p class="page-description">Configure your Study Buddy</p>
                    </div>
                    """)
                    
                    with gr.Group(elem_classes="setting-group"):
                        gr.HTML('<div class="setting-header">API Keys</div>')
                        tavily_input = gr.Textbox(label="Tavily API Key", type="password", value=TAVILY_API_KEY)
                        hf_input = gr.Textbox(label="HuggingFace Token", type="password", value=HF_TOKEN)
                    
                    with gr.Group(elem_classes="setting-group"):
                        gr.HTML('<div class="setting-header">Chunking Settings</div>')
                        chunk_size_input = gr.Number(label="Chunk Size", value=CHUNK_SIZE, minimum=128, maximum=2048)
                        chunk_overlap_input = gr.Number(label="Chunk Overlap", value=CHUNK_OVERLAP, minimum=0, maximum=500)
                    
                    with gr.Group(elem_classes="setting-group"):
                        gr.HTML('<div class="setting-header">Retrieval Settings</div>')
                        top_k_input = gr.Number(label="Top K Results", value=TOP_K, minimum=1, maximum=20)
                    
                    with gr.Row():
                        save_btn = gr.Button("Save Settings", variant="primary")
                        reset_btn = gr.Button("Reset to Defaults", variant="secondary")
        
        # Navigation with active state management
        pages = [upload_page, questions_page, flashcards_page, history_page, library_page, settings_page]
        nav_buttons = [upload_btn, questions_btn, flashcards_btn, history_btn, library_btn, settings_btn]
        
        def show_page(page_name, index, chunks):
            # Navigation Guard
            if page_name in ["questions", "flashcards", "history", "library"]:
                if index is None or not chunks:
                    gr.Warning("⚠️ Please upload a PDF and build the index first!")
                    page_name = "upload" # Force stay on upload page

            page_visible = [
                gr.update(visible=(page_name == "upload")),
                gr.update(visible=(page_name == "questions")),
                gr.update(visible=(page_name == "flashcards")),
                gr.update(visible=(page_name == "history")),
                gr.update(visible=(page_name == "library")),
                gr.update(visible=(page_name == "settings"))
            ]
            # Button styles: active button gets primary variant, others get secondary
            button_styles = [
                gr.update(variant="primary" if page_name == "upload" else "secondary"),
                gr.update(variant="primary" if page_name == "questions" else "secondary"),
                gr.update(variant="primary" if page_name == "flashcards" else "secondary"),
                gr.update(variant="primary" if page_name == "history" else "secondary"),
                gr.update(variant="primary" if page_name == "library" else "secondary"),
                gr.update(variant="primary" if page_name == "settings" else "secondary")
            ]
            return page_visible + button_styles
        
        all_outputs = pages + nav_buttons
        
        upload_btn.click(
            fn=show_page,
            inputs=[gr.State("upload"), index_state, chunks_state],
            outputs=all_outputs
        )
        
        questions_btn.click(
            fn=show_page,
            inputs=[gr.State("questions"), index_state, chunks_state],
            outputs=all_outputs
        )
        
        flashcards_btn.click(
            fn=show_page,
            inputs=[gr.State("flashcards"), index_state, chunks_state],
            outputs=all_outputs
        )
        
        history_btn.click(
            fn=show_page,
            inputs=[gr.State("history"), index_state, chunks_state],
            outputs=all_outputs
        ).then(format_chat_history_html, inputs=[chat_history_state], outputs=[history_output])
        
        library_btn.click(
            fn=show_page,
            inputs=[gr.State("library"), index_state, chunks_state],
            outputs=all_outputs
        ).then(format_flashcard_library_html, inputs=[flashcard_library_state], outputs=[library_output])
        
        settings_btn.click(
            fn=show_page,
            inputs=[gr.State("settings"), index_state, chunks_state],
            outputs=all_outputs
        )
        
        # Functionality
        build_index_btn.click(
            build_index_handler,
            inputs=[pdf_file, settings_state],
            outputs=[status_text, index_state, chunks_state, status_display]
        )
        
        ask_btn.click(
            ask_question_handler,
            inputs=[question_input, mode_hidden, inference_mode_hidden, index_state, chunks_state, settings_state, chat_history_state],
            outputs=[question_input, chat_history_state, answer_output]
        )
        
        generate_btn.click(
            generate_flashcards_handler,
            inputs=[flashcard_topic, num_flashcards, mode_hidden, index_state, chunks_state, settings_state, flashcard_library_state],
            outputs=[flashcard_preview, flashcard_library_state, library_output]
        )
        
        save_btn.click(
            save_settings_handler,
            inputs=[tavily_input, hf_input, chunk_size_input, chunk_overlap_input, top_k_input],
            outputs=[settings_state, status_text]
        )
        
        reset_btn.click(
            reset_settings_handler,
            outputs=[settings_state, tavily_input, hf_input, chunk_size_input, chunk_overlap_input, top_k_input, status_text]
        )
    
    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(share=True, debug=True)