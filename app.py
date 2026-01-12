import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import json
import pickle
from datetime import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS  
from langchain_groq import ChatGroq

# -------------------- LOAD ENV --------------------
load_dotenv()

# -------------------- CONSTANTS --------------------
THEME_FILE = "theme_preference.json"
DATA_FOLDER = "data"
SESSIONS_FOLDER = os.path.join(DATA_FOLDER, "sessions")
VECTORSTORE_FOLDER = os.path.join(DATA_FOLDER, "vectorstore")
CHUNKS_FILE = os.path.join(DATA_FOLDER, "text_chunks.pkl")
METADATA_FILE = os.path.join(DATA_FOLDER, "metadata.json")
CURRENT_SESSION_FILE = os.path.join(DATA_FOLDER, "current_session.json")

# Create folders if they don't exist
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(SESSIONS_FOLDER, exist_ok=True)
os.makedirs(VECTORSTORE_FOLDER, exist_ok=True)

# -------------------- SESSION MANAGEMENT --------------------
def create_new_session():
    """Create a new session"""
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_data = {
        'session_id': session_id,
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'messages': []
    }
    return session_data

def save_session(session_data):
    """Save session to file"""
    try:
        session_id = session_data['session_id']
        session_file = os.path.join(SESSIONS_FOLDER, f"{session_id}.json")
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
    except Exception as e:
        st.warning(f"Could not save session: {e}")

def load_session(session_id):
    """Load a specific session"""
    try:
        session_file = os.path.join(SESSIONS_FOLDER, f"{session_id}.json")
        if os.path.exists(session_file):
            with open(session_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"Could not load session: {e}")
    return None

def get_all_sessions():
    """Get all saved sessions"""
    sessions = []
    try:
        if os.path.exists(SESSIONS_FOLDER):
            for filename in os.listdir(SESSIONS_FOLDER):
                if filename.endswith('.json'):
                    session_file = os.path.join(SESSIONS_FOLDER, filename)
                    with open(session_file, 'r') as f:
                        session_data = json.load(f)
                        sessions.append(session_data)
        # Sort by created_at descending (newest first)
        sessions.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    except Exception as e:
        st.warning(f"Could not load sessions: {e}")
    return sessions

def save_current_session_id(session_id):
    """Save the current active session ID"""
    try:
        with open(CURRENT_SESSION_FILE, 'w') as f:
            json.dump({'current_session_id': session_id}, f)
    except Exception as e:
        st.warning(f"Could not save current session ID: {e}")

def load_current_session_id():
    """Load the current active session ID"""
    try:
        if os.path.exists(CURRENT_SESSION_FILE):
            with open(CURRENT_SESSION_FILE, 'r') as f:
                data = json.load(f)
                return data.get('current_session_id')
    except Exception as e:
        pass
    return None

def delete_session(session_id):
    """Delete a specific session"""
    try:
        session_file = os.path.join(SESSIONS_FOLDER, f"{session_id}.json")
        if os.path.exists(session_file):
            os.remove(session_file)
            return True
    except Exception as e:
        st.error(f"Could not delete session: {e}")
    return False

# -------------------- THEME PERSISTENCE --------------------
def save_theme(theme):
    """Save theme preference to local file"""
    try:
        with open(THEME_FILE, 'w') as f:
            json.dump({'theme': theme}, f)
    except Exception as e:
        st.warning(f"Could not save theme preference: {e}")

def load_theme():
    """Load theme preference from local file"""
    try:
        if os.path.exists(THEME_FILE):
            with open(THEME_FILE, 'r') as f:
                data = json.load(f)
                return data.get('theme', 'light')
    except Exception as e:
        st.warning(f"Could not load theme preference: {e}")
    return 'light'

# -------------------- DATA PERSISTENCE --------------------
def save_chunks(text_chunks):
    """Save text chunks to local file"""
    try:
        with open(CHUNKS_FILE, 'wb') as f:
            pickle.dump(text_chunks, f)
        return True
    except Exception as e:
        st.error(f"Could not save chunks: {e}")
        return False

def load_chunks():
    """Load text chunks from local file"""
    try:
        if os.path.exists(CHUNKS_FILE):
            with open(CHUNKS_FILE, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        st.warning(f"Could not load chunks: {e}")
    return None

def save_vectorstore(vectorstore):
    """Save FAISS vectorstore to local folder"""
    try:
        vectorstore.save_local(VECTORSTORE_FOLDER)
        return True
    except Exception as e:
        st.error(f"Could not save vectorstore: {e}")
        return False

def load_vectorstore():
    """Load FAISS vectorstore from local folder"""
    try:
        index_file = os.path.join(VECTORSTORE_FOLDER, "index.faiss")
        if os.path.exists(index_file):
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            vectorstore = FAISS.load_local(
                VECTORSTORE_FOLDER, 
                embeddings,
                allow_dangerous_deserialization=True
            )
            return vectorstore
    except Exception as e:
        st.warning(f"Could not load vectorstore: {e}")
    return None

def save_metadata(pdf_names, chunk_count):
    """Save metadata about processed PDFs"""
    try:
        metadata = {
            'pdf_names': pdf_names,
            'chunk_count': chunk_count,
            'processed_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
        return True
    except Exception as e:
        st.warning(f"Could not save metadata: {e}")
        return False

def load_metadata():
    """Load metadata about processed PDFs"""
    try:
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"Could not load metadata: {e}")
    return None

def clear_all_data():
    """Clear all stored data including all sessions"""
    try:
        if os.path.exists(CHUNKS_FILE):
            os.remove(CHUNKS_FILE)
        if os.path.exists(CURRENT_SESSION_FILE):
            os.remove(CURRENT_SESSION_FILE)
        for file in os.listdir(SESSIONS_FOLDER):
            os.remove(os.path.join(SESSIONS_FOLDER, file))
        for file in os.listdir(VECTORSTORE_FOLDER):
            os.remove(os.path.join(VECTORSTORE_FOLDER, file))
        if os.path.exists(METADATA_FILE):
            os.remove(METADATA_FILE)
        return True
    except Exception as e:
        st.error(f"Could not clear data: {e}")
        return False

# -------------------- PDF TEXT EXTRACTION --------------------
def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF documents"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf) 
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# -------------------- TEXT CHUNKING --------------------
def get_text_chunks(raw_text):
    """Split text into smaller chunks for embedding"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(raw_text)

# -------------------- VECTOR STORE --------------------
def get_vectorstore(text_chunks):
    if not text_chunks:
        raise ValueError("No text chunks to embed")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# -------------------- ANSWER GENERATION --------------------
def generate_answer(vectorstore, question):
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY not found in environment variables")
    
    relevant_docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3
    )
    
    prompt = f"""You are a helpful assistant that answers questions based on the provided context from PDF documents.

Context:
{context}

Question: {question}

Answer:"""
    
    response = llm.invoke(prompt)
    return {'result': response.content, 'source_documents': relevant_docs}

# -------------------- THEME HANDLER --------------------
def apply_theme(theme):
    transition_css = """
    .stApp,
    section[data-testid="stSidebar"],
    header[data-testid="stHeader"],
    input,
    textarea,
    .stButton > button {
        transition:
            background-color 0.25s ease,
            color 0.25s ease,
            border-color 0.25s ease,
            box-shadow 0.25s ease;
    }
    """

    if theme == "pink":
        st.markdown(f"""
        <style>
        {transition_css}
        .stApp {{ background-color:#fff1f5 !important; color:#3b0a1a !important; }}
        header[data-testid="stHeader"] {{ background-color:#ffe4ec !important; border-bottom:1px solid #f4b6c2 !important; }}
        section[data-testid="stSidebar"] {{ background-color:#ffe4ec !important; }}
        h1,h2,h3,h4,h5,h6,p,span,label {{ color:#3b0a1a !important; }}
        input,textarea {{ background-color:#fff7fa !important; color:#3b0a1a !important; border-radius:8px !important; border:1px solid #f4b6c2 !important; }}
        div[data-testid="stFileUploader"] {{ background-color:#fff7fa !important; border:1px dashed #f4b6c2 !important; border-radius:10px !important; padding:10px !important; }}
        .stButton > button {{ background-color:#ec407a !important; color:white !important; border-radius:8px !important; }}
        .stButton > button:hover {{ transform:translateY(-1px); box-shadow:0 6px 14px rgba(236,64,122,0.35); }}
        div[data-testid="stExpander"] {{ background-color:#fff7fa !important; border-radius:10px !important; border:1px solid #f4b6c2 !important; }}
        </style>
        """, unsafe_allow_html=True)
    else:  # light
        st.markdown(f"""
        <style>
        {transition_css}
        .stApp {{ background-color:#ffffff !important; color:#000000 !important; }}
        header[data-testid="stHeader"] {{ background-color:#ffffff !important; border-bottom:1px solid #ddd !important; }}
        section[data-testid="stSidebar"] {{ background-color:#f5f7fb !important; }}
        h1,h2,h3,h4,h5,h6,p,span,label {{ color:#000000 !important; }}
        input,textarea {{ background-color:#f0f2f6 !important; color:#000000 !important; border-radius:8px !important; border:1px solid #ccc !important; }}
        div[data-testid="stFileUploader"] {{ background-color:#ffffff !important; border:1px dashed #ccc !important; border-radius:10px !important; padding:10px !important; }}
        .stButton > button {{ background-color:#1976d2 !important; color:white !important; border-radius:8px !important; }}
        .stButton > button:hover {{ transform:translateY(-1px); box-shadow:0 6px 14px rgba(0,0,0,0.15); }}
        div[data-testid="stExpander"] {{ background-color:#ffffff !important; border-radius:10px !important; border:1px solid #ddd !important; }}
        </style>
        """, unsafe_allow_html=True)

# -------------------- MAIN APP --------------------
def main():
    st.set_page_config(page_title="Chat with PDFs", page_icon="📚", layout="wide")

    # Initialize current session
    if "current_session" not in st.session_state:
        # Try to load the last active session
        session_id = load_current_session_id()
        if session_id:
            session_data = load_session(session_id)
            if session_data:
                st.session_state.current_session = session_data
            else:
                st.session_state.current_session = create_new_session()
        else:
            st.session_state.current_session = create_new_session()
        save_current_session_id(st.session_state.current_session['session_id'])

    # Initialize scroll target
    if "scroll_to_index" not in st.session_state:
        st.session_state.scroll_to_index = None

    # Load saved theme on startup
    if "theme" not in st.session_state:
        st.session_state.theme = load_theme()

    # Load saved vectorstore on startup if available
    if "vectorstore" not in st.session_state:
        loaded_vectorstore = load_vectorstore()
        if loaded_vectorstore:
            st.session_state.vectorstore = loaded_vectorstore

    apply_theme(st.session_state.theme)

    # -------------------- TOP BAR --------------------
    col1, col2, col3, col4 = st.columns([5, 1, 1, 1])
    
    with col1:
        st.header("Chat with multiple PDFs 📚")
    
    with col2:
        if st.button("➕ New Chat", use_container_width=True):
            # Save current session before creating new one
            save_session(st.session_state.current_session)
            # Create new session
            st.session_state.current_session = create_new_session()
            save_current_session_id(st.session_state.current_session['session_id'])
            st.session_state.scroll_to_index = None
            st.rerun()
    
    with col3:
        theme_toggle = st.toggle(
            "🎨",
            value=(st.session_state.theme == "pink"),
            help="Toggle theme: Light/Pink",
            key="theme_toggle"
        )
        
        new_theme = "pink" if theme_toggle else "light"
        if new_theme != st.session_state.theme:
            st.session_state.theme = new_theme
            save_theme(new_theme)
            st.rerun()

    if not os.getenv("GROQ_API_KEY"):
        st.error("⚠️ GROQ_API_KEY not found in .env file. Please add it to use answer generation.")
        st.stop()

    # -------------------- SIDEBAR --------------------
    with st.sidebar:
        # Single-line Session Display
        metadata = load_metadata()
        if metadata:
            pdf_list = ', '.join(metadata['pdf_names'])
            st.markdown(f"**📁 Files:** {pdf_list} | **Chunks:** {metadata['chunk_count']}")
        
        st.divider()
        
        # Action Buttons
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("🗑️ Delete Chat", use_container_width=True, key="clear_chat_btn"):
                session_id = st.session_state.current_session['session_id']
                if delete_session(session_id):
                    st.session_state.current_session = create_new_session()
                    save_current_session_id(st.session_state.current_session['session_id'])
                    st.session_state.scroll_to_index = None
                    st.rerun()
        
        with col_b:
            if st.button("🔄 Clear All", use_container_width=True, key="clear_all_btn"):
                if clear_all_data():
                    if "vectorstore" in st.session_state:
                        del st.session_state.vectorstore
                    st.session_state.current_session = create_new_session()
                    save_current_session_id(st.session_state.current_session['session_id'])
                    st.session_state.scroll_to_index = None
                    st.rerun()
        
        st.divider()
        
        # All Sessions History
        all_sessions = get_all_sessions()
        if all_sessions:
            st.subheader("💬 Chat History")
            st.caption(f"📝 {len(all_sessions)} total sessions")
            
            for session in all_sessions:
                session_id = session['session_id']
                messages = session.get('messages', [])
                user_messages = [msg for msg in messages if msg["role"] == "user"]
                
                # Get first question as preview
                if user_messages:
                    first_q = user_messages[0]['content'][:30] + "..." if len(user_messages[0]['content']) > 30 else user_messages[0]['content']
                    label = f"{first_q} ({len(user_messages)} Q)"
                else:
                    label = f"Empty session"
                
                # Highlight current session
                is_current = session_id == st.session_state.current_session['session_id']
                button_label = f"▶ {label}" if is_current else label
                
                if st.button(
                    button_label,
                    key=f"session_{session_id}",
                    use_container_width=True,
                    type="primary" if is_current else "secondary"
                ):
                    if not is_current:
                        # Save current session before switching
                        save_session(st.session_state.current_session)
                        # Load the clicked session
                        st.session_state.current_session = session
                        save_current_session_id(session_id)
                        st.session_state.scroll_to_index = None
                        st.rerun()
        else:
            st.info("💬 No chat history yet")
        
        st.divider()

        # Upload Section
        upload_expander = st.expander("📄 Upload PDFs", expanded=False)
        with upload_expander:
            pdf_docs = st.file_uploader(
                "Select PDF files",
                accept_multiple_files=True,
                type=["pdf"],
                label_visibility="collapsed",
                key="pdf_uploader"
            )

            if st.button("🚀 Process PDFs", use_container_width=True, key="process_button"):
                if not pdf_docs:
                    st.toast("⚠️ Please upload at least one PDF.", icon="⚠️")
                else:
                    with st.spinner("Processing PDFs..."):
                        raw_text = get_pdf_text(pdf_docs)
                        if not raw_text.strip():
                            st.toast("❌ No readable text found in the PDFs.", icon="❌")
                        else:
                            text_chunks = get_text_chunks(raw_text)
                            if not text_chunks:
                                st.toast("❌ Failed to split text into chunks.", icon="❌")
                            else:
                                vectorstore = get_vectorstore(text_chunks)
                                st.session_state.vectorstore = vectorstore
                                pdf_names = [pdf.name for pdf in pdf_docs]
                                save_chunks(text_chunks)
                                save_vectorstore(vectorstore)
                                save_metadata(pdf_names, len(text_chunks))
                                st.toast("✅ PDFs processed successfully!", icon="✅")
                                st.rerun()
    
    # -------------------- MAIN CHAT AREA --------------------
    current_messages = st.session_state.current_session.get('messages', [])
    
    if not current_messages:
        st.info("👋 Welcome! Upload PDFs and start asking questions. Your conversation will be saved automatically.")
    
    # Display all messages from current session
    for idx, message in enumerate(current_messages):
        is_highlighted = st.session_state.scroll_to_index == idx
        
        if is_highlighted:
            st.markdown(f"""
            <div style="border: 2px solid {'#ec407a' if st.session_state.theme == 'pink' else '#1976d2'}; 
                 border-radius: 10px; padding: 5px; margin: 5px 0; 
                 background-color: {'#fff7fa' if st.session_state.theme == 'pink' else '#e3f2fd'};">
            """, unsafe_allow_html=True)
            st.session_state.scroll_to_index = None
        
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "source_docs" in message:
                with st.expander("📄 Source Chunks Used"):
                    for i, content in enumerate(message["source_docs"], 1):
                        st.write(f"**Chunk {i}:**")
                        st.write(content)
                        st.write("---")
        
        if is_highlighted:
            st.markdown("</div>", unsafe_allow_html=True)

    # Chat Input
    user_question = st.chat_input("Type your question here...")
    
    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)
        
        st.session_state.current_session['messages'].append({"role": "user", "content": user_question})
        save_session(st.session_state.current_session)

        if "vectorstore" not in st.session_state:
            with st.chat_message("assistant"):
                warning_msg = "⚠️ Please upload and process PDFs first."
                st.warning(warning_msg)
                st.session_state.current_session['messages'].append({
                    "role": "assistant", 
                    "content": warning_msg
                })
                save_session(st.session_state.current_session)
        else:
            with st.chat_message("assistant"):
                with st.spinner("Generating answer..."):
                    try:
                        result = generate_answer(st.session_state.vectorstore, user_question)
                        st.markdown(result['result'])
                        source_contents = [doc.page_content for doc in result['source_documents']]
                        with st.expander("📄 Source Chunks Used"):
                            for i, content in enumerate(source_contents, 1):
                                st.write(f"**Chunk {i}:**")
                                st.write(content)
                                st.write("---")
                        
                        st.session_state.current_session['messages'].append({
                            "role": "assistant", 
                            "content": result['result'],
                            "source_docs": source_contents
                        })
                        save_session(st.session_state.current_session)
                    except Exception as e:
                        error_msg = f"Error generating answer: {str(e)}"
                        st.error(error_msg)
                        st.session_state.current_session['messages'].append({
                            "role": "assistant", 
                            "content": error_msg
                        })
                        save_session(st.session_state.current_session)

# -------------------- RUN --------------------
if __name__ == "__main__":
    main()