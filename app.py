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
VECTORSTORE_FOLDER = os.path.join(DATA_FOLDER, "vectorstore")
CHUNKS_FILE = os.path.join(DATA_FOLDER, "text_chunks.pkl")
METADATA_FILE = os.path.join(DATA_FOLDER, "metadata.json")
# NEW: Added for history persistence
CHAT_HISTORY_FILE = os.path.join(DATA_FOLDER, "chat_history.json")

# Create data folder if it doesn't exist
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(VECTORSTORE_FOLDER, exist_ok=True)

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

# -------------------- NEW: CHAT HISTORY PERSISTENCE --------------------
def save_chat_history(history):
    """Save chat history to local file"""
    try:
        with open(CHAT_HISTORY_FILE, 'w') as f:
            json.dump(history, f)
    except Exception as e:
        pass

def load_chat_history():
    """Load chat history from local file"""
    try:
        if os.path.exists(CHAT_HISTORY_FILE):
            with open(CHAT_HISTORY_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        pass
    return []

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
    """Clear all stored data"""
    try:
        if os.path.exists(CHUNKS_FILE):
            os.remove(CHUNKS_FILE)
        if os.path.exists(CHAT_HISTORY_FILE):
            os.remove(CHAT_HISTORY_FILE)
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


# -------------------- SIMILARITY SEARCH --------------------
def search_relevant_chunks(vectorstore, question, k=3):
    relevant_docs = vectorstore.similarity_search(question, k=k)
    return [doc.page_content for doc in relevant_docs]


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
    st.set_page_config(page_title="Chat with PDFs", page_icon="📚")

    # Load saved theme on startup
    if "theme" not in st.session_state:
        st.session_state.theme = load_theme()

    # Load saved vectorstore on startup if available
    if "vectorstore" not in st.session_state:
        loaded_vectorstore = load_vectorstore()
        if loaded_vectorstore:
            st.session_state.vectorstore = loaded_vectorstore

    # NEW: Load saved chat history on startup
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_chat_history()

    apply_theme(st.session_state.theme)

    # -------------------- THEME TOGGLE AT TOP --------------------
    col1, col2, col3 = st.columns([6, 1, 1])
    
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
    
    with col1:
        st.header("Chat with multiple PDFs 📚")

    if not os.getenv("GROQ_API_KEY"):
        st.error("⚠️ GROQ_API_KEY not found in .env file. Please add it to use answer generation.")
        st.stop()

    # -------------------- SIDEBAR --------------------
    metadata = load_metadata()
    if metadata:
        with st.sidebar:
            st.success("📁 Data loaded from storage")
            with st.expander("ℹ️ Loaded Data Info"):
                st.write(f"**PDFs:** {', '.join(metadata['pdf_names'])}")
                st.write(f"**Chunks:** {metadata['chunk_count']}")
                st.write(f"**Processed:** {metadata['processed_date']}")
            
            # Sidebar History List (NEW)
            st.subheader("🕒 History")
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.write(f"• {msg['content'][:25]}...")

            if st.button("🗑️ Clear Stored Data"):
                if clear_all_data():
                    if "vectorstore" in st.session_state:
                        del st.session_state.vectorstore
                    st.session_state.chat_history = [] 
                    st.success("✅ All data cleared!")
                    st.rerun()

    with st.sidebar:
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
                    processing_placeholder = st.empty()
                    if st.session_state.theme == "pink":
                        processing_placeholder.markdown("""
                        <div style="padding: 10px;">
                            <div style="background: linear-gradient(90deg, #ffe4ec 0%, #ffc9d9 50%, #ffe4ec 100%); 
                                        background-size: 200% 100%; 
                                        animation: shimmer 1.5s infinite;
                                        height: 20px; 
                                        border-radius: 8px; 
                                        margin-bottom: 10px;">
                            </div>
                            <div style="background: linear-gradient(90deg, #ffe4ec 0%, #ffc9d9 50%, #ffe4ec 100%); 
                                        background-size: 200% 100%; 
                                        animation: shimmer 1.5s infinite;
                                        height: 20px; 
                                        border-radius: 8px;">
                            </div>
                        </div>
                        <style>
                        @keyframes shimmer {
                            0% { background-position: 200% 0; }
                            100% { background-position: -200% 0; }
                        }
                        </style>
                        """, unsafe_allow_html=True)
                        
                        raw_text = get_pdf_text(pdf_docs)
                        if not raw_text.strip():
                            processing_placeholder.empty()
                            st.toast("❌ No readable text found in the PDFs.", icon="❌")
                        else:
                            text_chunks = get_text_chunks(raw_text)
                            if not text_chunks:
                                processing_placeholder.empty()
                                st.toast("❌ Failed to split text into chunks.", icon="❌")
                            else:
                                vectorstore = get_vectorstore(text_chunks)
                                st.session_state.vectorstore = vectorstore
                                pdf_names = [pdf.name for pdf in pdf_docs]
                                save_chunks(text_chunks)
                                save_vectorstore(vectorstore)
                                save_metadata(pdf_names, len(text_chunks))
                                processing_placeholder.empty()
                                st.toast("✅ PDFs processed and saved successfully!", icon="✅")
                                st.rerun()
                    else:
                        with processing_placeholder:
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
                                        processing_placeholder.empty()
                                        st.toast("✅ PDFs processed and saved successfully!", icon="✅")
                                        st.rerun()
    
    # -------------------- MAIN CHAT AREA --------------------
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "source_docs" in message:
                with st.expander("📄 Source Chunks Used"):
                    for i, content in enumerate(message["source_docs"], 1):
                        st.write(f"**Chunk {i}:**")
                        st.write(content)
                        st.write("---")

    user_question = st.chat_input("Type your question here...")
    
    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)
        
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        save_chat_history(st.session_state.chat_history) # Persistent Save

        if "vectorstore" not in st.session_state:
            with st.chat_message("assistant"):
                st.warning("⚠️ Please upload and process PDFs first.")
        else:
            with st.chat_message("assistant"):
                if st.session_state.theme == "pink":
                    skeleton_placeholder = st.empty()
                    skeleton_placeholder.markdown("""
                    <div style="padding: 20px;">
                        <div style="background: linear-gradient(90deg, #ffe4ec 0%, #ffc9d9 50%, #ffe4ec 100%); 
                                    background-size: 200% 100%; 
                                    animation: shimmer 1.5s infinite;
                                    height: 40px; 
                                    border-radius: 8px; 
                                    margin-bottom: 10px;">
                        </div>
                        <div style="background: linear-gradient(90deg, #ffe4ec 0%, #ffc9d9 50%, #ffe4ec 100%); 
                                    background-size: 200% 100%; 
                                    animation: shimmer 1.5s infinite;
                                    height: 100px; 
                                    border-radius: 8px;">
                        </div>
                    </div>
                    <style>
                    @keyframes shimmer {
                        0% { background-position: 200% 0; }
                        100% { background-position: -200% 0; }
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    try:
                        result = generate_answer(st.session_state.vectorstore, user_question)
                        skeleton_placeholder.empty()
                        st.markdown(result['result'])
                        source_contents = [doc.page_content for doc in result['source_documents']]
                        with st.expander("📄 Source Chunks Used"):
                            for i, content in enumerate(source_contents, 1):
                                st.write(f"**Chunk {i}:**")
                                st.write(content)
                                st.write("---")
                        
                        st.session_state.chat_history.append({
                            "role": "assistant", 
                            "content": result['result'],
                            "source_docs": source_contents
                        })
                        save_chat_history(st.session_state.chat_history) # Persistent Save

                    except Exception as e:
                        skeleton_placeholder.empty()
                        st.error(f"Error generating answer: {str(e)}")
                
                else:
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
                            
                            st.session_state.chat_history.append({
                                "role": "assistant", 
                                "content": result['result'],
                                "source_docs": source_contents
                            })
                            save_chat_history(st.session_state.chat_history) # Persistent Save
                        except Exception as e:
                            st.error(f"Error generating answer: {str(e)}")


# -------------------- RUN --------------------
if __name__ == "__main__":
    main()