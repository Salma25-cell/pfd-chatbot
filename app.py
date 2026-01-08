import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS  
from langchain_groq import ChatGroq

# -------------------- LOAD ENV --------------------
load_dotenv()

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
    """
    Create vector store using FREE HuggingFace embeddings
    
    CHANGED: Now using HuggingFaceEmbeddings instead of GoogleGenerativeAIEmbeddings
    - Completely free (no API costs)
    - Runs locally (no rate limits)
    - No API key required
    - Model downloads once (~90MB) then cached
    """
    if not text_chunks:
        raise ValueError("No text chunks to embed")

    # Create embeddings using HuggingFace model (runs on your machine)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Lightweight, fast model
        model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have GPU for faster processing
        encode_kwargs={'normalize_embeddings': True}  # Better similarity search results
    )

    # Create FAISS vector store from text chunks
    vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )

    return vectorstore


# -------------------- SIMILARITY SEARCH --------------------
def search_relevant_chunks(vectorstore, question, k=3):
    """
    Find the most relevant chunks for a question 
    Parameters:
    - vectorstore: the FAISS database with all chunks
    - question: user's question as a string
    - k: how many chunks to retrieve (default: 3)

    Returns:
    - list of relevant text chunks
    """
    # Search for similar chunks 
    # This converts the question to numbers and finds matching chunks
    relevant_docs = vectorstore.similarity_search(question, k=k)

    # Extract just the text from the results
    relevant_texts = [doc.page_content for doc in relevant_docs]

    return relevant_texts


# -------------------- ANSWER GENERATION --------------------
def generate_answer(vectorstore, question):
    """
    Generate an answer using retrieved chunks + LLM (RAG)
    Simple approach without chains - easier to debug and more reliable
    
    Parameters:
    - vectorstore: FAISS vector store with document chunks
    - question: user's question
    
    Returns:
    - dictionary with 'result' (answer) and 'source_documents' (chunks used)
    """
    # Check if API key exists
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY not found in environment variables")
    
    # Step 1: Get relevant document chunks
    relevant_docs = vectorstore.similarity_search(question, k=3)
    
    # Step 2: Combine chunks into context
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Step 3: Initialize Groq LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",  # Fast and free model
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3  # Lower = more focused, higher = more creative
    )
    
    # Step 4: Create prompt
    prompt = f"""You are a helpful assistant that answers questions based on the provided context from PDF documents.

Use the following context to answer the question. 
If you cannot find the answer in the context, say "I cannot find that information in the provided documents."
Be concise and specific in your answer.

Context:
{context}

Question: {question}

Answer:"""
    
    # Step 5: Get response from LLM
    response = llm.invoke(prompt)
    
    # Return in same format as before for compatibility
    return {
        'result': response.content,
        'source_documents': relevant_docs
    }
# -------------------- THEME HANDLER --------------------
# -------------------- FULL THEME HANDLER --------------------
def apply_theme(theme):
    if theme == "dark":
        st.markdown("""
        <style>
        .stApp {
            background-color: #0d1117 !important;
            color: #e6edf3 !important;
        }

        header[data-testid="stHeader"] {
            background-color: #0d1117 !important;
            border-bottom: 1px solid #30363d !important;
        }

        section[data-testid="stSidebar"] {
            background-color: #161b22 !important;
        }

        h1, h2, h3, h4, h5, h6, p, span, label {
            color: #e6edf3 !important;
        }

        input, textarea {
            background-color: #21262d !important;
            color: #e6edf3 !important;
            border-radius: 8px !important;
            border: 1px solid #30363d !important;
        }

        div[data-testid="stFileUploader"] {
            background-color: #161b22 !important;
            border: 1px dashed #30363d !important;
            border-radius: 10px !important;
            padding: 10px !important;
        }

        .stButton > button {
            background-color: #238636 !important;
            color: white !important;
            border-radius: 8px !important;
        }

        div[data-testid="stExpander"] {
            background-color: #161b22 !important;
            border: 1px solid #30363d !important;
            border-radius: 10px !important;
        }
        </style>
        """, unsafe_allow_html=True)

    elif theme == "pink":
        st.markdown("""
        <style>
        .stApp {
            background-color: #fff1f5 !important;
            color: #3b0a1a !important;
        }

        header[data-testid="stHeader"] {
            background-color: #ffe4ec !important;
            border-bottom: 1px solid #f4b6c2 !important;
        }

        section[data-testid="stSidebar"] {
            background-color: #ffe4ec !important;
        }

        h1, h2, h3, h4, h5, h6, p, span, label {
            color: #3b0a1a !important;
        }

        input, textarea {
            background-color: #fff7fa !important;
            color: #3b0a1a !important;
            border-radius: 8px !important;
            border: 1px solid #f4b6c2 !important;
        }

        div[data-testid="stFileUploader"] {
            background-color: #fff7fa !important;
            border: 1px dashed #f4b6c2 !important;
            border-radius: 10px !important;
            padding: 10px !important;
        }

        .stButton > button {
            background-color: #ec407a !important;
            color: white !important;
            border-radius: 8px !important;
        }

        .stButton > button:hover {
            background-color: #d81b60 !important;
        }

        div[data-testid="stExpander"] {
            background-color: #fff7fa !important;
            border-radius: 10px !important;
            border: 1px solid #f4b6c2 !important;
        }
        </style>
        """, unsafe_allow_html=True)

    else:  # LIGHT
        st.markdown("""
        <style>
        .stApp {
            background-color: #ffffff !important;
            color: #000000 !important;
        }

        header[data-testid="stHeader"] {
            background-color: #ffffff !important;
            border-bottom: 1px solid #ddd !important;
        }

        section[data-testid="stSidebar"] {
            background-color: #f5f7fb !important;
        }

        h1, h2, h3, h4, h5, h6, p, span, label {
            color: #000000 !important;
        }

        input, textarea {
            background-color: #f0f2f6 !important;
            color: #000000 !important;
            border-radius: 8px !important;
            border: 1px solid #ccc !important;
        }

        div[data-testid="stFileUploader"] {
            background-color: #ffffff !important;
            border: 1px dashed #ccc !important;
            border-radius: 10px !important;
            padding: 10px !important;
        }

        .stButton > button {
            background-color: #1976d2 !important;
            color: white !important;
            border-radius: 8px !important;
        }

        div[data-testid="stExpander"] {
            background-color: #ffffff !important;
            border-radius: 10px !important;
            border: 1px solid #ddd !important;
        }
        </style>
        """, unsafe_allow_html=True)


# -------------------- MAIN APP --------------------
def main():
        # ---------- THEME TOGGLE ----------

    st.set_page_config(
        page_title="Chat with PDFs",
        page_icon="📚"
    )
    if "theme" not in st.session_state:
         st.session_state.theme = "light"

    with st.sidebar:
        st.markdown("## 🎨 Theme")
        theme_choice = st.radio(
        "Select theme",
        ["Light", "Dark", "Pink"]
        )

    st.session_state.theme = theme_choice.lower()
    apply_theme(st.session_state.theme)

    st.header("Chat with multiple PDFs 📚")
    
    # Info box
    st.info("💡 Using free local HuggingFace embeddings + Groq LLM for answers")

    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        st.error("⚠️ GROQ_API_KEY not found in .env file. Please add it to use answer generation.")
        st.stop()

    user_question = st.text_input("Ask a question about your documents:")

    with st.sidebar:
        st.subheader("Your documents")

        pdf_docs = st.file_uploader(
            "Upload PDFs and click Process",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF.")
                st.stop()

            with st.spinner("Reading and indexing PDFs..."):
                # Step 1: Extract text from PDFs
                raw_text = get_pdf_text(pdf_docs)

                if not raw_text.strip():
                    st.error("No readable text found in the PDFs.")
                    st.stop()

                # Step 2: Split text into chunks
                text_chunks = get_text_chunks(raw_text)

                if not text_chunks:
                    st.error("Failed to split text into chunks.")
                    st.stop()

                st.write(f"✅ Total chunks: {len(text_chunks)}")

                # Step 3: Create vector store with embeddings
                with st.spinner("Creating embeddings (first run may take a moment)..."):
                    vectorstore = get_vectorstore(text_chunks)
                
                # Store in session state for later use
                st.session_state.vectorstore = vectorstore

                st.success("✅ PDFs processed successfully! You can now ask questions.")

    # Handle user questions
    if user_question:
        if "vectorstore" not in st.session_state:
            st.warning("⚠️ Please upload and process PDFs first.")
        else:
            # Generate answer using RAG
            with st.spinner("🤔 Thinking and generating answer..."):
                try:
                    result = generate_answer(
                        st.session_state.vectorstore,
                        user_question
                    )
                    
                    # Display the answer
                    st.write("### 💬 Answer:")
                    st.write(result['result'])
                    
                    # Show source chunks in an expander
                    with st.expander("📄 View source chunks used"):
                        st.write("These are the document sections used to generate the answer:")
                        for i, doc in enumerate(result['source_documents'], 1):
                            st.write(f"**Chunk {i}:**")
                            st.write(doc.page_content)
                            st.write("---")
                
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
                    st.write("Please check your GROQ_API_KEY and try again.")


# -------------------- RUN --------------------
if __name__ == "__main__":
    main()

