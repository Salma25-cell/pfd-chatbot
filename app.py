import streamlit as st
import os
from dotenv import load_dotenv  
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISSstramlit

load_dotenv()

def get_pdf_text(pdf_docs):
 #----extract text from uploaded pdf documents
   text=""
   for pdf in pdf_docs:
      pdf_reader= PdfReader(pdf)
      for page in pdf_reader.pages:
         page_text = page.extract_text()
         if page_text:
            text += page_text + "\n"
   return text    


#-------- creating text chunks
def get_text_chunks(raw_text):
   #--split text into smalller chunks for embeddings
    splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n","\n"," "," "],
    chunk_size=1000,
    chunk_overlap=200,
    )
    return splitter.split_text(raw_text)

#----VECTOR STORE------
def get_vectorstore(text_chunks):
   if not text_chunks:
      raise ValueError("no text chunks to embedded")
   
   #creation of embeddings usng huggibng face model(runs on your machine)
   embeddings= HuggingFaceEmbeddings(
      model_name="sentence-transformers/all-MiniLM-L6-v2",
      model_kwargs={'device':'cpu'},
      encode_kwargs={'normalize_embeddings':True}
   )

#Create FAISS vector store from chunks
   vectorstore = FAISS.from_texts(
      texts = text_chunks,
      embedding=embeddings
   )
   return vectorstore

def main():
    st.set_page_config(
       page_title="chat with multiple PDFs", page_icon=":books:")

    st.header("Chat with multiple PDFs :books:")
    st.info("Using free local Huggingface embeddings- no API costs or limits")


    user_question=st.text_input("Ask a question about your documents:")

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs= st.file_uploader(
           "Upload Your PDFs Here and Click on 'Process'",
           accept_multiple_files=True,
           type=['pdf'])
        
        if st.button("Process"):
            if not pdf_docs:
               st.error("Please Upload atleast one PDF")
               st.stop()

            with st.spinner("Reading and Indexing PDFs.."):
               #Step1: Extract Text From PDFs
              #get pdf text
               raw_text = get_pdf_text(pdf_docs)
               if not raw_text.strip():
                  st.error("No Readable text Found in the PDF.")
                  st.stop()
               #--st.write(raw_text)

               #Step2 : Split text into Chunks
               text_chunks = get_text_chunks(raw_text)

               if not text_chunks:
                  st.error("Failed to split text into chunks.")
                  st.stop()
                 #st.write("text_chunks)")
                 #st.success(f"Processed {len(text_chunks)} text chunks!")
               st.write(f"Total Chunks:{len(text_chunks)}")
            #Step 3: Create Vector Store with embeddings

               #get vector store
               with st.spinner("Creating embeddings( first run may take a moment)....."):
                vectorstore = get_vectorstore(text_chunks)

                #stores in session state for later use
               st.session_state.vectorstore= vectorstore 
               st.success("PDFs Processed Successfully")
         
      #Handle User3 Questions
            if user_question:
               if "vectorstore"not in st.session_state:
                  st.warning("PLease upload and process PDFs first.")
               else:
                  st.info("Vector Store ready. Chat Logic Can be Added next.")   
                  # TODO: Add q/a functionality here




if __name__ == '__main__':
    main()
   