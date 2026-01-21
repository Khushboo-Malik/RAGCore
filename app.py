import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# Import your existing logic from the src folder
from src.ingest import load_and_chunk
from src.embeddings import get_embedding_model
from src.vector_store import create_or_load_vectorstore
from src.retriever import get_retriever
from src.rag_chain import create_rag_chain

# Page Config
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

st.title("📄 PDF Chat Assistant")

# Sidebar for file upload
with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Session State for the RAG Chain
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# Logic: Process the PDF only when a new file is uploaded
if uploaded_file and st.session_state.rag_chain is None:
    with st.spinner("Processing PDF... (This might take a moment)"):
        # 1. Save uploaded file to a temporary file
        # (PyMuPDFLoader needs a real file path, not just bytes)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # 2. Run your existing pipeline
        try:
            load_dotenv()
            
            # Ingest
            chunks = load_and_chunk(tmp_file_path)
            
            # Embed
            embedding_model = get_embedding_model()
            
            # Store
            # Note: We use a separate DB folder for the web app to avoid conflicts
            vectorstore = create_or_load_vectorstore(chunks, embedding_model)
            
            # Retrieve & Chain
            retriever = get_retriever(vectorstore)
            st.session_state.rag_chain = create_rag_chain(retriever)
            
            st.success("PDF Loaded! You can now chat.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
        finally:
            # Cleanup the temp file
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

# UI: Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# UI: Chat Input
if prompt := st.chat_input("Ask a question about your PDF..."):
    # 1. Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate Assistant Response
    if st.session_state.rag_chain:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag_chain.invoke(prompt)
                    st.markdown(response)
                    # Add to history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error generating response: {e}")
    else:
        st.warning("Please upload a PDF file first!")