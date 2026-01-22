import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# Import your existing logic
from src.ingest import load_and_chunk
from src.embeddings import get_embedding_model
from src.vector_store import create_or_load_vectorstore
from src.retriever import get_retriever
from src.rag_chain import create_rag_chain

# --- CONFIGURATION ---
st.set_page_config(page_title="AskYourDocs", page_icon="🧠")

st.title("AskYourDocs 🧠")
st.caption("Upload a PDF and ask questions about its content.")

# Sidebar
with st.sidebar:
    st.header("About This App")
    st.markdown(
        """
        This tool uses **Retrieval-Augmented Generation (RAG)** to let you chat with your documents. 
        
        **How it works:**
        1. Upload a PDF.
        2. The AI reads and understands the content.
        3. Ask any question, and it finds the exact answer from your file.
        
        *Built with LangChain, Groq, and Streamlit.*
        """
    )
    
    st.divider()
    
    st.subheader("Settings") 
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "last_uploaded" not in st.session_state:
    st.session_state.last_uploaded = None

# --- LOGIC TO HANDLE FILE UPLOADS ---
if uploaded_file:
    if uploaded_file.name != st.session_state.last_uploaded:
        st.session_state.messages = []
        st.session_state.rag_chain = None
        st.session_state.last_uploaded = uploaded_file.name

    if st.session_state.rag_chain is None:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                load_dotenv()
                chunks = load_and_chunk(tmp_file_path)
                
                if not chunks:
                    st.error("⚠️ No text found in this PDF! It might be a scanned image.")
                    st.stop()
                
                embedding_model = get_embedding_model()
                vectorstore = create_or_load_vectorstore(chunks, embedding_model)
                retriever = get_retriever(vectorstore)
                st.session_state.rag_chain = create_rag_chain(retriever)
                
                st.success(f"Ready! You are now chatting with: {uploaded_file.name}")
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)

# --- CHAT INTERFACE ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.rag_chain:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag_chain.invoke(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                
                # --- NEW: BEAUTIFUL ERROR HANDLING ---
                except Exception as e:
                    error_msg = str(e)
                    if "429" in error_msg or "Resource exhausted" in error_msg:
                        st.warning("""🚦 You’ve hit today’s curiosity limit!
To keep things fair for everyone, this demo allows a limited number of questions per day.
Take a short break, try again later, or restart the app with a different model to keep exploring.""")
                    else:
                        st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("Please upload a PDF file first!")