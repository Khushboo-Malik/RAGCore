import os
from dotenv import load_dotenv

# Import your project modules as src.*
from src.ingest import load_and_chunk
from src.embeddings import get_embedding_model
from src.vector_store import create_or_load_vectorstore
from src.retriever import get_retriever
from src.rag_chain import create_rag_chain

# Load environment variables from the root .env
# This ensures the API key is loaded from the base RAGCore folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(os.path.dirname(BASE_DIR), ".env")
load_dotenv(dotenv_path)

def main():
    # 1. Load and split PDF into chunks
    # Note: Ensure your PDF is at RAGCore/data/documents/sample.pdf
    pdf_path = "data/documents/sample.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"Error: Could not find file at {pdf_path}")
        return

    print(f"--- Loading and chunking: {pdf_path} ---")
    chunks = load_and_chunk(pdf_path)

    # 2. Get embedding model (HuggingFace)
    print("--- Initializing Embedding Model ---")
    embedding_model = get_embedding_model()

    # 3. Create or load vectorstore (Chroma)
    print("--- Setting up Vector Store ---")
    vectorstore = create_or_load_vectorstore(chunks, embedding_model)

    # 4. Get retriever from vectorstore
    retriever = get_retriever(vectorstore)

    # 5. Create modern LCEL RAG chain
    print("--- Creating RAG Chain ---")
    rag_chain = create_rag_chain(retriever)

    # Interactive query loop
    print("\n" + "="*30)
    print("RAG Pipeline Ready! Type 'exit' to quit.")
    print("="*30)
    
    while True:
        query = input("\nAsk a question: ")
        if query.lower() in ["exit", "quit"]:
            break

        if not query.strip():
            continue

        try:
            # Using LCEL with StrOutputParser returns the answer as a string
            answer = rag_chain.invoke(query)
            print("\nAnswer:", answer)
        except Exception as e:
            print(f"\nAn error occurred during generation: {e}")

if __name__ == "__main__":
    main()