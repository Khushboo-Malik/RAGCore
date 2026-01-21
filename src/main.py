from ingest import load_and_chunk
from embeddings import get_embedding_model
from vector_store import create_or_load_vectorstore
from retriever import get_retriever
from rag_chain import create_rag_chain
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    chunks = load_and_chunk("data/documents/sample.pdf")

    embedding_model = get_embedding_model()
    vectorstore = create_or_load_vectorstore(chunks, embedding_model)

    retriever = get_retriever(vectorstore)
    rag_chain = create_rag_chain(retriever)

    while True:
        query = input("\nAsk a question (or 'exit'): ")
        if query.lower() == "exit":
            break

        answer = rag_chain.run(query)
        print("\nAnswer:", answer)
