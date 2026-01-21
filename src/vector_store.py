from langchain_chroma import Chroma

def create_or_load_vectorstore(chunks, embedding_model):
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="./chroma_db" 
    )
    return vectorstore