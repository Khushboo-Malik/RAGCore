from langchain_chroma import Chroma

def create_or_load_vectorstore(chunks, embedding_model):
    """
    Creates an in-memory vector store. 
    This is perfect for demos because it avoids file locking issues.
    """
    # 1. Create the vector store in RAM (persist_directory=None)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=None  # This forces it to run in memory
    )
    
    return vectorstore