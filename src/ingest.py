# Updated to use PyMuPDFLoader for better stability
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 

def load_and_chunk(file_path):
    # PyMuPDF is more robust and won't crash on 'bbox' errors
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)
    return chunks