from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI

def create_rag_chain(retriever):
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    return chain
