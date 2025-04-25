from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from src.config import OPENAI_API_KEY

persist_directory = ".chroma"

def ask(query):
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=None)
    retriever = vectordb.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=OPENAI_API_KEY), retriever=retriever)
    return qa.run(query)
