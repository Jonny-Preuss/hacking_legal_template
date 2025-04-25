from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import OPENAI_API_KEY

embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

def embed_and_store(text, persist_directory=".chroma"):
    docs = splitter.create_documents([text])
    vectordb = Chroma.from_documents(docs, embedding, persist_directory=persist_directory)
    vectordb.persist()
    return vectordb
