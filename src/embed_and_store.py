# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import OPENAI_API_KEY

persist_directory = ".chroma"
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

def embed_and_store(text, persist_directory=".chroma"):
    docs = splitter.create_documents([text])
    vectordb = Chroma.from_documents(docs, embedding, persist_directory=persist_directory)
    # vectordb.persist()
    return vectordb

def setup_vectordb():
    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    try:
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
        # Maybe also check if vectordb is empty here if you want
        print("VectorDB loaded from disk.")
    except Exception as e:
        print("No existing VectorDB found. You may need to ingest documents first.")
        raise e
    return vectordb