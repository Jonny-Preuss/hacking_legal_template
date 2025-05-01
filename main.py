from src import config as con
from src import ingest_parse as ip
from src import chunk
from src import embed_and_store as emb
from src import query_pipeline as qp
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
import logging

#TODO(Jonny): Set up logging and linting

FILE_PATH = "data/Handelsblatt_Artikel_Dobelli.pdf"

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()  # Still logs to console
    ]
)

# configure LLM settings
llm = ChatOpenAI(
    api_key=con.OPENAI_API_KEY,
    model=con.OPENAI_MODEL_NAME,
    temperature=1.0
)

logging.info("Loading the document...")
# parse PDF text
test_text = ip.load_pdf(FILE_PATH)
# print(test_text)

# chunk
# TODO(Jonny): Add chunking code

logging.info("Generating embeddings...")
# generate embeddings for chunks
emb.embed_and_store(test_text)

# create vector database
vectordb = emb.setup_vectordb()

logging.info("Generating an answer to your question...")
# generate QA against loaded document
answer = qp.ask("Worum geht es in dem Interview mit Dobelli?", llm, vectordb)
logging.info(f"The result is: {answer}")