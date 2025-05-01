from src import config as con
from src import ingest_parse as ip
from src import chunk
from src import embed_and_store as emb
from src import query_pipeline as qp
from langchain.chat_models import ChatOpenAI

#TODO(Jonny): Set up logging and linting

FILE_PATH = "data/Handelsblatt_Artikel_Dobelli.pdf"

llm = ChatOpenAI(
    openai_api_key=con.OPENAI_API_KEY,
    model_name=con.OPENAI_MODEL_NAME,
    temperature=1.0
)

# parse PDF text
test_text = ip.load_pdf(FILE_PATH)

# chunk
# TODO(Jonny): Add chunking code

# generate embeddings for chunks
emb.embed_and_store(test_text)

# create vector database
vectordb = emb.setup_vectordb()

# generate QA against loaded document
answer = qp.ask("What is this interview about?", llm, vectordb)
print(answer)