import src.config as con
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model=con.OPENAI_MODEL_NAME, api_key=con.OPENAI_API_KEY)  # or use os.environ

response = llm.invoke("Say hello")
print(response.content)
