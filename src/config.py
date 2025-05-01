import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "o4-mini")

# Google
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
