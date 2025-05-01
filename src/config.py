import os
from dotenv import load_dotenv

load_dotenv()
# print("RAW ENV VALUE:", os.environ.get("OPENAI_MODEL_NAME"))

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "o4-mini")


# Validation (warn if environment values look suspicious)
if isinstance(OPENAI_MODEL_NAME, str) and "os.getenv(" in OPENAI_MODEL_NAME:
    raise ValueError(f"Invalid OPENAI_MODEL_NAME: {OPENAI_MODEL_NAME!r}")

# Google
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


