import os
from dotenv import load_dotenv

load_dotenv()  

MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.0
MAX_TOKENS = 300

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")
