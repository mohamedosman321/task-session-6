"""LLM setup. The agent uses Groq (free tier) through LlamaIndex."""

import os

from dotenv import load_dotenv
from llama_index.llms.groq import Groq

load_dotenv()

DEFAULT_MODEL = "llama-3.3-70b-versatile"


def read_api_key():
    """Look for the key in the environment, then in Colab Secrets."""
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key
    try:
        from google.colab import userdata  # only exists inside Google Colab

        return userdata.get("GROQ_API_KEY")
    except Exception:
        return None


def get_llm(model=None, temperature=0.0):
    """Create the Groq LLM. Temperature 0 keeps tool calls stable."""
    api_key = read_api_key()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Get a free key at https://console.groq.com, "
            "then put it in a .env file (see .env.example) or in Colab Secrets."
        )
    model = model or os.getenv("GROQ_MODEL", DEFAULT_MODEL)
    return Groq(model=model, api_key=api_key, temperature=temperature)
