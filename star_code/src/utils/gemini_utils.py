import os
from google import genai

def get_client(api_key=None):
    api_key = api_key or os.environ.get("GEMINI_API_KEY", None)

    return genai.Client(api_key=api_key)
