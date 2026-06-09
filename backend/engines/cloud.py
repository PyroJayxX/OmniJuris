"""
engines/cloud.py
OmniJuris — Cloud inference via Gemini Flash API
"""

import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL = "gemini-3.5-flash"
_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment.")
        _client = genai.Client(api_key=api_key)
    return _client


def generate_cloud(prompt: str) -> str:
    try:
        client = _get_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=1024,
            ),
        )
        return response.text.strip()
    except ValueError as e:
        return f"Cloud engine unavailable: {str(e)}"
    except Exception as e:
        return f"Cloud engine error: {str(e)}"