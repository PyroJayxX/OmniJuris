"""
engines/local.py
OmniJuris — Local inference via Ollama
Runs Qwen3:4b-instruct locally. Supports thinking mode.
"""

import requests
import json

OLLAMA_URL      = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
DEFAULT_LOCAL_MODEL = "qwen3:4b-instruct"   # fallback only, used when no model requested/discovered
THINKING_MODEL  = "qwen3:4b-instruct"   # same model, thinking controlled by prompt


def list_local_models() -> dict:
    """
    Query Ollama for the models currently installed locally.

    Returns
    -------
    {
        "available": bool,       # False if Ollama is unreachable or has no models
        "models":    list[str],  # installed model names
        "message":   str | None, # human-readable status when not available
    }
    Never raises.
    """
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=5)
        response.raise_for_status()
        data = response.json()
        names = [
            m.get("name") or m.get("model")
            for m in data.get("models", [])
            if m.get("name") or m.get("model")
        ]
        if not names:
            return {
                "available": False,
                "models": [],
                "message": "Ollama is running but no models are installed. Run: ollama pull qwen3:4b-instruct",
            }
        return {"available": True, "models": names, "message": None}
    except requests.exceptions.ConnectionError:
        return {
            "available": False,
            "models": [],
            "message": "Local engine unavailable. Ensure Ollama is running with: ollama serve",
        }
    except requests.exceptions.Timeout:
        return {"available": False, "models": [], "message": "Ollama did not respond in time."}
    except Exception as e:
        return {"available": False, "models": [], "message": f"Local engine error: {str(e)}"}


def generate_local(prompt: str, thinking_mode: bool = False, model: str | None = None) -> str:
    """
    Generate a response using the local Ollama model.

    Parameters
    ----------
    prompt        : full constructed prompt from core/prompt.py
    thinking_mode : if True, enables Qwen3 extended reasoning
    model         : Ollama model name to use; falls back to DEFAULT_LOCAL_MODEL if omitted

    Returns
    -------
    Generated answer string
    """
    # Qwen3 thinking mode is enabled by appending /think to the prompt
    # or by setting the think parameter in the options
    payload = {
        "model":  model or DEFAULT_LOCAL_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,    # low temp for factual legal answers
            "top_p":       0.9,
            "num_predict": 1024,   # max tokens to generate
        },
    }

    # Qwen3 thinking mode — inject think flag via system context
    if thinking_mode:
        payload["options"]["think"] = True

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return "Local engine unavailable. Ensure Ollama is running with: ollama serve"
    except requests.exceptions.Timeout:
        return "Local engine timed out. The model may be loading — try again in a moment."
    except Exception as e:
        return f"Local engine error: {str(e)}"

import asyncio

async def stream_local(prompt: str, thinking_mode: bool = False, model: str | None = None):
    payload = {
        "model": model or DEFAULT_LOCAL_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 4096,
        },
    }
    if thinking_mode:
        payload["options"]["think"] = True

    try:
        import httpx
        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream("POST", OLLAMA_URL, json=payload) as response:
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        if data.get("response"):
                            yield data["response"]
                        if data.get("done"):
                            break
    except Exception as e:
        yield f"Local engine error: {str(e)}"