"""
engines/local.py
OmniJuris — Local inference via Ollama
Runs Qwen3:4b-instruct locally. Supports thinking mode.
"""

import requests
import json

OLLAMA_URL      = "http://localhost:11434/api/generate"
LOCAL_MODEL     = "qwen3:4b-instruct"
THINKING_MODEL  = "qwen3:4b-instruct"   # same model, thinking controlled by prompt


def generate_local(prompt: str, thinking_mode: bool = False) -> str:
    """
    Generate a response using the local Ollama model.

    Parameters
    ----------
    prompt        : full constructed prompt from core/prompt.py
    thinking_mode : if True, enables Qwen3 extended reasoning

    Returns
    -------
    Generated answer string
    """
    # Qwen3 thinking mode is enabled by appending /think to the prompt
    # or by setting the think parameter in the options
    payload = {
        "model":  LOCAL_MODEL,
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

async def stream_local(prompt: str, thinking_mode: bool = False):
    payload = {
        "model": LOCAL_MODEL,
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