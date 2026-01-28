import requests
import os

# Allow overriding the URL via environment variable
OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/api/chat")
MODEL_NAME = "qwen2.5:7b"

def generate(prompt):
    """
    prompt = {
        "system": "...",
        "user": "..."
    }
    """

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ],
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json()["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Error communicating with LLM: {str(e)}"