import requests


OLLAMA_URL = "http://localhost:11434/api/chat"
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

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()

    return response.json()["message"]["content"]