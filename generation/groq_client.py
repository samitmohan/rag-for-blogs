import logging
from typing import Generator

from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def generate(prompt: dict) -> str:
    """Non-streaming generation. Returns the full response text."""
    client = _get_client()
    messages = [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": prompt["user"]},
    ]
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        stream=False,
    )
    return response.choices[0].message.content


def generate_stream(prompt: dict) -> Generator[str, None, None]:
    """Streaming generation. Yields tokens as they arrive."""
    client = _get_client()
    messages = [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": prompt["user"]},
    ]
    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content
