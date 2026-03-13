import logging

from generation.groq_client import generate as groq_generate

logger = logging.getLogger(__name__)


def generate(prompt):
    """
    prompt = {
        "system": "...",
        "user": "..."
    }
    """
    try:
        return groq_generate(prompt)
    except Exception as e:
        logger.error("LLM generation failed: %s", e)
        return f"Error communicating with LLM: {str(e)}"
