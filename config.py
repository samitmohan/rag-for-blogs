import os


# -- LLM --
GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL: str = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# -- Embeddings --
EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

# -- FAISS --
DATA_DIR: str = "data"
FAISS_DIR: str = os.path.join(DATA_DIR, "faiss")

# -- Chunking --
CHUNK_TARGET_TOKENS: int = 200
CHUNK_OVERLAP_TOKENS: int = 50

# -- Retrieval --
CONFIDENCE_THRESHOLD: float = 1.0

# -- Logging --
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
