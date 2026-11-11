
# config.py
# Shared settings for every script in PresentationMaker.

import os
from pathlib import Path

# Keep Hugging Face output quiet.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

BASE_DIR = Path(__file__).resolve().parent

# A generation request can provide its own workspace.
# If no workspace is provided, use the original project directory.
WORKSPACE_DIR = Path(
    os.environ.get("PRESENTATIONMAKER_WORKSPACE", BASE_DIR)
)

# Request-specific folders
INPUT_DIR = WORKSPACE_DIR / "input_ppts"
OUTPUT_DIR = WORKSPACE_DIR / "output"

# These remain shared at project level
DATA_DIR = BASE_DIR / "data"
ENV_FILE = BASE_DIR / ".env"

# Knowledge-base files
RECORDS_FILE = OUTPUT_DIR / "slide_records.json"
EMBEDDINGS_FILE = OUTPUT_DIR / "slide_embeddings.npy"
BGE_EMBEDDINGS_FILE = OUTPUT_DIR / "slide_embeddings_bge.npy"

EVALUATION_FILE = DATA_DIR / "evaluation_queries.json"

# Embedding configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
MAX_SEQ_LENGTH = 256

# Gemini configuration
GEMINI_MODEL = "gemini-3.1-flash-lite"

# Create the output folder if it doesn't exist.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
