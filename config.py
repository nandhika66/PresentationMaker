# Shared settings for every script in PresentationMaker.
# All paths are relative to this file, so the project works
# no matter which folder you run the commands from.

import os
from pathlib import Path

# Keep Hugging Face output quiet.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "input_ppts"
OUTPUT_DIR = BASE_DIR / "output"
DATA_DIR = BASE_DIR / "data"

RECORDS_FILE = OUTPUT_DIR / "slide_records.json"
EMBEDDINGS_FILE = OUTPUT_DIR / "slide_embeddings.npy"
EVALUATION_FILE = DATA_DIR / "evaluation_queries.json"
ENV_FILE = BASE_DIR / ".env"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
MAX_SEQ_LENGTH = 256  # same limit sentence-transformers uses for MiniLM

GEMINI_MODEL = "gemini-3.5-flash-lite"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
