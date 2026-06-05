from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(ROOT_DIR / ".env")

DATA_PATH = ROOT_DIR / "data" / "tasks.csv"
EVALUATION_DATA_PATH = ROOT_DIR / "data" / "evaluation_tasks.csv"
STORAGE_DIR = ROOT_DIR / "storage" / "chroma"
EVALUATION_STORAGE_DIR = ROOT_DIR / "storage" / "evaluation_chroma"
OUTPUTS_DIR = ROOT_DIR / "outputs"

COLLECTION_NAME = "tickets"
DEFAULT_TOP_K = 5

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1")
