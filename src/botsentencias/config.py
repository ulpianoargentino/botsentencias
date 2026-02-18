from __future__ import annotations

import importlib
import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv_if_available() -> None:
    if importlib.util.find_spec("dotenv") is None:
        return
    dotenv_module = importlib.import_module("dotenv")
    dotenv_module.load_dotenv()


_load_dotenv_if_available()


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


@dataclass(slots=True)
class Settings:
    data_dir: Path = Path(os.getenv("DATA_DIR", "./data"))
    pdf_dir: Path = Path(os.getenv("PDF_DIR", "./data/pdfs"))
    chroma_dir: Path = Path(os.getenv("CHROMA_DIR", "./data/chroma"))
    db_path: Path = Path(os.getenv("DB_PATH", "./data/registry.sqlite"))

    base_url: str = os.getenv(
        "BASE_URL", "https://sjservicios.csjn.gov.ar/sj/tomosFallos.do?method=iniciar"
    )
    min_year: int = _int_env("MIN_YEAR", 1994)
    request_timeout: int = _int_env("REQUEST_TIMEOUT", 45)
    request_delay_seconds: float = _float_env("REQUEST_DELAY_SECONDS", 1.5)
    user_agent: str = os.getenv("USER_AGENT", "botsentencias/1.0")

    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", "sentence-transformers")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    llm_backend: str = os.getenv("LLM_BACKEND", "local-echo")
    llm_model: str = os.getenv("LLM_MODEL", "llama3.1:8b")
    openai_chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")

    top_k: int = _int_env("TOP_K", 8)
    bm25_k: int = _int_env("BM25_K", 6)
    chunk_max_tokens: int = _int_env("CHUNK_MAX_TOKENS", 1000)
    chunk_overlap_tokens: int = _int_env("CHUNK_OVERLAP_TOKENS", 100)

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
