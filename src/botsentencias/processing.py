from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
import fitz
from chromadb.api.models.Collection import Collection

from botsentencias.config import Settings
from botsentencias.registry import Registry


logger = logging.getLogger(__name__)
PAGE_RE = re.compile(r"p[aá]g\.?\s*(\d+)", re.IGNORECASE)


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    text: str
    fallo_name: str | None
    tomo: str | None
    page: str | None
    year: int | None
    source_pdf: str


class Embedder:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.backend = settings.embedding_backend.lower().strip()
        if self.backend == "sentence-transformers":
            import importlib
            import importlib.util

            if importlib.util.find_spec("sentence_transformers") is None:
                raise RuntimeError("Falta dependencia sentence-transformers. Instale requirements.txt")
            sentence_transformers_module = importlib.import_module("sentence_transformers")
            self.model = sentence_transformers_module.SentenceTransformer(settings.embedding_model)
        elif self.backend == "openai":
            import importlib
            import importlib.util

            if not settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY es obligatorio para EMBEDDING_BACKEND=openai")
            if importlib.util.find_spec("openai") is None:
                raise RuntimeError("Falta dependencia openai. Instale requirements.txt")
            openai_module = importlib.import_module("openai")
            self.client = openai_module.OpenAI(api_key=settings.openai_api_key)
        else:
            raise ValueError(f"Embedding backend no soportado: {self.backend}")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.backend == "sentence-transformers":
            return self.model.encode(texts, normalize_embeddings=True).tolist()
        response = self.client.embeddings.create(model=self.settings.openai_embedding_model, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


class PdfProcessor:
    def __init__(self, settings: Settings, registry: Registry, embedder: Embedder) -> None:
        self.settings = settings
        self.registry = registry
        self.embedder = embedder
        self.chroma_client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self.collection: Collection = self.chroma_client.get_or_create_collection(
            name="fallos_chunks", metadata={"hnsw:space": "cosine"}
        )

    def process_all_downloaded(self) -> None:
        rows = self.registry.list_downloaded_pdfs()
        for row in rows:
            pdf_id = row["id"]
            file_path = Path(row["file_path"])
            if not file_path.exists():
                logger.warning("PDF inexistente: %s", file_path)
                continue
            pdf_hash = self._file_hash(file_path)
            previous = self.registry.get_processing_record(pdf_id)
            if previous and previous["status"] == "processed" and previous["content_hash"] == pdf_hash:
                logger.debug("Saltando PDF ya procesado: %s", file_path)
                continue

            try:
                chunks = self._extract_chunks(file_path=file_path, row=row)
                if not chunks:
                    self.registry.set_processing_status(pdf_id, "error", content_hash=pdf_hash, error_message="Sin texto")
                    continue
                self._upsert_chunks(pdf_id, chunks)
                self.registry.set_processing_status(
                    pdf_id,
                    "processed",
                    content_hash=pdf_hash,
                    chunk_count=len(chunks),
                )
                logger.info("Procesado %s: %s chunks", file_path.name, len(chunks))
            except Exception as exc:
                self.registry.set_processing_status(
                    pdf_id, "error", content_hash=pdf_hash, error_message=str(exc)
                )
                logger.exception("Error procesando %s", file_path)

    @staticmethod
    def _file_hash(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()

    def _extract_chunks(self, file_path: Path, row) -> list[Chunk]:
        with fitz.open(file_path) as doc:
            pages_text = [page.get_text("text") for page in doc]

        full_text = "\n".join(pages_text)
        tokens = full_text.split()
        max_tokens = self.settings.chunk_max_tokens
        overlap = self.settings.chunk_overlap_tokens
        chunks: list[Chunk] = []
        step = max(1, max_tokens - overlap)

        for i in range(0, len(tokens), step):
            token_slice = tokens[i : i + max_tokens]
            if not token_slice:
                continue
            text = " ".join(token_slice).strip()
            if len(text) < 60:
                continue
            chunk_id = hashlib.md5(f"{file_path}:{i}".encode("utf-8")).hexdigest()
            page = self._infer_page(text)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=text,
                    fallo_name=file_path.stem,
                    tomo=row["tomo"],
                    page=page,
                    year=row["year"],
                    source_pdf=str(file_path),
                )
            )
        return chunks

    @staticmethod
    def _infer_page(text: str) -> str | None:
        match = PAGE_RE.search(text[:300])
        return match.group(1) if match else None

    def _upsert_chunks(self, pdf_id: int, chunks: list[Chunk]) -> None:
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.embed_texts(texts)
        metadatas = [
            {
                "fallo_name": chunk.fallo_name or "",
                "tomo": chunk.tomo or "",
                "page": chunk.page or "",
                "year": int(chunk.year) if chunk.year else 0,
                "source_pdf": chunk.source_pdf,
            }
            for chunk in chunks
        ]
        ids = [chunk.chunk_id for chunk in chunks]

        self.collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

        for chunk in chunks:
            self.registry.add_chunk_record(
                chunk_id=chunk.chunk_id,
                pdf_id=pdf_id,
                fallo_name=chunk.fallo_name,
                tomo=chunk.tomo,
                page=chunk.page,
                year=chunk.year,
                text_excerpt=chunk.text,
            )
