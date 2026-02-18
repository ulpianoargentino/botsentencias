from __future__ import annotations

import logging
from dataclasses import dataclass

import chromadb
from rank_bm25 import BM25Okapi

from botsentencias.config import Settings
from botsentencias.processing import Embedder
from botsentencias.registry import Registry


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RetrievedChunk:
    text: str
    metadata: dict
    score: float


class Retriever:
    def __init__(self, settings: Settings, embedder: Embedder):
        self.settings = settings
        self.embedder = embedder
        self.chroma_client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self.collection = self.chroma_client.get_or_create_collection(
            name="fallos_chunks", metadata={"hnsw:space": "cosine"}
        )
        self._bm25_model: BM25Okapi | None = None
        self._bm25_documents: list[str] = []
        self._bm25_metadatas: list[dict] = []

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k or self.settings.top_k
        query_embedding = self.embedder.embed_query(query)
        result = self.collection.query(query_embeddings=[query_embedding], n_results=k)

        vector_hits = []
        for idx, text in enumerate(result.get("documents", [[]])[0]):
            vector_hits.append(
                RetrievedChunk(
                    text=text,
                    metadata=result.get("metadatas", [[]])[0][idx],
                    score=1 - float(result.get("distances", [[]])[0][idx]),
                )
            )

        bm25_hits = self._retrieve_bm25(query)
        merged = self._merge_hits(vector_hits, bm25_hits, max_items=k)
        return merged

    def _retrieve_bm25(self, query: str) -> list[RetrievedChunk]:
        if self._bm25_model is None:
            self._build_bm25()
        if self._bm25_model is None:
            return []

        query_tokens = query.lower().split()
        scores = self._bm25_model.get_scores(query_tokens)
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.settings.bm25_k]
        hits: list[RetrievedChunk] = []
        for idx in ranked_idx:
            score = float(scores[idx])
            if score <= 0:
                continue
            hits.append(
                RetrievedChunk(
                    text=self._bm25_documents[idx], metadata=self._bm25_metadatas[idx], score=score
                )
            )
        return hits

    def _build_bm25(self) -> None:
        payload = self.collection.get(include=["documents", "metadatas"])
        documents = payload.get("documents", [])
        metadatas = payload.get("metadatas", [])
        if not documents:
            self._bm25_model = None
            return
        self._bm25_documents = list(documents)
        self._bm25_metadatas = list(metadatas)
        tokenized = [doc.lower().split() for doc in documents]
        self._bm25_model = BM25Okapi(tokenized)

    @staticmethod
    def _merge_hits(
        vector_hits: list[RetrievedChunk], bm25_hits: list[RetrievedChunk], max_items: int
    ) -> list[RetrievedChunk]:
        by_text: dict[str, RetrievedChunk] = {}
        for hit in vector_hits + bm25_hits:
            key = hit.text[:200]
            existing = by_text.get(key)
            if existing is None or hit.score > existing.score:
                by_text[key] = hit
        merged = sorted(by_text.values(), key=lambda item: item.score, reverse=True)
        return merged[:max_items]


class LlmClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.backend = settings.llm_backend.lower().strip()

    def generate(self, question: str, contexts: list[RetrievedChunk]) -> str:
        prompt = self._build_prompt(question, contexts)
        if self.backend == "local-echo":
            return self._local_echo(question, contexts)
        if self.backend == "ollama":
            return self._ollama(prompt)
        if self.backend == "openai":
            return self._openai(prompt)
        if self.backend == "anthropic":
            return self._anthropic(prompt)
        raise ValueError(f"LLM backend no soportado: {self.backend}")

    @staticmethod
    def _build_prompt(question: str, contexts: list[RetrievedChunk]) -> str:
        blocks = []
        for idx, ctx in enumerate(contexts, start=1):
            md = ctx.metadata
            blocks.append(
                f"[{idx}] Fallo: {md.get('fallo_name')}; Tomo: {md.get('tomo')}; Página: {md.get('page')}; Año: {md.get('year')}\n"
                f"Texto: {ctx.text[:1800]}"
            )
        context_text = "\n\n".join(blocks)
        return (
            "Sos un asistente jurídico argentino. Respondé en español formal y citá explícitamente los fallos usados. "
            "Si la evidencia es insuficiente, decilo.\n\n"
            f"Pregunta: {question}\n\n"
            f"Contexto recuperado:\n{context_text}\n\n"
            "Estructura requerida:\n"
            "1) Respuesta jurídica fundada.\n"
            "2) Fuentes citadas con formato: Fallo (Tomo:X, Página:Y, Año:Z).\n"
            "3) Nivel de certeza (alto/medio/bajo)."
        )

    @staticmethod
    def _local_echo(question: str, contexts: list[RetrievedChunk]) -> str:
        if not contexts:
            return (
                "No encontré fragmentos relevantes en la base. La información disponible es insuficiente para "
                "responder con certeza."
            )
        lines = [
            "Respuesta preliminar (modo local-echo):",
            "Se recuperaron antecedentes potencialmente relevantes. Para una respuesta final, active un backend LLM.",
            f"Pregunta: {question}",
            "Fuentes:",
        ]
        for item in contexts:
            md = item.metadata
            lines.append(
                f"- {md.get('fallo_name')} (Tomo:{md.get('tomo')}, Página:{md.get('page')}, Año:{md.get('year')}) | score={item.score:.3f}"
            )
        lines.append("Nivel de certeza: bajo (respuesta de diagnóstico).")
        return "\n".join(lines)

    def _ollama(self, prompt: str) -> str:
        import requests

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": self.settings.llm_model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return response.json().get("response", "")

    def _openai(self, prompt: str) -> str:
        import importlib
        import importlib.util

        if not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY es obligatorio para LLM_BACKEND=openai")
        if importlib.util.find_spec("openai") is None:
            raise RuntimeError("Falta dependencia openai. Instale requirements.txt")

        openai_module = importlib.import_module("openai")
        client = openai_module.OpenAI(api_key=self.settings.openai_api_key)
        response = client.chat.completions.create(
            model=self.settings.openai_chat_model,
            messages=[
                {"role": "system", "content": "Asistente jurídico argentino."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content or ""

    def _anthropic(self, prompt: str) -> str:
        import importlib
        import importlib.util

        if not self.settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY es obligatorio para LLM_BACKEND=anthropic")
        if importlib.util.find_spec("anthropic") is None:
            raise RuntimeError("Falta dependencia anthropic. Instale requirements.txt")

        anthropic_module = importlib.import_module("anthropic")
        client = anthropic_module.Anthropic(api_key=self.settings.anthropic_api_key)
        response = client.messages.create(
            model=self.settings.llm_model,
            max_tokens=900,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        if response.content:
            return response.content[0].text
        return ""


class RagAgent:
    def __init__(self, settings: Settings, registry: Registry):
        self.retriever = Retriever(settings=settings, embedder=Embedder(settings))
        self.llm = LlmClient(settings=settings)
        self.registry = registry

    def ask(self, question: str, top_k: int | None = None) -> dict:
        chunks = self.retriever.retrieve(question, top_k=top_k)
        answer = self.llm.generate(question, chunks)
        sources = [
            {
                "fallo_name": c.metadata.get("fallo_name"),
                "tomo": c.metadata.get("tomo"),
                "page": c.metadata.get("page"),
                "year": c.metadata.get("year"),
                "score": c.score,
            }
            for c in chunks
        ]
        return {"answer": answer, "sources": sources}
