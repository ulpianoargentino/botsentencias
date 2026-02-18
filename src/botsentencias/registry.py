from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


class Registry:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self.connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS pdf_registry (
                    id INTEGER PRIMARY KEY,
                    source_url TEXT UNIQUE,
                    file_path TEXT,
                    file_name TEXT,
                    tomo TEXT,
                    year INTEGER,
                    discovered_at TEXT,
                    downloaded_at TEXT,
                    status TEXT,
                    error_message TEXT
                );

                CREATE TABLE IF NOT EXISTS processing_registry (
                    id INTEGER PRIMARY KEY,
                    pdf_id INTEGER UNIQUE,
                    content_hash TEXT,
                    processed_at TEXT,
                    status TEXT,
                    chunk_count INTEGER,
                    error_message TEXT,
                    FOREIGN KEY (pdf_id) REFERENCES pdf_registry(id)
                );

                CREATE TABLE IF NOT EXISTS chunks_registry (
                    id INTEGER PRIMARY KEY,
                    chunk_id TEXT UNIQUE,
                    pdf_id INTEGER,
                    fallo_name TEXT,
                    tomo TEXT,
                    page TEXT,
                    year INTEGER,
                    text_excerpt TEXT,
                    FOREIGN KEY (pdf_id) REFERENCES pdf_registry(id)
                );
                """
            )

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def upsert_pdf_record(
        self,
        source_url: str,
        file_path: str,
        file_name: str,
        tomo: str | None,
        year: int | None,
        status: str,
        error_message: str | None = None,
    ) -> None:
        now = self.now_iso()
        downloaded_at = now if status == "downloaded" else None
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO pdf_registry
                    (source_url, file_path, file_name, tomo, year, discovered_at, downloaded_at, status, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_url) DO UPDATE SET
                    file_path=excluded.file_path,
                    file_name=excluded.file_name,
                    tomo=excluded.tomo,
                    year=excluded.year,
                    downloaded_at=COALESCE(excluded.downloaded_at, pdf_registry.downloaded_at),
                    status=excluded.status,
                    error_message=excluded.error_message
                """,
                (source_url, file_path, file_name, tomo, year, now, downloaded_at, status, error_message),
            )

    def list_downloaded_pdfs(self):
        with self.connection() as conn:
            return conn.execute(
                "SELECT * FROM pdf_registry WHERE status = 'downloaded' ORDER BY year, file_name"
            ).fetchall()

    def get_pdf_by_url(self, source_url: str):
        with self.connection() as conn:
            return conn.execute("SELECT * FROM pdf_registry WHERE source_url = ?", (source_url,)).fetchone()

    def set_processing_status(
        self,
        pdf_id: int,
        status: str,
        content_hash: str | None = None,
        chunk_count: int | None = None,
        error_message: str | None = None,
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO processing_registry
                    (pdf_id, content_hash, processed_at, status, chunk_count, error_message)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(pdf_id) DO UPDATE SET
                    content_hash=excluded.content_hash,
                    processed_at=excluded.processed_at,
                    status=excluded.status,
                    chunk_count=excluded.chunk_count,
                    error_message=excluded.error_message
                """,
                (pdf_id, content_hash, self.now_iso(), status, chunk_count, error_message),
            )

    def get_processing_record(self, pdf_id: int):
        with self.connection() as conn:
            return conn.execute(
                "SELECT * FROM processing_registry WHERE pdf_id = ?", (pdf_id,)
            ).fetchone()

    def add_chunk_record(
        self,
        chunk_id: str,
        pdf_id: int,
        fallo_name: str | None,
        tomo: str | None,
        page: str | None,
        year: int | None,
        text_excerpt: str,
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO chunks_registry
                    (chunk_id, pdf_id, fallo_name, tomo, page, year, text_excerpt)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (chunk_id, pdf_id, fallo_name, tomo, page, year, text_excerpt[:500]),
            )

    def export_pdf_registry_csv(self, output_path: Path) -> None:
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM pdf_registry ORDER BY year, file_name").fetchall()
        headers = rows[0].keys() if rows else []
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            if headers:
                f.write(",".join(headers) + "\n")
                for row in rows:
                    f.write(",".join('' if v is None else str(v).replace(',', ';') for v in row) + "\n")
