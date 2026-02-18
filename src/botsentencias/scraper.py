from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from botsentencias.config import Settings
from botsentencias.registry import Registry


logger = logging.getLogger(__name__)
PDF_RE = re.compile(r"\.pdf($|\?)", re.IGNORECASE)
YEAR_RE = re.compile(r"(19\d{2}|20\d{2})")


class PdfScraper:
    def __init__(self, settings: Settings, registry: Registry) -> None:
        self.settings = settings
        self.registry = registry
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": settings.user_agent})

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=12), reraise=True)
    def _get(self, url: str) -> requests.Response:
        response = self.session.get(url, timeout=self.settings.request_timeout)
        response.raise_for_status()
        return response

    def crawl_pdf_links(self) -> list[dict]:
        logger.info("Iniciando crawl desde %s", self.settings.base_url)
        to_visit = [self.settings.base_url]
        visited: set[str] = set()
        found_pdfs: list[dict] = []

        while to_visit:
            url = to_visit.pop(0)
            if url in visited:
                continue
            visited.add(url)
            try:
                response = self._get(url)
            except Exception as exc:
                logger.exception("Error al obtener %s: %s", url, exc)
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            for anchor in soup.find_all("a", href=True):
                href = anchor.get("href", "").strip()
                full_url = urljoin(url, href)
                if full_url in visited:
                    continue

                if PDF_RE.search(full_url):
                    metadata = self._build_pdf_metadata(full_url, anchor.get_text(" ", strip=True), parent_url=url)
                    if metadata["year"] is None or metadata["year"] >= self.settings.min_year:
                        found_pdfs.append(metadata)
                elif self._looks_like_index_page(full_url):
                    to_visit.append(full_url)

            time.sleep(self.settings.request_delay_seconds)

        logger.info("Crawl finalizado. PDFs encontrados: %s", len(found_pdfs))
        unique = {item["url"]: item for item in found_pdfs}
        return list(unique.values())

    @staticmethod
    def _looks_like_index_page(url: str) -> bool:
        parsed = urlparse(url)
        if not parsed.netloc.endswith("csjn.gov.ar"):
            return False
        if parsed.path.endswith(".pdf"):
            return False
        return any(key in url.lower() for key in ["fallos", "tomo", "indice", "method="])

    def _build_pdf_metadata(self, pdf_url: str, anchor_text: str, parent_url: str) -> dict:
        filename = Path(urlparse(pdf_url).path).name or "fallo.pdf"
        source_text = " ".join([pdf_url, anchor_text, parent_url])
        year_match = YEAR_RE.search(source_text)
        year = int(year_match.group(1)) if year_match else None
        tomo = self._extract_tomo(source_text)
        return {
            "url": pdf_url,
            "file_name": filename,
            "year": year,
            "tomo": tomo,
            "anchor_text": anchor_text,
            "parent_url": parent_url,
        }

    @staticmethod
    def _extract_tomo(text: str) -> str | None:
        lower = text.lower()
        match = re.search(r"tomo\s*(\d+)", lower)
        return match.group(1) if match else None

    def download_pdfs(self, pdf_items: list[dict]) -> None:
        for item in pdf_items:
            source_url = item["url"]
            known = self.registry.get_pdf_by_url(source_url)
            if known and known["status"] == "downloaded" and Path(known["file_path"]).exists():
                logger.debug("Saltando ya descargado: %s", source_url)
                continue

            year_dir = str(item["year"] or "desconocido")
            target_dir = self.settings.pdf_dir / year_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / item["file_name"]

            try:
                self._download_file(source_url, target_path)
                self.registry.upsert_pdf_record(
                    source_url=source_url,
                    file_path=str(target_path),
                    file_name=item["file_name"],
                    tomo=item["tomo"],
                    year=item["year"],
                    status="downloaded",
                )
                logger.info("Descargado %s", target_path)
            except Exception as exc:
                self.registry.upsert_pdf_record(
                    source_url=source_url,
                    file_path=str(target_path),
                    file_name=item["file_name"],
                    tomo=item["tomo"],
                    year=item["year"],
                    status="error",
                    error_message=str(exc),
                )
                logger.exception("Error descargando %s", source_url)
            time.sleep(self.settings.request_delay_seconds)

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=10), reraise=True)
    def _download_file(self, url: str, target_path: Path) -> None:
        with self.session.get(url, stream=True, timeout=self.settings.request_timeout) as response:
            response.raise_for_status()
            with target_path.open("wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 128):
                    if chunk:
                        f.write(chunk)
