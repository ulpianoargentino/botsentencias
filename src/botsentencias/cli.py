from __future__ import annotations

import argparse
import json
import logging

from botsentencias.config import settings
from botsentencias.logging_utils import setup_logging
from botsentencias.registry import Registry


logger = logging.getLogger(__name__)


def cmd_scrape(args: argparse.Namespace) -> None:
    from botsentencias.scraper import PdfScraper

    registry = Registry(settings.db_path)
    scraper = PdfScraper(settings=settings, registry=registry)
    items = scraper.crawl_pdf_links()
    if args.limit:
        items = items[: args.limit]

    if args.discover_only:
        print(json.dumps(items, indent=2, ensure_ascii=False))
        return

    scraper.download_pdfs(items)
    registry.export_pdf_registry_csv(settings.data_dir / "pdf_registry.csv")


def cmd_process(_: argparse.Namespace) -> None:
    from botsentencias.processing import Embedder, PdfProcessor

    registry = Registry(settings.db_path)
    embedder = Embedder(settings=settings)
    processor = PdfProcessor(settings=settings, registry=registry, embedder=embedder)
    processor.process_all_downloaded()


def cmd_ask(args: argparse.Namespace) -> None:
    from botsentencias.rag import RagAgent

    registry = Registry(settings.db_path)
    agent = RagAgent(settings=settings, registry=registry)
    response = agent.ask(args.question, top_k=args.top_k)
    print("\n=== RESPUESTA ===\n")
    print(response["answer"])
    print("\n=== FUENTES ===\n")
    print(json.dumps(response["sources"], indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agente RAG para Fallos CSJN")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scrape = subparsers.add_parser("scrape", help="Crawlea y descarga PDFs")
    scrape.add_argument("--limit", type=int, default=None, help="Limitar cantidad para prueba")
    scrape.add_argument(
        "--discover-only",
        action="store_true",
        help="Solo descubre y muestra enlaces PDF sin descargar.",
    )
    scrape.set_defaults(func=cmd_scrape)

    process = subparsers.add_parser("process", help="Procesa PDFs y genera embeddings")
    process.set_defaults(func=cmd_process)

    ask = subparsers.add_parser("ask", help="Realiza pregunta al agente")
    ask.add_argument("question", type=str, help="Pregunta jurídica")
    ask.add_argument("--top-k", type=int, default=None, help="Top K de recuperación")
    ask.set_defaults(func=cmd_ask)

    return parser


def main() -> None:
    setup_logging()
    settings.ensure_dirs()
    parser = build_parser()
    args = parser.parse_args()
    logger.info("Ejecutando comando: %s", args.command)
    args.func(args)


if __name__ == "__main__":
    main()
