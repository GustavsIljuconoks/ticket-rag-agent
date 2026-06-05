from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (
    COLLECTION_NAME,
    DATA_PATH,
    DEFAULT_TOP_K,
    EVALUATION_DATA_PATH,
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_MODEL,
    OLLAMA_EMBED_MODEL,
    STORAGE_DIR,
)
from src.data_loader import DataValidationError, load_tasks
from src.estimator import (
    build_explanation_prompt,
    estimate_from_matches,
    query_text,
    render_estimate,
)
from src.evaluator import evaluate, render_evaluation
from src.ollama_client import OllamaClient, OllamaError
from src.vector_store import TicketVectorStore, VectorStoreError


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "index":
            return run_index(args)
        if args.command == "status":
            return run_status(args)
        if args.command == "estimate":
            return run_estimate(args)
        if args.command == "evaluate":
            return run_evaluate(args)
    except (DataValidationError, OllamaError, VectorStoreError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    parser.print_help()
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local RAG task estimation prototype using Ollama and ChromaDB."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DATA_PATH,
        help=f"Task CSV path. Default: {DATA_PATH}",
    )
    parser.add_argument(
        "--storage",
        type=Path,
        default=STORAGE_DIR,
        help=f"Chroma storage path. Default: {STORAGE_DIR}",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of similar tickets to retrieve. Default: {DEFAULT_TOP_K}",
    )

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("index", help="Embed CSV tickets into the persisted Chroma index.")
    subparsers.add_parser("status", help="Show persisted index status.")

    estimate_parser = subparsers.add_parser(
        "estimate", help="Estimate a new task from similar historical tickets."
    )
    estimate_parser.add_argument("task", help="New task title or description.")
    estimate_parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip Ollama chat explanation and render deterministic evidence only.",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Run leakage-safe evaluation against held-out tickets."
    )
    evaluate_parser.add_argument(
        "--test-csv",
        type=Path,
        default=EVALUATION_DATA_PATH,
        help=(
            "Held-out evaluation CSV path. "
            f"Default: {EVALUATION_DATA_PATH}. "
            "If the file is missing, evaluate falls back to a split of --csv."
        ),
    )
    evaluate_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Evaluation results CSV path. Default: outputs/evaluation_results.csv",
    )

    return parser


def make_ollama_client() -> OllamaClient:
    return OllamaClient(
        base_url=OLLAMA_BASE_URL,
        embed_model=OLLAMA_EMBED_MODEL,
        chat_model=OLLAMA_CHAT_MODEL,
    )


def run_index(args) -> int:
    tasks = load_tasks(args.csv)
    store = TicketVectorStore(args.storage, COLLECTION_NAME)
    store.index_tasks(tasks, make_ollama_client())
    print(f"Indexed {len(tasks)} tickets into {args.storage}")
    return 0


def run_status(args) -> int:
    store = TicketVectorStore(args.storage, COLLECTION_NAME)
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Storage: {args.storage}")
    print(f"Indexed tickets: {store.count()}")
    return 0


def run_estimate(args) -> int:
    store = TicketVectorStore(args.storage, COLLECTION_NAME)
    if store.count() == 0:
        raise ValueError("No indexed tickets found. Run: python src/main.py index")

    ollama_client = make_ollama_client()
    matches = store.query(query_text(args.task), ollama_client, min(args.top_k, store.count()))
    stats = estimate_from_matches(matches)

    explanation = None
    if not args.no_llm:
        prompt = build_explanation_prompt(args.task, matches, stats)
        explanation = ollama_client.explain(prompt)

    print(render_estimate(args.task, matches, stats, explanation))
    return 0


def run_evaluate(args) -> int:
    train_tasks = load_tasks(args.csv)
    test_tasks = load_tasks(args.test_csv) if args.test_csv.exists() else None
    summary = evaluate(
        train_tasks=train_tasks,
        ollama_client=make_ollama_client(),
        test_tasks=test_tasks,
        output_path=args.output,
        top_k=args.top_k,
    )
    print(render_evaluation(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
