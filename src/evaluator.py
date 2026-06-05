from __future__ import annotations

import csv
from pathlib import Path

from .config import COLLECTION_NAME, DEFAULT_TOP_K, EVALUATION_STORAGE_DIR, OUTPUTS_DIR
from .data_loader import format_hours, sort_tasks_for_split
from .estimator import estimate_from_matches, query_text
from .planning_views import format_story_point_label, story_points_for_hours
from .vector_store import TicketVectorStore


def evaluate(
    train_tasks: list[dict],
    ollama_client,
    test_tasks: list[dict] | None = None,
    output_path: Path | None = None,
    evaluation_storage_dir: Path = EVALUATION_STORAGE_DIR,
    top_k: int = DEFAULT_TOP_K,
    test_ratio: float = 0.2,
) -> dict:
    if test_tasks is None:
        sorted_tasks = sort_tasks_for_split(train_tasks)
        test_size = max(1, round(len(sorted_tasks) * test_ratio))

        if len(sorted_tasks) - test_size < 2:
            raise ValueError("Need at least two training tickets for evaluation.")

        train_tasks = sorted_tasks[:-test_size]
        test_tasks = sorted_tasks[-test_size:]
    elif not test_tasks:
        raise ValueError("Test CSV contains no valid task rows.")

    assert test_tasks is not None
    guard_against_test_leakage(train_tasks, test_tasks)

    store = TicketVectorStore(evaluation_storage_dir, COLLECTION_NAME)
    store.index_tasks(train_tasks, ollama_client)

    global_average = sum(task["actual_hours"] for task in train_tasks) / len(train_tasks)
    global_median = median([task["actual_hours"] for task in train_tasks])
    global_average_story_points = story_points_for_hours(global_average)
    global_median_story_points = story_points_for_hours(global_median)

    rows = []
    for task in test_tasks:
        query = query_text(f"{task['title']}. {task['description']}")
        matches = store.query(query, ollama_client, top_k=min(top_k, len(train_tasks)))
        rag_stats = estimate_from_matches(matches)
        actual = float(task["actual_hours"])
        actual_story_points = story_points_for_hours(actual)
        rag_story_points = story_points_for_hours(rag_stats["point_estimate"])

        rows.append(
            {
                "task_id": task["task_id"],
                "title": task["title"],
                "actual_hours": actual,
                "actual_story_points": actual_story_points,
                "global_average_prediction": global_average,
                "global_average_absolute_error": abs(global_average - actual),
                "global_average_story_points_prediction": global_average_story_points,
                "global_average_story_points_absolute_error": abs(
                    global_average_story_points - actual_story_points
                ),
                "global_median_prediction": global_median,
                "global_median_absolute_error": abs(global_median - actual),
                "global_median_story_points_prediction": global_median_story_points,
                "global_median_story_points_absolute_error": abs(
                    global_median_story_points - actual_story_points
                ),
                "rag_median_prediction": rag_stats["point_estimate"],
                "rag_median_absolute_error": abs(
                    rag_stats["point_estimate"] - actual
                ),
                "rag_median_story_points_prediction": rag_story_points,
                "rag_median_story_points_absolute_error": abs(
                    rag_story_points - actual_story_points
                ),
                "rag_confidence": rag_stats["confidence"],
                "retrieved_task_ids": " ".join(
                    match["metadata"]["task_id"] for match in matches
                ),
            }
        )

    summary = {
        "train_count": len(train_tasks),
        "test_count": len(test_tasks),
        "evaluation_storage_dir": evaluation_storage_dir,
        "global_average_mae": mean(
            row["global_average_absolute_error"] for row in rows
        ),
        "global_average_story_points_mae": mean(
            row["global_average_story_points_absolute_error"] for row in rows
        ),
        "global_median_mae": mean(row["global_median_absolute_error"] for row in rows),
        "global_median_story_points_mae": mean(
            row["global_median_story_points_absolute_error"] for row in rows
        ),
        "rag_median_mae": mean(row["rag_median_absolute_error"] for row in rows),
        "rag_median_story_points_mae": mean(
            row["rag_median_story_points_absolute_error"] for row in rows
        ),
        "rows": rows,
    }

    output_path = output_path or OUTPUTS_DIR / "evaluation_results.csv"
    write_evaluation_results(output_path, rows)
    summary["output_path"] = output_path

    return summary


def guard_against_test_leakage(train_tasks: list[dict], test_tasks: list[dict]) -> None:
    train_ids = {task["task_id"] for task in train_tasks}
    duplicate_ids = sorted(train_ids.intersection(task["task_id"] for task in test_tasks))
    if duplicate_ids:
        duplicates = ", ".join(duplicate_ids)
        raise ValueError(
            f"Evaluation test tickets are also present in training data: {duplicates}"
        )


def write_evaluation_results(output_path: Path, rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_evaluation(summary: dict) -> str:
    return "\n".join(
        [
            "Evaluation results",
            "",
            f"Train tickets: {summary['train_count']}",
            f"Test tickets: {summary['test_count']}",
            f"Evaluation index: {summary['evaluation_storage_dir']}",
            "",
            f"Global average MAE: {format_hours(summary['global_average_mae'])}h",
            "Global average derived story point MAE: "
            f"{format_story_point_label(summary['global_average_story_points_mae'])}",
            f"Global median MAE: {format_hours(summary['global_median_mae'])}h",
            "Global median derived story point MAE: "
            f"{format_story_point_label(summary['global_median_story_points_mae'])}",
            f"RAG median MAE: {format_hours(summary['rag_median_mae'])}h",
            "RAG median derived story point MAE: "
            f"{format_story_point_label(summary['rag_median_story_points_mae'])}",
            "",
            f"Saved: {summary['output_path']}",
        ]
    )


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def median(values: list[float]) -> float:
    sorted_values = sorted(values)
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[midpoint]
    return (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2
