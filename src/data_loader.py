from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


REQUIRED_COLUMNS = {"task_id", "title", "description", "actual_hours"}
OPTIONAL_DOCUMENT_COLUMNS = [
    "type",
    "project",
    "module",
    "developer",
    "estimate_hours",
    "created_at",
]


class DataValidationError(ValueError):
    """Raised when the task CSV cannot be used for estimation."""


def load_tasks(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        raise DataValidationError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise DataValidationError("CSV file has no header row.")

        missing_columns = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise DataValidationError(f"CSV is missing required columns: {missing}")

        tasks = []
        for line_number, row in enumerate(reader, start=2):
            task = clean_task_row(row, line_number)
            if task is not None:
                tasks.append(task)

    if not tasks:
        raise DataValidationError("CSV contains no valid task rows.")

    return tasks


def clean_task_row(row: dict, line_number: int) -> dict | None:
    title = (row.get("title") or "").strip()
    description = (row.get("description") or "").strip()
    task_id = (row.get("task_id") or "").strip()
    actual_hours_raw = (row.get("actual_hours") or "").strip()

    if not task_id or not title or not description or not actual_hours_raw:
        return None

    try:
        actual_hours = float(actual_hours_raw)
    except ValueError as exc:
        raise DataValidationError(
            f"Invalid actual_hours value on line {line_number}: {actual_hours_raw}"
        ) from exc

    if actual_hours <= 0:
        raise DataValidationError(
            f"actual_hours must be greater than zero on line {line_number}."
        )

    cleaned = {key: (value or "").strip() for key, value in row.items()}
    cleaned["task_id"] = task_id
    cleaned["title"] = title
    cleaned["description"] = description
    cleaned["actual_hours"] = actual_hours
    return cleaned


def create_task_document(task: dict) -> str:
    lines = [
        f"Task ID: {task.get('task_id', '')}",
        f"Title: {task.get('title', '')}",
        f"Description: {task.get('description', '')}",
    ]

    for column in OPTIONAL_DOCUMENT_COLUMNS:
        value = task.get(column)
        if value not in (None, ""):
            label = column.replace("_", " ").title()
            lines.append(f"{label}: {value}")

    lines.append(f"Actual hours: {format_hours(task['actual_hours'])}")
    return "\n".join(lines)


def task_metadata(task: dict) -> dict:
    metadata = {
        "task_id": task["task_id"],
        "title": task["title"],
        "description": task["description"],
        "actual_hours": float(task["actual_hours"]),
    }

    for column in OPTIONAL_DOCUMENT_COLUMNS:
        value = task.get(column)
        if value not in (None, ""):
            metadata[column] = value

    return metadata


def format_hours(value: float) -> str:
    return f"{value:g}"


def sort_tasks_for_split(tasks: Iterable[dict]) -> list[dict]:
    return sorted(tasks, key=lambda task: (task.get("created_at", ""), task["task_id"]))
