from __future__ import annotations

import json
import math
import statistics

from .data_loader import format_hours


def query_text(task_description: str) -> str:
    return f"Title: {task_description}\nDescription: {task_description}"


def estimate_from_matches(matches: list[dict]) -> dict:
    if not matches:
        raise ValueError("Cannot estimate without similar historical tickets.")

    hours = [float(match["metadata"]["actual_hours"]) for match in matches]
    average = statistics.mean(hours)
    median = statistics.median(hours)
    minimum = min(hours)
    maximum = max(hours)
    std_dev = statistics.pstdev(hours) if len(hours) > 1 else 0.0
    q1 = percentile(hours, 25)
    q3 = percentile(hours, 75)

    confidence = confidence_for(matches, median, std_dev)

    return {
        "point_estimate": median,
        "range_low": q1,
        "range_high": q3,
        "average": average,
        "median": median,
        "min": minimum,
        "max": maximum,
        "std_dev": std_dev,
        "confidence": confidence,
    }


def percentile(values: list[float], percent: float) -> float:
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * (percent / 100)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)

    if lower_index == upper_index:
        return sorted_values[int(position)]

    lower = sorted_values[lower_index]
    upper = sorted_values[upper_index]
    weight = position - lower_index
    return lower + (upper - lower) * weight


def confidence_for(matches: list[dict], median: float, std_dev: float) -> str:
    top_similarity = matches[0]["similarity"] if matches else 0.0
    enough_matches = len(matches) >= 5
    low_variance = median > 0 and std_dev < median * 0.4

    if top_similarity > 0.75 and enough_matches and low_variance:
        return "high"
    if top_similarity > 0.55 and len(matches) >= 3:
        return "medium"
    return "low"


def build_explanation_prompt(
    new_task: str,
    matches: list[dict],
    stats: dict,
) -> str:
    evidence = []
    for match in matches:
        metadata = match["metadata"]
        evidence.append(
            {
                "task_id": metadata["task_id"],
                "title": metadata["title"],
                "description": metadata["description"],
                "type": metadata.get("type", ""),
                "project": metadata.get("project", ""),
                "module": metadata.get("module", ""),
                "actual_hours": metadata["actual_hours"],
                "similarity": round(match["similarity"], 3),
            }
        )

    payload = {
        "new_task": new_task,
        "suggested_estimate": {
            "range": f"{format_hours(stats['range_low'])}-{format_hours(stats['range_high'])} hours",
            "point_estimate": stats["point_estimate"],
            "confidence": stats["confidence"],
        },
        "similar_historical_tickets": evidence,
        "calculated_statistics": {
            "median_actual_hours": stats["median"],
            "average_actual_hours": stats["average"],
            "min_actual_hours": stats["min"],
            "max_actual_hours": stats["max"],
            "std_dev_actual_hours": stats["std_dev"],
        },
    }

    return (
        "Use this JSON evidence to explain the suggested estimate. "
        "Do not change the numeric estimate. Do not invent tickets. "
        "Return only JSON with keys: reasoning, risks, similar_tasks_used. "
        "The risks value must be an array of short strings, not one string.\n\n"
        f"{json.dumps(payload, indent=2)}"
    )


def normalize_text_list(value) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return []

        lines = [
            line.strip().lstrip("-*0123456789. ").strip()
            for line in cleaned.splitlines()
        ]
        lines = [line for line in lines if line]
        return lines or [cleaned]

    if isinstance(value, list):
        normalized = []
        for item in value:
            if isinstance(item, str):
                cleaned = item.strip()
            elif isinstance(item, dict):
                cleaned = json.dumps(item, ensure_ascii=True)
            else:
                cleaned = str(item).strip()

            if cleaned:
                normalized.append(cleaned)

        return normalized

    return [str(value).strip()] if str(value).strip() else []


def render_estimate(
    new_task: str,
    matches: list[dict],
    stats: dict,
    explanation: dict | None = None,
) -> str:
    lines = [
        "New task:",
        new_task,
        "",
        "Suggested estimate:",
        f"{format_hours(stats['range_low'])}-{format_hours(stats['range_high'])} hours",
        "",
        "Point estimate:",
        f"{format_hours(stats['point_estimate'])} hours",
        "",
        "Confidence:",
        stats["confidence"].title(),
        "",
        "Similar historical tickets:",
    ]

    for index, match in enumerate(matches, start=1):
        metadata = match["metadata"]
        lines.append(
            f"{index}. {metadata['title']} - actual: "
            f"{format_hours(float(metadata['actual_hours']))}h - "
            f"similarity: {match['similarity']:.2f}"
        )

    lines.extend(
        [
            "",
            "Statistics:",
            f"Median: {format_hours(stats['median'])}h",
            f"Average: {format_hours(stats['average'])}h",
            f"Min/Max: {format_hours(stats['min'])}h/{format_hours(stats['max'])}h",
            f"Std dev: {format_hours(stats['std_dev'])}h",
        ]
    )

    if explanation:
        reasoning = explanation.get("reasoning")
        risks = normalize_text_list(explanation.get("risks"))

        if reasoning:
            lines.extend(["", "Reason:", str(reasoning).strip()])

        if risks:
            lines.extend(["", "Risks:"])
            for risk in risks:
                lines.append(f"- {risk}")

    return "\n".join(lines)
