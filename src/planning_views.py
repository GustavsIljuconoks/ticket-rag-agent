from __future__ import annotations

from collections.abc import Sequence

from .config import STORY_POINT_MAPPING

StoryPointMapping = Sequence[tuple[float, float]]


def story_points_for_hours(
    hours: float, mapping: StoryPointMapping = STORY_POINT_MAPPING
) -> float:
    if hours <= 0:
        raise ValueError("hours must be greater than zero for story point mapping.")

    for max_hours, points in mapping:
        if hours <= max_hours:
            return float(points)

    raise ValueError("Story point mapping must include an open-ended final band.")


def planning_views_from_stats(stats: dict) -> dict:
    return {
        "time": {
            "range_low": stats["range_low"],
            "range_high": stats["range_high"],
            "point_estimate": stats["point_estimate"],
        },
        "story_points": {
            "range_low": story_points_for_hours(stats["range_low"]),
            "range_high": story_points_for_hours(stats["range_high"]),
            "point_estimate": story_points_for_hours(stats["point_estimate"]),
        },
    }


def format_story_points(value: float) -> str:
    return f"{value:g}"


def format_story_point_label(value: float) -> str:
    unit = "point" if value == 1 else "points"
    return f"{format_story_points(value)} {unit}"
