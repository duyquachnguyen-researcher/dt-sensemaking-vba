"""Analyze statement differences by function for Q13."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


TS_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%m/%d/%Y %H:%M:%S",
)

GRID_MIN = 1
GRID_MAX = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q13 function difference analysis")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("inputs/Placement.csv"),
        help="Path to Placement.csv",
    )
    parser.add_argument(
        "--participants",
        type=Path,
        default=Path("Participant.csv"),
        help="Path to Participant.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/q13_statement_function_gaps.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("outputs/q13_summary.txt"),
        help="Output summary text path",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top statements to list",
    )
    return parser.parse_args()


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        pass
    for fmt in TS_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return None


def coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader]


def deduplicate_latest(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    latest: dict[tuple[str, str], tuple[tuple[bool, datetime | None, int], dict[str, str]]] = {}
    for index, row in enumerate(rows):
        token = (row.get("token") or "").strip()
        statement_id = (row.get("canonical_id") or "").strip()
        if not token or not statement_id:
            continue
        ts_value = parse_timestamp(row.get("ts"))
        key = (token, statement_id)
        sort_key = (ts_value is not None, ts_value, index)
        if key not in latest or sort_key > latest[key][0]:
            latest[key] = (sort_key, row)
    return [item[1] for item in latest.values()]


def cleaned_rows(rows: Iterable[dict[str, str]], group_map: dict[str, str]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        token = (row.get("token") or "").strip()
        statement_id = (row.get("canonical_id") or "").strip()
        if not token or not statement_id:
            continue
        group = group_map.get(token)
        if not group:
            continue
        x_val = coerce_float(row.get("x"))
        y_val = coerce_float(row.get("y"))
        if x_val is None or y_val is None:
            continue
        if not (GRID_MIN <= x_val <= GRID_MAX and GRID_MIN <= y_val <= GRID_MAX):
            continue
        cleaned.append(
            {
                "token": token,
                "statement_id": statement_id,
                "group": group,
                "x": x_val,
                "y": y_val,
            }
        )
    return cleaned


def load_group_map(path: Path, column: str) -> dict[str, str]:
    rows = read_rows(path)
    mapping: dict[str, str] = {}
    for row in rows:
        token = (row.get("token") or "").strip()
        group = (row.get(column) or "").strip()
        if token and group:
            mapping[token] = group
    return mapping


def centroid_distance(a: dict[str, float], b: dict[str, float]) -> float:
    return math.sqrt((a["mean_x"] - b["mean_x"]) ** 2 + (a["mean_y"] - b["mean_y"]) ** 2)


def statement_group_gaps(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: {"x": [], "y": []})
    )
    for row in rows:
        grouped[row["statement_id"]][row["group"]]["x"].append(row["x"])
        grouped[row["statement_id"]][row["group"]]["y"].append(row["y"])

    results: list[dict[str, Any]] = []
    for statement_id, group_data in grouped.items():
        centroids = []
        for group, values in group_data.items():
            if not values["x"]:
                continue
            centroids.append(
                {
                    "group": group,
                    "mean_x": mean(values["x"]),
                    "mean_y": mean(values["y"]),
                }
            )
        if len(centroids) < 2:
            continue
        max_gap = 0.0
        min_x = min(item["mean_x"] for item in centroids)
        max_x = max(item["mean_x"] for item in centroids)
        min_y = min(item["mean_y"] for item in centroids)
        max_y = max(item["mean_y"] for item in centroids)
        for i, left in enumerate(centroids):
            for right in centroids[i + 1 :]:
                max_gap = max(max_gap, centroid_distance(left, right))
        results.append(
            {
                "statement_id": statement_id,
                "groups_with_data": len(centroids),
                "max_gap": max_gap,
                "delta_x": max_x - min_x,
                "delta_y": max_y - min_y,
            }
        )
    return results


def summarize(stats: list[dict[str, Any]], top_n: int) -> str:
    lines = [
        "Q13 Function Differences Summary",
        "=================================",
        f"Statements included: {len(stats)}",
        "",
        "Top function-separated statements",
        "----------------------------------",
    ]
    for row in sorted(stats, key=lambda item: item["max_gap"], reverse=True)[: max(top_n, 0)]:
        lines.append(
            f"{row['statement_id']}: max_gap={row['max_gap']:.2f}, delta_x={row['delta_x']:.2f}, delta_y={row['delta_y']:.2f}"
        )
    return "\n".join(lines)


def write_csv(path: Path, stats: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["statement_id", "groups_with_data", "max_gap", "delta_x", "delta_y"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in stats:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    raw_rows = len(rows)

    deduped = deduplicate_latest(rows)
    dedup_rows = len(deduped)

    group_map = load_group_map(args.participants, "function")
    cleaned = cleaned_rows(deduped, group_map)
    stats = statement_group_gaps(cleaned)
    write_csv(args.output_csv, stats)

    summary_text = summarize(stats, args.top_n)
    sanity_lines = [
        "",
        "Sanity checks",
        "-------------",
        f"Rows in input: {raw_rows}",
        f"Rows after dedup: {dedup_rows}",
        f"Rows after group filter: {len(cleaned)}",
        f"Statements with >=2 functions: {len(stats)}",
    ]
    summary_text = summary_text + "\n" + "\n".join(sanity_lines)

    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(summary_text, encoding="utf-8")
    print(summary_text)


if __name__ == "__main__":
    main()
