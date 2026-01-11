"""Assess raw statement alignment with seed priorities for Q20."""

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
QUADRANT_CUTOFF = 4.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q20 raw alignment analysis")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("inputs/Placement.csv"),
        help="Path to Placement.csv",
    )
    parser.add_argument(
        "--seed-statements",
        type=Path,
        default=Path("StatementPub.csv"),
        help="Path to StatementPub.csv",
    )
    parser.add_argument(
        "--raw-statements",
        type=Path,
        default=Path("StatementRaw.csv"),
        help="Path to StatementRaw.csv",
    )
    parser.add_argument(
        "--theme-column",
        default="cluster",
        help="Theme column in StatementPub.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/q20_raw_alignment.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("outputs/q20_summary.txt"),
        help="Output summary text path",
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


def load_theme_mapping(path: Path, theme_column: str) -> dict[str, str]:
    rows = read_rows(path)
    mapping: dict[str, str] = {}
    for row in rows:
        statement_id = (row.get("id") or "").strip()
        if not statement_id:
            continue
        theme = (row.get(theme_column) or "").strip()
        if not theme and theme_column != "cluster":
            theme = (row.get("cluster") or "").strip()
        if not theme:
            theme = (row.get("tags") or "").strip()
        if not theme:
            continue
        mapping[statement_id] = theme
    return mapping


def load_raw_texts(path: Path) -> dict[str, str]:
    rows = read_rows(path)
    mapping: dict[str, str] = {}
    for row in rows:
        statement_id = (row.get("id") or "").strip()
        text = (row.get("text") or "").strip()
        if statement_id:
            mapping[statement_id] = text
    return mapping


def cleaned_rows(
    rows: Iterable[dict[str, str]],
    seed_ids: set[str],
    raw_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    for row in rows:
        statement_id = (row.get("canonical_id") or "").strip()
        if not statement_id:
            continue
        token = (row.get("token") or "").strip()
        if not token:
            continue
        x_val = coerce_float(row.get("x"))
        y_val = coerce_float(row.get("y"))
        if x_val is None or y_val is None:
            continue
        if not (GRID_MIN <= x_val <= GRID_MAX and GRID_MIN <= y_val <= GRID_MAX):
            continue
        entry = {"token": token, "statement_id": statement_id, "x": x_val, "y": y_val}
        if statement_id in seed_ids:
            seed_rows.append(entry)
        elif statement_id in raw_ids:
            raw_rows.append(entry)
    return seed_rows, raw_rows


def theme_centroids(rows: Iterable[dict[str, Any]], theme_map: dict[str, str]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"x": [], "y": []})
    for row in rows:
        theme = theme_map.get(row["statement_id"])
        if not theme:
            continue
        grouped[theme]["x"].append(row["x"])
        grouped[theme]["y"].append(row["y"])

    centroids: dict[str, dict[str, float]] = {}
    for theme, values in grouped.items():
        if not values["x"]:
            continue
        centroids[theme] = {"mean_x": mean(values["x"]), "mean_y": mean(values["y"])}
    return centroids


def raw_centroids(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"x": [], "y": []})
    for row in rows:
        grouped[row["statement_id"]]["x"].append(row["x"])
        grouped[row["statement_id"]]["y"].append(row["y"])

    return {
        statement_id: {"mean_x": mean(values["x"]), "mean_y": mean(values["y"])}
        for statement_id, values in grouped.items()
        if values["x"]
    }


def centroid_distance(a: dict[str, float], b: dict[str, float]) -> float:
    return math.sqrt((a["mean_x"] - b["mean_x"]) ** 2 + (a["mean_y"] - b["mean_y"]) ** 2)


def quadrant(mean_x: float, mean_y: float) -> str:
    return ("high" if mean_x >= QUADRANT_CUTOFF else "low") + "-" + (
        "high" if mean_y >= QUADRANT_CUTOFF else "low"
    )


def alignment_rows(
    raw_centroid_map: dict[str, dict[str, float]],
    theme_centroid_map: dict[str, dict[str, float]],
    raw_texts: dict[str, str],
) -> list[dict[str, Any]]:
    rows = []
    for statement_id, centroid in raw_centroid_map.items():
        best_theme = ""
        best_distance = float("inf")
        for theme, theme_centroid in theme_centroid_map.items():
            distance = centroid_distance(centroid, theme_centroid)
            if distance < best_distance:
                best_distance = distance
                best_theme = theme
        if not best_theme:
            continue
        theme_centroid = theme_centroid_map[best_theme]
        rows.append(
            {
                "statement_id": statement_id,
                "statement_text": raw_texts.get(statement_id, ""),
                "mean_x": centroid["mean_x"],
                "mean_y": centroid["mean_y"],
                "closest_theme": best_theme,
                "distance": best_distance,
                "raw_quadrant": quadrant(centroid["mean_x"], centroid["mean_y"]),
                "theme_quadrant": quadrant(theme_centroid["mean_x"], theme_centroid["mean_y"]),
                "quadrant_match": quadrant(centroid["mean_x"], centroid["mean_y"]) == quadrant(
                    theme_centroid["mean_x"], theme_centroid["mean_y"]
                ),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "Q20 Raw Alignment Summary\n===========================\nNo alignment rows found."

    avg_distance = mean([row["distance"] for row in rows])
    match_rate = mean([1.0 if row["quadrant_match"] else 0.0 for row in rows])

    lines = [
        "Q20 Raw Alignment Summary",
        "===========================",
        f"Raw statements analyzed: {len(rows)}",
        f"Average distance to closest seed theme: {avg_distance:.2f}",
        f"Quadrant match rate: {match_rate:.2%}",
    ]
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "statement_id",
        "statement_text",
        "mean_x",
        "mean_y",
        "closest_theme",
        "distance",
        "raw_quadrant",
        "theme_quadrant",
        "quadrant_match",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    raw_rows_count = len(rows)

    deduped = deduplicate_latest(rows)
    dedup_rows = len(deduped)

    theme_map = load_theme_mapping(args.seed_statements, args.theme_column)
    raw_texts = load_raw_texts(args.raw_statements)

    seed_ids = set(theme_map)
    raw_ids = set(raw_texts)

    seed_rows, raw_rows = cleaned_rows(deduped, seed_ids, raw_ids)
    theme_centroid_map = theme_centroids(seed_rows, theme_map)
    raw_centroid_map = raw_centroids(raw_rows)

    rows_out = alignment_rows(raw_centroid_map, theme_centroid_map, raw_texts)
    write_csv(args.output_csv, rows_out)

    summary_text = summarize(rows_out)
    sanity_lines = [
        "",
        "Sanity checks",
        "-------------",
        f"Rows in input: {raw_rows_count}",
        f"Rows after dedup: {dedup_rows}",
        f"Seed placements: {len(seed_rows)}",
        f"Raw placements: {len(raw_rows)}",
        f"Seed themes with centroids: {len(theme_centroid_map)}",
    ]
    summary_text = summary_text + "\n" + "\n".join(sanity_lines)

    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(summary_text, encoding="utf-8")
    print(summary_text)


if __name__ == "__main__":
    main()
