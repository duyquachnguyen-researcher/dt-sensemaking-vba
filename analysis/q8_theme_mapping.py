"""Analyze how seed themes map to grid regions for Q8.

Uses the theme labels in StatementPub.csv (default column: cluster) to compute
theme-level centroids and dispersion across placements.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
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
    parser = argparse.ArgumentParser(description="Q8 theme mapping analysis")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Placement.csv"),
        help="Path to Placement.csv",
    )
    parser.add_argument(
        "--statements",
        type=Path,
        default=Path("StatementPub.csv"),
        help="Path to StatementPub.csv",
    )
    parser.add_argument(
        "--theme-column",
        default="cluster",
        help="Column in StatementPub.csv that stores the theme label",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/q8_theme_mapping.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("outputs/q8_summary.txt"),
        help="Output summary text path",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top theme contrasts to list",
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


def load_theme_mapping(path: Path, theme_column: str) -> tuple[dict[str, str], int]:
    rows = read_rows(path)
    mapping: dict[str, str] = {}
    missing_theme = 0
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
            missing_theme += 1
            continue
        mapping[statement_id] = theme
    return mapping, missing_theme


def cleaned_rows(rows: Iterable[dict[str, str]], theme_map: dict[str, str]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        token = (row.get("token") or "").strip()
        statement_id = (row.get("canonical_id") or "").strip()
        if not token or not statement_id:
            continue
        theme = theme_map.get(statement_id)
        if not theme:
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
                "theme": theme,
                "x": x_val,
                "y": y_val,
            }
        )
    return cleaned


def theme_stats(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[Any]]] = defaultdict(
        lambda: {"x": [], "y": [], "tokens": [], "statements": []}
    )
    for row in rows:
        theme = row["theme"]
        grouped[theme]["x"].append(row["x"])
        grouped[theme]["y"].append(row["y"])
        grouped[theme]["tokens"].append(row["token"])
        grouped[theme]["statements"].append(row["statement_id"])

    results: list[dict[str, Any]] = []
    for theme, data in grouped.items():
        x_vals = data["x"]
        y_vals = data["y"]
        results.append(
            {
                "theme": theme,
                "n_statements": len(set(data["statements"])),
                "n_participants": len(set(data["tokens"])),
                "n_placements": len(x_vals),
                "mean_x": mean(x_vals),
                "mean_y": mean(y_vals),
                "sd_x": stdev(x_vals) if len(x_vals) > 1 else 0.0,
                "sd_y": stdev(y_vals) if len(y_vals) > 1 else 0.0,
            }
        )
    return results


def centroid_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    return math.sqrt((a["mean_x"] - b["mean_x"]) ** 2 + (a["mean_y"] - b["mean_y"]) ** 2)


def summarize(stats: list[dict[str, Any]], top_n: int) -> str:
    lines = [
        "Q8 Theme Mapping Summary",
        "========================",
        f"Themes included: {len(stats)}",
        "",
        "Theme centroids",
        "--------------",
    ]

    for row in sorted(stats, key=lambda item: (-item["mean_y"], item["mean_x"])):
        lines.append(
            "{theme}: mean_x={mean_x:.2f}, mean_y={mean_y:.2f}, sd_x={sd_x:.2f}, sd_y={sd_y:.2f}, "
            "placements={n_placements}, statements={n_statements}".format(**row)
        )

    if stats:
        comparisons: list[tuple[float, str]] = []
        for i, left in enumerate(stats):
            for right in stats[i + 1 :]:
                distance = centroid_distance(left, right)
                comparisons.append((distance, f"{left['theme']} vs {right['theme']}: {distance:.2f}"))
        comparisons.sort(key=lambda item: item[0], reverse=True)
        lines.extend(
            [
                "",
                "Top centroid contrasts",
                "-----------------------",
            ]
        )
        for _, line in comparisons[: max(top_n, 0)]:
            lines.append(line)
    else:
        lines.append("No themes had placements after filtering.")

    return "\n".join(lines)


def write_csv(path: Path, stats: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "theme",
        "n_statements",
        "n_participants",
        "n_placements",
        "mean_x",
        "mean_y",
        "sd_x",
        "sd_y",
    ]
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

    theme_map, missing_theme = load_theme_mapping(args.statements, args.theme_column)
    cleaned = cleaned_rows(deduped, theme_map)
    stats = theme_stats(cleaned)

    write_csv(args.output_csv, stats)

    summary_text = summarize(stats, args.top_n)
    participants = {row["token"] for row in cleaned}
    statements_with_data = {row["statement_id"] for row in cleaned}

    sanity_lines = [
        "",
        "Sanity checks",
        "-------------",
        f"Rows in input: {raw_rows}",
        f"Rows after dedup: {dedup_rows}",
        f"Rows after theme filter: {len(cleaned)}",
        f"Participants (unique tokens): {len(participants)}",
        f"Statements with theme data: {len(statements_with_data)}",
        f"Statements missing theme labels: {missing_theme}",
    ]
    summary_text = summary_text + "\n" + "\n".join(sanity_lines)

    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(summary_text, encoding="utf-8")

    print(summary_text)


if __name__ == "__main__":
    main()
