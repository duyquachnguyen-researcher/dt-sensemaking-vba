"""Analyze participant extremity and consistency for Q10."""

from __future__ import annotations

import argparse
import csv
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
EXTREME_VALUES = {GRID_MIN, GRID_MAX}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q10 participant extremity analysis")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("inputs/Placement.csv"),
        help="Path to Placement.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/q10_participant_extremes.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("outputs/q10_summary.txt"),
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


def cleaned_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        token = (row.get("token") or "").strip()
        statement_id = (row.get("canonical_id") or "").strip()
        if not token or not statement_id:
            continue
        x_val = coerce_float(row.get("x"))
        y_val = coerce_float(row.get("y"))
        if x_val is None or y_val is None:
            continue
        if not (GRID_MIN <= x_val <= GRID_MAX and GRID_MIN <= y_val <= GRID_MAX):
            continue
        cleaned.append({"token": token, "statement_id": statement_id, "x": x_val, "y": y_val})
    return cleaned


def participant_stats(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"x": [], "y": []})
    for row in rows:
        grouped[row["token"]]["x"].append(row["x"])
        grouped[row["token"]]["y"].append(row["y"])

    results: list[dict[str, Any]] = []
    for token, data in grouped.items():
        x_vals = data["x"]
        y_vals = data["y"]
        n = len(x_vals)
        extreme_count = sum(1 for x_val, y_val in zip(x_vals, y_vals) if x_val in EXTREME_VALUES or y_val in EXTREME_VALUES)
        results.append(
            {
                "token": token,
                "n_placements": n,
                "extreme_rate": extreme_count / n if n else 0.0,
                "sd_x": stdev(x_vals) if len(x_vals) > 1 else 0.0,
                "sd_y": stdev(y_vals) if len(y_vals) > 1 else 0.0,
            }
        )
    return results


def summarize(stats: list[dict[str, Any]]) -> str:
    if not stats:
        return "Q10 Extremity Summary\n======================\nNo participant data found."

    extreme_rates = [row["extreme_rate"] for row in stats]
    sd_vals = [(row["sd_x"] + row["sd_y"]) / 2 for row in stats]

    lines = [
        "Q10 Extremity Summary",
        "======================",
        f"Participants included: {len(stats)}",
        f"Average extreme rate: {mean(extreme_rates):.2%}",
        f"Average within-person SD: {mean(sd_vals):.2f}",
        "",
        "Most extreme participants",
        "--------------------------",
    ]

    for row in sorted(stats, key=lambda item: item["extreme_rate"], reverse=True)[:5]:
        lines.append(
            f"{row['token']}: extreme_rate={row['extreme_rate']:.2%}, sd_x={row['sd_x']:.2f}, sd_y={row['sd_y']:.2f}"
        )

    lines.extend([
        "",
        "Most consistent participants",
        "------------------------------",
    ])

    for row in sorted(stats, key=lambda item: item["sd_x"] + item["sd_y"])[:5]:
        lines.append(
            f"{row['token']}: sd_x={row['sd_x']:.2f}, sd_y={row['sd_y']:.2f}, extreme_rate={row['extreme_rate']:.2%}"
        )

    return "\n".join(lines)


def write_csv(path: Path, stats: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["token", "n_placements", "extreme_rate", "sd_x", "sd_y"]
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

    cleaned = cleaned_rows(deduped)
    stats = participant_stats(cleaned)

    write_csv(args.output_csv, stats)

    summary_text = summarize(stats)
    sanity_lines = [
        "",
        "Sanity checks",
        "-------------",
        f"Rows in input: {raw_rows}",
        f"Rows after dedup: {dedup_rows}",
        f"Rows after cleaning: {len(cleaned)}",
    ]
    summary_text = summary_text + "\n" + "\n".join(sanity_lines)

    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(summary_text, encoding="utf-8")
    print(summary_text)


if __name__ == "__main__":
    main()
