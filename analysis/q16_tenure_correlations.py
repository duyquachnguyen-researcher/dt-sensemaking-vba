"""Correlate tenure with placement tendencies for Q16."""

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
    parser = argparse.ArgumentParser(description="Q16 tenure correlation analysis")
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
        default=Path("outputs/q16_tenure_correlations.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("outputs/q16_summary.txt"),
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


def load_tenure_map(path: Path) -> dict[str, float]:
    rows = read_rows(path)
    mapping: dict[str, float] = {}
    for row in rows:
        token = (row.get("token") or "").strip()
        tenure = coerce_float(row.get("tenure_years"))
        if token and tenure is not None:
            mapping[token] = tenure
    return mapping


def participant_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"x": [], "y": []})
    for row in rows:
        grouped[row["token"]]["x"].append(row["x"])
        grouped[row["token"]]["y"].append(row["y"])

    metrics: dict[str, dict[str, float]] = {}
    for token, values in grouped.items():
        x_vals = values["x"]
        y_vals = values["y"]
        n = len(x_vals)
        extreme = sum(1 for x_val, y_val in zip(x_vals, y_vals) if x_val in (1, 7) or y_val in (1, 7))
        metrics[token] = {
            "n_placements": n,
            "mean_x": mean(x_vals),
            "mean_y": mean(y_vals),
            "sd_x": stdev(x_vals) if len(x_vals) > 1 else 0.0,
            "sd_y": stdev(y_vals) if len(y_vals) > 1 else 0.0,
            "extreme_rate": extreme / n if n else 0.0,
        }
    return metrics


def pearson_corr(values_x: list[float], values_y: list[float]) -> float:
    if len(values_x) < 2 or len(values_y) < 2:
        return float("nan")
    mean_x = mean(values_x)
    mean_y = mean(values_y)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(values_x, values_y))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in values_x))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in values_y))
    if denom_x == 0 or denom_y == 0:
        return float("nan")
    return num / (denom_x * denom_y)


def summarize(pairs: list[tuple[float, dict[str, float]]]) -> str:
    tenure_vals = [item[0] for item in pairs]
    mean_x_vals = [item[1]["mean_x"] for item in pairs]
    mean_y_vals = [item[1]["mean_y"] for item in pairs]
    extreme_vals = [item[1]["extreme_rate"] for item in pairs]

    corr_x = pearson_corr(tenure_vals, mean_x_vals)
    corr_y = pearson_corr(tenure_vals, mean_y_vals)
    corr_extreme = pearson_corr(tenure_vals, extreme_vals)

    lines = [
        "Q16 Tenure Correlation Summary",
        "===============================",
        f"Participants included: {len(pairs)}",
        f"Correlation tenure vs mean_x: {corr_x:.3f}",
        f"Correlation tenure vs mean_y: {corr_y:.3f}",
        f"Correlation tenure vs extreme_rate: {corr_extreme:.3f}",
    ]
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "token",
        "tenure_years",
        "n_placements",
        "mean_x",
        "mean_y",
        "sd_x",
        "sd_y",
        "extreme_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    raw_rows = len(rows)

    deduped = deduplicate_latest(rows)
    dedup_rows = len(deduped)

    cleaned = cleaned_rows(deduped)
    tenure_map = load_tenure_map(args.participants)
    metrics = participant_metrics(cleaned)

    paired = []
    output_rows = []
    for token, data in metrics.items():
        tenure = tenure_map.get(token)
        if tenure is None:
            continue
        paired.append((tenure, data))
        output_rows.append({"token": token, "tenure_years": tenure, **data})

    write_csv(args.output_csv, output_rows)

    summary_text = summarize(paired)
    sanity_lines = [
        "",
        "Sanity checks",
        "-------------",
        f"Rows in input: {raw_rows}",
        f"Rows after dedup: {dedup_rows}",
        f"Rows after cleaning: {len(cleaned)}",
        f"Participants with tenure data: {len(paired)}",
    ]
    summary_text = summary_text + "\n" + "\n".join(sanity_lines)

    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(summary_text, encoding="utf-8")
    print(summary_text)


if __name__ == "__main__":
    main()
