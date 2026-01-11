"""Correlate hybrid percentage with placement patterns for Q17."""

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
    parser = argparse.ArgumentParser(description="Q17 hybrid correlation analysis")
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
        default=Path("outputs/q17_hybrid_correlations.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("outputs/q17_summary.txt"),
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


def cleaned_rows(rows: Iterable[dict[str, str]], hybrid_map: dict[str, float]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        token = (row.get("token") or "").strip()
        statement_id = (row.get("canonical_id") or "").strip()
        if not token or not statement_id:
            continue
        hybrid_pct = hybrid_map.get(token)
        if hybrid_pct is None:
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
                "hybrid_pct": hybrid_pct,
                "x": x_val,
                "y": y_val,
            }
        )
    return cleaned


def load_hybrid_map(path: Path) -> dict[str, float]:
    rows = read_rows(path)
    mapping: dict[str, float] = {}
    for row in rows:
        token = (row.get("token") or "").strip()
        hybrid = coerce_float(row.get("hybrid_pct"))
        if token and hybrid is not None:
            mapping[token] = hybrid
    return mapping


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


def statement_correlations(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"hybrid": [], "x": [], "y": []})
    for row in rows:
        data = grouped[row["statement_id"]]
        data["hybrid"].append(row["hybrid_pct"])
        data["x"].append(row["x"])
        data["y"].append(row["y"])

    results: list[dict[str, Any]] = []
    for statement_id, data in grouped.items():
        corr_x = pearson_corr(data["hybrid"], data["x"])
        corr_y = pearson_corr(data["hybrid"], data["y"])
        results.append(
            {
                "statement_id": statement_id,
                "n": len(data["hybrid"]),
                "corr_x": corr_x,
                "corr_y": corr_y,
            }
        )
    return results


def participant_correlations(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"hybrid": [], "x": [], "y": []})
    for row in rows:
        grouped[row["token"]]["hybrid"].append(row["hybrid_pct"])
        grouped[row["token"]]["x"].append(row["x"])
        grouped[row["token"]]["y"].append(row["y"])

    hybrid_vals = []
    mean_x_vals = []
    mean_y_vals = []
    for token, data in grouped.items():
        hybrid_vals.append(mean(data["hybrid"]))
        mean_x_vals.append(mean(data["x"]))
        mean_y_vals.append(mean(data["y"]))

    return {
        "corr_mean_x": pearson_corr(hybrid_vals, mean_x_vals),
        "corr_mean_y": pearson_corr(hybrid_vals, mean_y_vals),
    }


def summarize(stats: list[dict[str, Any]], participant_corr: dict[str, float], top_n: int) -> str:
    lines = [
        "Q17 Hybrid Correlation Summary",
        "===============================",
        f"Statements included: {len(stats)}",
        f"Correlation hybrid% vs participant mean_x: {participant_corr['corr_mean_x']:.3f}",
        f"Correlation hybrid% vs participant mean_y: {participant_corr['corr_mean_y']:.3f}",
        "",
        "Top statements by |corr_x|",
        "---------------------------",
    ]

    for row in sorted(stats, key=lambda item: abs(item["corr_x"]), reverse=True)[: max(top_n, 0)]:
        lines.append(f"{row['statement_id']}: corr_x={row['corr_x']:.3f}, corr_y={row['corr_y']:.3f}")

    return "\n".join(lines)


def write_csv(path: Path, stats: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["statement_id", "n", "corr_x", "corr_y"]
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

    hybrid_map = load_hybrid_map(args.participants)
    cleaned = cleaned_rows(deduped, hybrid_map)
    stats = statement_correlations(cleaned)
    participant_corr = participant_correlations(cleaned)

    write_csv(args.output_csv, stats)

    summary_text = summarize(stats, participant_corr, args.top_n)
    sanity_lines = [
        "",
        "Sanity checks",
        "-------------",
        f"Rows in input: {raw_rows}",
        f"Rows after dedup: {dedup_rows}",
        f"Rows after hybrid filter: {len(cleaned)}",
        f"Statements with hybrid data: {len(stats)}",
    ]
    summary_text = summary_text + "\n" + "\n".join(sanity_lines)

    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(summary_text, encoding="utf-8")
    print(summary_text)


if __name__ == "__main__":
    main()
