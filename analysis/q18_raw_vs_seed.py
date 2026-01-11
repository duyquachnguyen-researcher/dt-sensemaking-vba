"""Compare raw vs seed placement distributions for Q18."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
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
    parser = argparse.ArgumentParser(description="Q18 raw vs seed comparison")
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
        "--output-csv",
        type=Path,
        default=Path("outputs/q18_raw_vs_seed.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("outputs/q18_summary.txt"),
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


def load_statement_ids(path: Path, id_column: str = "id") -> set[str]:
    rows = read_rows(path)
    return {(row.get(id_column) or "").strip() for row in rows if (row.get(id_column) or "").strip()}


def classify_rows(
    rows: Iterable[dict[str, str]],
    seed_ids: set[str],
    raw_ids: set[str],
) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        token = (row.get("token") or "").strip()
        statement_id = (row.get("canonical_id") or "").strip()
        if not token or not statement_id:
            continue
        if statement_id in seed_ids:
            category = "seed"
        elif statement_id in raw_ids:
            category = "raw"
        else:
            continue
        x_val = coerce_float(row.get("x"))
        y_val = coerce_float(row.get("y"))
        cell_id = (row.get("cell_id") or "").strip()
        if x_val is None or y_val is None:
            continue
        if not (GRID_MIN <= x_val <= GRID_MAX and GRID_MIN <= y_val <= GRID_MAX):
            continue
        cleaned.append(
            {
                "token": token,
                "statement_id": statement_id,
                "category": category,
                "x": x_val,
                "y": y_val,
                "cell_id": cell_id,
            }
        )
    return cleaned


def cell_counts(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)
    for row in rows:
        key = (row["category"], row["cell_id"])
        grouped[key] += 1
        totals[row["category"]] += 1

    results = []
    for (category, cell_id), count in grouped.items():
        total = totals[category]
        results.append(
            {
                "category": category,
                "cell_id": cell_id,
                "count": count,
                "pct": count / total if total else 0.0,
            }
        )
    return results


def category_summary(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"x": [], "y": []})
    for row in rows:
        grouped[row["category"]]["x"].append(row["x"])
        grouped[row["category"]]["y"].append(row["y"])

    summary: dict[str, dict[str, float]] = {}
    for category, values in grouped.items():
        x_vals = values["x"]
        y_vals = values["y"]
        n = len(x_vals)
        extreme = sum(1 for x_val, y_val in zip(x_vals, y_vals) if x_val in (1, 7) or y_val in (1, 7))
        summary[category] = {
            "n": n,
            "mean_x": mean(x_vals) if x_vals else 0.0,
            "mean_y": mean(y_vals) if y_vals else 0.0,
            "median_x": median(x_vals) if x_vals else 0.0,
            "median_y": median(y_vals) if y_vals else 0.0,
            "extreme_rate": extreme / n if n else 0.0,
        }
    return summary


def summarize(category_stats: dict[str, dict[str, float]]) -> str:
    lines = [
        "Q18 Raw vs Seed Summary",
        "========================",
    ]
    for category in ("seed", "raw"):
        data = category_stats.get(category)
        if not data:
            lines.append(f"{category.title()}: no data")
            continue
        lines.append(
            "{cat}: n={n}, mean_x={mean_x:.2f}, mean_y={mean_y:.2f}, "
            "median_x={median_x:.2f}, median_y={median_y:.2f}, extreme_rate={extreme_rate:.2%}".format(
                cat=category.title(), **data
            )
        )
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["category", "cell_id", "count", "pct"]
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

    seed_ids = load_statement_ids(args.seed_statements, "id")
    raw_ids = load_statement_ids(args.raw_statements, "id")

    cleaned = classify_rows(deduped, seed_ids, raw_ids)
    counts = cell_counts(cleaned)
    stats = category_summary(cleaned)

    write_csv(args.output_csv, counts)

    summary_text = summarize(stats)
    sanity_lines = [
        "",
        "Sanity checks",
        "-------------",
        f"Rows in input: {raw_rows}",
        f"Rows after dedup: {dedup_rows}",
        f"Rows after type filter: {len(cleaned)}",
        f"Seed placements: {stats.get('seed', {}).get('n', 0)}",
        f"Raw placements: {stats.get('raw', {}).get('n', 0)}",
    ]
    summary_text = summary_text + "\n" + "\n".join(sanity_lines)

    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(summary_text, encoding="utf-8")
    print(summary_text)


if __name__ == "__main__":
    main()
