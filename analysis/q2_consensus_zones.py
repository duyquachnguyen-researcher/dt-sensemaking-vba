"""Analyze consensus zones for Q2.

Aggregates seed statement placements into a 7x7 grid and reports counts and
percentages per cell.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
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
    parser = argparse.ArgumentParser(description="Q2 consensus zone analysis")
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
        "--output-csv",
        type=Path,
        default=Path("outputs/q2_grid_counts.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("outputs/q2_summary.txt"),
        help="Output summary text path",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top cells to list",
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


def coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
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


def load_seed_ids(path: Path) -> set[str]:
    rows = read_rows(path)
    seed_ids: set[str] = set()
    for row in rows:
        statement_id = (row.get("id") or "").strip()
        if statement_id:
            seed_ids.add(statement_id)
    return seed_ids


def cleaned_rows(rows: Iterable[dict[str, str]], seed_ids: set[str]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        token = (row.get("token") or "").strip()
        statement_id = (row.get("canonical_id") or "").strip()
        if not token or not statement_id:
            continue
        if statement_id not in seed_ids:
            continue
        x_val = coerce_int(row.get("x"))
        y_val = coerce_int(row.get("y"))
        if x_val is None or y_val is None:
            continue
        if not (GRID_MIN <= x_val <= GRID_MAX and GRID_MIN <= y_val <= GRID_MAX):
            continue
        cleaned.append({"token": token, "statement_id": statement_id, "x": x_val, "y": y_val})
    return cleaned


def cell_id_for(x_val: int, y_val: int) -> int:
    return (x_val - 1) * GRID_MAX + y_val


def compute_grid(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    counts: dict[tuple[int, int], dict[str, Any]] = defaultdict(lambda: {"count": 0, "tokens": set()})
    for row in rows:
        key = (row["x"], row["y"])
        counts[key]["count"] += 1
        counts[key]["tokens"].add(row["token"])

    results: list[dict[str, Any]] = []
    for x_val in range(GRID_MIN, GRID_MAX + 1):
        for y_val in range(GRID_MIN, GRID_MAX + 1):
            cell = counts.get((x_val, y_val), {"count": 0, "tokens": set()})
            results.append(
                {
                    "x": x_val,
                    "y": y_val,
                    "cell_id": cell_id_for(x_val, y_val),
                    "count": cell["count"],
                    "participant_count": len(cell["tokens"]),
                }
            )
    totals = {
        "total_placements": sum(row["count"] for row in results),
        "total_participants": len({token for row in rows for token in [row["token"]]}),
    }
    return results, totals


def add_percentages(
    grid: list[dict[str, Any]],
    total_placements: int,
    total_participants: int,
) -> list[dict[str, Any]]:
    for row in grid:
        row["percent_of_placements"] = (
            (row["count"] / total_placements) * 100 if total_placements else 0.0
        )
        row["percent_of_participants"] = (
            (row["participant_count"] / total_participants) * 100 if total_participants else 0.0
        )
    return grid


def write_grid_csv(path: Path, grid: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "x",
        "y",
        "cell_id",
        "count",
        "percent_of_placements",
        "participant_count",
        "percent_of_participants",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in grid:
            writer.writerow(row)


def format_summary(
    totals: dict[str, int],
    grid: list[dict[str, Any]],
    top_n: int,
    raw_rows: int,
    dedup_rows: int,
    seed_rows: int,
    cleaned_rows_count: int,
) -> str:
    non_empty = [row for row in grid if row["count"] > 0]
    sorted_cells = sorted(
        non_empty,
        key=lambda row: (row["count"], row["participant_count"], row["cell_id"]),
        reverse=True,
    )
    top_cells = sorted_cells[: max(top_n, 0)]
    top_total = sum(row["count"] for row in top_cells)
    top_percent = (top_total / totals["total_placements"] * 100) if totals["total_placements"] else 0.0

    lines = [
        "Q2 Consensus Zones Summary",
        "===========================",
        f"Seed placements analyzed: {totals['total_placements']}",
        f"Participants represented: {totals['total_participants']}",
        f"Non-empty cells: {len(non_empty)} of {GRID_MAX * GRID_MAX}",
        f"Top {top_n} cells share: {top_total} placements ({top_percent:.2f}%)",
        "",
        "Top cells by placement count",
        "-----------------------------",
    ]
    if top_cells:
        for row in top_cells:
            lines.append(
                "Cell ({x},{y}) [ID {cell_id}]: {count} placements ({p_pct:.2f}%), "
                "{p_count} participants ({pp_pct:.2f}%)".format(
                    x=row["x"],
                    y=row["y"],
                    cell_id=row["cell_id"],
                    count=row["count"],
                    p_pct=row["percent_of_placements"],
                    p_count=row["participant_count"],
                    pp_pct=row["percent_of_participants"],
                )
            )
    else:
        lines.append("No placements available.")

    sanity_lines = [
        "",
        "Sanity checks",
        "-------------",
        f"Rows in input: {raw_rows}",
        f"Rows after dedup: {dedup_rows}",
        f"Rows after seed filter: {seed_rows}",
        f"Rows after cleaning x/y: {cleaned_rows_count}",
    ]
    return "\n".join(lines + sanity_lines)


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    raw_rows = len(rows)

    deduped = deduplicate_latest(rows)
    dedup_rows = len(deduped)

    seed_ids = load_seed_ids(args.statements)
    seed_rows = [row for row in deduped if (row.get("canonical_id") or "").strip() in seed_ids]

    cleaned = cleaned_rows(deduped, seed_ids)

    grid, totals = compute_grid(cleaned)
    grid = add_percentages(grid, totals["total_placements"], totals["total_participants"])

    write_grid_csv(args.output_csv, grid)
    summary_text = format_summary(
        totals,
        grid,
        args.top_n,
        raw_rows,
        dedup_rows,
        len(seed_rows),
        len(cleaned),
    )
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(summary_text, encoding="utf-8")

    print(summary_text)


if __name__ == "__main__":
    main()
