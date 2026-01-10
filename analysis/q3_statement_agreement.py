"""Analyze statement-level agreement for Q3.

Computes agreement rates for seed statements based on exact cell matches and
placements within a 1-cell radius of the statement centroid.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev
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
    parser = argparse.ArgumentParser(description="Q3 statement agreement analysis")
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
        default=Path("outputs/q3_statement_agreement.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("outputs/q3_summary.txt"),
        help="Output summary text path",
    )
    parser.add_argument(
        "--min-participants",
        type=int,
        default=2,
        help="Minimum participants per statement",
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
        x_val = coerce_float(row.get("x"))
        y_val = coerce_float(row.get("y"))
        if x_val is None or y_val is None:
            continue
        if not (GRID_MIN <= x_val <= GRID_MAX and GRID_MIN <= y_val <= GRID_MAX):
            continue
        cleaned.append({"token": token, "statement_id": statement_id, "x": x_val, "y": y_val})
    return cleaned


def cell_key(x_val: float, y_val: float) -> tuple[int, int]:
    return int(round(x_val)), int(round(y_val))


def agreement_stats(
    rows: Iterable[dict[str, Any]],
    min_participants: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[Any]]] = defaultdict(lambda: {"x": [], "y": [], "cells": [], "tokens": []})
    for row in rows:
        statement_id = row["statement_id"]
        grouped[statement_id]["x"].append(row["x"])
        grouped[statement_id]["y"].append(row["y"])
        grouped[statement_id]["cells"].append(cell_key(row["x"], row["y"]))
        grouped[statement_id]["tokens"].append(row["token"])

    results: list[dict[str, Any]] = []
    for statement_id, data in grouped.items():
        tokens = set(data["tokens"])
        n_participants = len(tokens)
        if n_participants < min_participants:
            continue
        mean_x = mean(data["x"])
        mean_y = mean(data["y"])
        sd_x = stdev(data["x"]) if len(data["x"]) > 1 else float("nan")
        sd_y = stdev(data["y"]) if len(data["y"]) > 1 else float("nan")

        cell_counts: dict[tuple[int, int], int] = defaultdict(int)
        for cell in data["cells"]:
            cell_counts[cell] += 1
        top_cell = max(cell_counts.items(), key=lambda item: item[1])
        top_cell_count = top_cell[1]
        exact_agreement = top_cell_count / len(data["cells"]) if data["cells"] else 0.0

        within_radius = 0
        for x_val, y_val in zip(data["x"], data["y"]):
            if abs(x_val - mean_x) <= 1 and abs(y_val - mean_y) <= 1:
                within_radius += 1
        radius_agreement = within_radius / len(data["x"]) if data["x"] else 0.0

        results.append(
            {
                "statement_id": statement_id,
                "n_participants": n_participants,
                "mean_x": mean_x,
                "mean_y": mean_y,
                "sd_x": sd_x,
                "sd_y": sd_y,
                "top_cell_x": top_cell[0][0],
                "top_cell_y": top_cell[0][1],
                "top_cell_count": top_cell_count,
                "exact_agreement": exact_agreement,
                "radius_agreement": radius_agreement,
            }
        )
    return results


def summarize(stats: list[dict[str, Any]], top_n: int) -> str:
    exact_vals = [row["exact_agreement"] for row in stats]
    radius_vals = [row["radius_agreement"] for row in stats]
    lines = [
        "Q3 Statement Agreement Summary",
        "===============================",
        f"Statements included: {len(stats)}",
        f"Mean exact agreement: {mean(exact_vals):.3f}" if exact_vals else "Mean exact agreement: n/a",
        f"Median exact agreement: {median(exact_vals):.3f}" if exact_vals else "Median exact agreement: n/a",
        f"Mean 1-cell radius agreement: {mean(radius_vals):.3f}" if radius_vals else "Mean 1-cell radius agreement: n/a",
        f"Median 1-cell radius agreement: {median(radius_vals):.3f}" if radius_vals else "Median 1-cell radius agreement: n/a",
        "",
        "Top statements by exact agreement",
        "---------------------------------",
    ]

    sorted_exact = sorted(stats, key=lambda row: (row["exact_agreement"], row["n_participants"]), reverse=True)
    for row in sorted_exact[: max(top_n, 0)]:
        lines.append(
            "{statement_id}: {exact:.2%} exact (n={n}) top cell ({x},{y})".format(
                statement_id=row["statement_id"],
                exact=row["exact_agreement"],
                n=row["n_participants"],
                x=row["top_cell_x"],
                y=row["top_cell_y"],
            )
        )

    lines.extend(
        [
            "",
            "Top statements by 1-cell radius agreement",
            "-----------------------------------------",
        ]
    )
    sorted_radius = sorted(
        stats,
        key=lambda row: (row["radius_agreement"], row["n_participants"]),
        reverse=True,
    )
    for row in sorted_radius[: max(top_n, 0)]:
        lines.append(
            "{statement_id}: {radius:.2%} within 1 cell (n={n})".format(
                statement_id=row["statement_id"],
                radius=row["radius_agreement"],
                n=row["n_participants"],
            )
        )
    return "\n".join(lines)


def write_csv(path: Path, stats: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "statement_id",
        "n_participants",
        "mean_x",
        "mean_y",
        "sd_x",
        "sd_y",
        "top_cell_x",
        "top_cell_y",
        "top_cell_count",
        "exact_agreement",
        "radius_agreement",
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

    seed_ids = load_seed_ids(args.statements)
    seed_rows = [row for row in deduped if (row.get("canonical_id") or "").strip() in seed_ids]

    cleaned = cleaned_rows(deduped, seed_ids)
    stats = agreement_stats(cleaned, args.min_participants)

    write_csv(args.output_csv, stats)

    summary_text = summarize(stats, args.top_n)
    sanity_lines = [
        "",
        "Sanity checks",
        "-------------",
        f"Rows in input: {raw_rows}",
        f"Rows after dedup: {dedup_rows}",
        f"Rows after seed filter: {len(seed_rows)}",
        f"Rows after cleaning x/y: {len(cleaned)}",
        f"Statements with >= {args.min_participants} participants: {len(stats)}",
    ]
    summary_text = summary_text + "\n" + "\n".join(sanity_lines)

    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(summary_text, encoding="utf-8")

    print(summary_text)


if __name__ == "__main__":
    main()
