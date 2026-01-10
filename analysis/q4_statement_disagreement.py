"""Analyze statement-level disagreement for Q4.

Computes controversy scores for seed statements based on the mean Euclidean
Distance to the statement centroid, along with dispersion metrics.
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
    parser = argparse.ArgumentParser(description="Q4 statement disagreement analysis")
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
        default=Path("outputs/q4_statement_disagreement.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("outputs/q4_summary.txt"),
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


def load_seed_statement_texts(path: Path) -> dict[str, str]:
    rows = read_rows(path)
    seed_texts: dict[str, str] = {}
    for row in rows:
        statement_id = (row.get("id") or "").strip()
        if not statement_id:
            continue
        text = (row.get("text_final") or row.get("subtitle") or "").strip()
        seed_texts[statement_id] = text
    return seed_texts


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


def disagreement_stats(rows: Iterable[dict[str, Any]], min_participants: int) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[Any]]] = defaultdict(lambda: {"x": [], "y": [], "tokens": []})
    for row in rows:
        statement_id = row["statement_id"]
        grouped[statement_id]["x"].append(row["x"])
        grouped[statement_id]["y"].append(row["y"])
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
        distances = [math.hypot(x_val - mean_x, y_val - mean_y) for x_val, y_val in zip(data["x"], data["y"])]
        mean_distance = mean(distances) if distances else 0.0
        max_distance = max(distances) if distances else 0.0

        results.append(
            {
                "statement_id": statement_id,
                "n_participants": n_participants,
                "mean_x": mean_x,
                "mean_y": mean_y,
                "sd_x": sd_x,
                "sd_y": sd_y,
                "mean_distance": mean_distance,
                "max_distance": max_distance,
            }
        )
    return results


def summarize(stats: list[dict[str, Any]], top_n: int, statement_texts: dict[str, str]) -> str:
    distance_vals = [row["mean_distance"] for row in stats]
    sd_x_vals = [row["sd_x"] for row in stats]
    sd_y_vals = [row["sd_y"] for row in stats]
    lines = [
        "Q4 Statement Disagreement Summary",
        "==================================",
        f"Statements included: {len(stats)}",
        f"Mean controversy score: {mean(distance_vals):.3f}" if distance_vals else "Mean controversy score: n/a",
        f"Median controversy score: {median(distance_vals):.3f}" if distance_vals else "Median controversy score: n/a",
        f"Mean SDx: {mean(sd_x_vals):.3f}" if sd_x_vals else "Mean SDx: n/a",
        f"Mean SDy: {mean(sd_y_vals):.3f}" if sd_y_vals else "Mean SDy: n/a",
        "",
        "Top statements by controversy score",
        "-----------------------------------",
    ]

    sorted_distance = sorted(
        stats,
        key=lambda row: (row["mean_distance"], row["n_participants"]),
        reverse=True,
    )
    for row in sorted_distance[: max(top_n, 0)]:
        statement_text = statement_texts.get(row["statement_id"], "Statement text unavailable")
        lines.append(
            "{statement_id}: {text} | {score:.2f} mean dist (sdx={sdx:.2f}, sdy={sdy:.2f}, n={n})".format(
                statement_id=row["statement_id"],
                text=statement_text,
                score=row["mean_distance"],
                sdx=row["sd_x"],
                sdy=row["sd_y"],
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
        "mean_distance",
        "max_distance",
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

    seed_texts = load_seed_statement_texts(args.statements)
    seed_ids = set(seed_texts)
    seed_rows = [row for row in deduped if (row.get("canonical_id") or "").strip() in seed_ids]

    cleaned = cleaned_rows(deduped, seed_ids)
    stats = disagreement_stats(cleaned, args.min_participants)

    write_csv(args.output_csv, stats)

    summary_text = summarize(stats, args.top_n, seed_texts)
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
