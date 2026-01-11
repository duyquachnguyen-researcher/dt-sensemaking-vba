"""Analyze whether raw statements reinforce or contradict seed priorities for Q20."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime
from math import sqrt
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q20 raw vs seed priority alignment")
    parser.add_argument(
        "--placements",
        type=Path,
        default=Path("Placement.csv"),
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
        default=Path("outputs/q20_raw_seed_alignment.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("outputs/q20_summary.txt"),
        help="Output summary text path",
    )
    parser.add_argument(
        "--min-placements",
        type=int,
        default=1,
        help="Minimum placements per statement to include",
    )
    parser.add_argument(
        "--midpoint",
        type=float,
        default=4.0,
        help="Midpoint threshold for quadrant splits",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top closest/farthest statements to list",
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
        cleaned.append({"token": token, "statement_id": statement_id, "x": x_val, "y": y_val})
    return cleaned


def load_statement_texts(path: Path, id_fields: list[str], text_fields: list[str]) -> dict[str, str]:
    rows = read_rows(path)
    mapping: dict[str, str] = {}
    for row in rows:
        statement_id = ""
        for field in id_fields:
            statement_id = (row.get(field) or "").strip()
            if statement_id:
                break
        if not statement_id:
            continue
        text_value = ""
        for field in text_fields:
            text_value = (row.get(field) or "").strip()
            if text_value:
                break
        mapping[statement_id] = text_value or statement_id
    return mapping


def statement_stats(rows: Iterable[dict[str, Any]], min_placements: int) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, list[Any]]] = defaultdict(lambda: {"x": [], "y": [], "tokens": []})
    for row in rows:
        statement_id = row["statement_id"]
        grouped[statement_id]["x"].append(row["x"])
        grouped[statement_id]["y"].append(row["y"])
        grouped[statement_id]["tokens"].append(row["token"])

    results: dict[str, dict[str, Any]] = {}
    for statement_id, data in grouped.items():
        n_placements = len(data["x"])
        if n_placements < min_placements:
            continue
        n_participants = len(set(data["tokens"]))
        sd_x = stdev(data["x"]) if len(data["x"]) > 1 else 0.0
        sd_y = stdev(data["y"]) if len(data["y"]) > 1 else 0.0
        results[statement_id] = {
            "statement_id": statement_id,
            "n_placements": n_placements,
            "n_participants": n_participants,
            "mean_x": mean(data["x"]),
            "mean_y": mean(data["y"]),
            "sd_x": sd_x,
            "sd_y": sd_y,
        }
    return results


def quadrant_label(x_val: float, y_val: float, midpoint: float) -> tuple[str, bool, bool]:
    x_high = x_val >= midpoint
    y_high = y_val >= midpoint
    if x_high and y_high:
        label = "Easy / High impact"
    elif x_high and not y_high:
        label = "Easy / Low impact"
    elif not x_high and y_high:
        label = "Hard / High impact"
    else:
        label = "Hard / Low impact"
    return label, x_high, y_high


def alignment_label(raw_x_high: bool, raw_y_high: bool, seed_x_high: bool, seed_y_high: bool) -> str:
    if raw_x_high == seed_x_high and raw_y_high == seed_y_high:
        return "reinforces"
    if raw_x_high != seed_x_high and raw_y_high != seed_y_high:
        return "contradicts"
    return "mixed"


def nearest_seed(raw_stat: dict[str, Any], seed_stats: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    closest: dict[str, Any] | None = None
    raw_x = raw_stat["mean_x"]
    raw_y = raw_stat["mean_y"]
    for seed in seed_stats.values():
        dx = raw_x - seed["mean_x"]
        dy = raw_y - seed["mean_y"]
        distance = sqrt(dx * dx + dy * dy)
        if closest is None or distance < closest["distance"]:
            closest = {
                "seed_id": seed["statement_id"],
                "seed_mean_x": seed["mean_x"],
                "seed_mean_y": seed["mean_y"],
                "seed_sd_x": seed["sd_x"],
                "seed_sd_y": seed["sd_y"],
                "distance": distance,
                "delta_x": dx,
                "delta_y": dy,
            }
    return closest


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_statement_row(row: dict[str, Any]) -> str:
    return (
        f"- {row['raw_text']} (raw {row['raw_id']}) → seed {row['seed_id']} "
        f"distance {row['distance']:.2f}, {row['alignment']}, "
        f"quadrants {row['raw_quadrant']} vs {row['seed_quadrant']}"
    )


def write_summary(
    path: Path,
    raw_stats: dict[str, dict[str, Any]],
    seed_stats: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
    top_n: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    distances = [row["distance"] for row in rows]
    alignment_counts = Counter(row["alignment"] for row in rows)
    closest = sorted(rows, key=lambda row: row["distance"])[:top_n]
    farthest = sorted(rows, key=lambda row: row["distance"], reverse=True)[:top_n]

    with path.open("w", encoding="utf-8") as handle:
        handle.write("Q20 Raw vs Seed Alignment Summary\n")
        handle.write("================================\n")
        handle.write(f"Raw statements with placements: {len(raw_stats)}\n")
        handle.write(f"Seed statements with placements: {len(seed_stats)}\n")
        handle.write(f"Raw statements mapped to a seed: {len(rows)}\n")
        if distances:
            handle.write(f"Mean distance to nearest seed: {mean(distances):.2f}\n")
            handle.write(f"Median distance to nearest seed: {median(distances):.2f}\n")
        handle.write("\nAlignment breakdown\n")
        handle.write("-------------------\n")
        for key in ("reinforces", "mixed", "contradicts"):
            handle.write(f"{key.title()}: {alignment_counts.get(key, 0)}\n")

        handle.write("\nClosest raw statements to seed priorities\n")
        handle.write("----------------------------------------\n")
        for row in closest:
            handle.write(f"{format_statement_row(row)}\n")

        handle.write("\nFarthest raw statements from seed priorities\n")
        handle.write("--------------------------------------------\n")
        for row in farthest:
            handle.write(f"{format_statement_row(row)}\n")


def main() -> None:
    args = parse_args()
    raw_texts = load_statement_texts(args.raw_statements, ["id"], ["text"])
    seed_texts = load_statement_texts(args.seed_statements, ["canonical_id", "id"], ["text_final", "subtitle"])
    raw_ids = set(raw_texts.keys())
    seed_ids = set(seed_texts.keys())

    placements = read_rows(args.placements)
    deduped = deduplicate_latest(placements)
    cleaned = cleaned_rows(deduped)

    raw_rows = [row for row in cleaned if row["statement_id"] in raw_ids]
    seed_rows = [row for row in cleaned if row["statement_id"] in seed_ids]

    raw_stats = statement_stats(raw_rows, args.min_placements)
    seed_stats = statement_stats(seed_rows, args.min_placements)

    results: list[dict[str, Any]] = []
    for raw_id, raw_stat in raw_stats.items():
        nearest = nearest_seed(raw_stat, seed_stats)
        if nearest is None:
            continue
        raw_quadrant, raw_x_high, raw_y_high = quadrant_label(raw_stat["mean_x"], raw_stat["mean_y"], args.midpoint)
        seed_quadrant, seed_x_high, seed_y_high = quadrant_label(
            nearest["seed_mean_x"], nearest["seed_mean_y"], args.midpoint
        )
        alignment = alignment_label(raw_x_high, raw_y_high, seed_x_high, seed_y_high)
        results.append(
            {
                "raw_id": raw_id,
                "raw_text": raw_texts.get(raw_id, raw_id),
                "raw_n_placements": raw_stat["n_placements"],
                "raw_n_participants": raw_stat["n_participants"],
                "raw_mean_x": round(raw_stat["mean_x"], 3),
                "raw_mean_y": round(raw_stat["mean_y"], 3),
                "raw_sd_x": round(raw_stat["sd_x"], 3),
                "raw_sd_y": round(raw_stat["sd_y"], 3),
                "raw_quadrant": raw_quadrant,
                "seed_id": nearest["seed_id"],
                "seed_text": seed_texts.get(nearest["seed_id"], nearest["seed_id"]),
                "seed_mean_x": round(nearest["seed_mean_x"], 3),
                "seed_mean_y": round(nearest["seed_mean_y"], 3),
                "seed_sd_x": round(nearest["seed_sd_x"], 3),
                "seed_sd_y": round(nearest["seed_sd_y"], 3),
                "seed_quadrant": seed_quadrant,
                "distance": round(nearest["distance"], 3),
                "delta_x": round(nearest["delta_x"], 3),
                "delta_y": round(nearest["delta_y"], 3),
                "alignment": alignment,
            }
        )

    results.sort(key=lambda row: row["distance"])

    fieldnames = [
        "raw_id",
        "raw_text",
        "raw_n_placements",
        "raw_n_participants",
        "raw_mean_x",
        "raw_mean_y",
        "raw_sd_x",
        "raw_sd_y",
        "raw_quadrant",
        "seed_id",
        "seed_text",
        "seed_mean_x",
        "seed_mean_y",
        "seed_sd_x",
        "seed_sd_y",
        "seed_quadrant",
        "distance",
        "delta_x",
        "delta_y",
        "alignment",
    ]
    write_csv(args.output_csv, results, fieldnames)
    write_summary(args.output_summary, raw_stats, seed_stats, results, args.top_n)


if __name__ == "__main__":
    main()
