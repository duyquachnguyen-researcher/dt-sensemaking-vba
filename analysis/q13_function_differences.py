"""Analyze placement differences by participant function for Q13."""

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
    parser = argparse.ArgumentParser(description="Q13 placement differences by participant function")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Placement.csv"),
        help="Path to Placement.csv",
    )
    parser.add_argument(
        "--participants",
        type=Path,
        default=Path("Participant.csv"),
        help="Path to Participant.csv",
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
        default=Path("outputs/q13_function_differences.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("outputs/q13_summary.txt"),
        help="Output summary text path",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=2,
        help="Minimum placements per function to include in tests",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of function pairs to list",
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


def load_seed_statement_ids(path: Path) -> set[str]:
    rows = read_rows(path)
    return {(row.get("id") or "").strip() for row in rows if (row.get("id") or "").strip()}


def load_functions(path: Path) -> dict[str, str]:
    rows = read_rows(path)
    token_to_function: dict[str, str] = {}
    for row in rows:
        token = (row.get("token") or "").strip()
        if not token:
            continue
        function = (row.get("function") or "").strip()
        if not function:
            continue
        token_to_function[token] = function
    return token_to_function


def cleaned_rows(
    rows: Iterable[dict[str, str]],
    seed_ids: set[str],
    token_to_function: dict[str, str],
) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        token = (row.get("token") or "").strip()
        statement_id = (row.get("canonical_id") or "").strip()
        if not token or not statement_id:
            continue
        if statement_id not in seed_ids:
            continue
        function = token_to_function.get(token)
        if not function:
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
                "function": function,
                "x": x_val,
                "y": y_val,
            }
        )
    return cleaned


def rank_data(values: list[float]) -> tuple[list[float], list[int]]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    tie_sizes: list[int] = []
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        tie_sizes.append(j - i + 1)
        i = j + 1
    return ranks, tie_sizes


def gammainc_series(a: float, x: float) -> float:
    gln = math.lgamma(a)
    ap = a
    summation = 1.0 / a
    delta = summation
    for _ in range(1, 200):
        ap += 1.0
        delta *= x / ap
        summation += delta
        if abs(delta) < abs(summation) * 3e-7:
            break
    return summation * math.exp(-x + a * math.log(x) - gln)


def gammainc_cf(a: float, x: float) -> float:
    gln = math.lgamma(a)
    b = x + 1.0 - a
    c = 1.0 / 1.0e-30
    d = 1.0 / b
    h = d
    for i in range(1, 200):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1.0e-30:
            d = 1.0e-30
        c = b + an / c
        if abs(c) < 1.0e-30:
            c = 1.0e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-7:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def gammaincc(a: float, x: float) -> float:
    if x < 0 or a <= 0:
        return float("nan")
    if x == 0:
        return 1.0
    if x < a + 1.0:
        return 1.0 - gammainc_series(a, x)
    return gammainc_cf(a, x)


def kruskal_wallis(groups: list[list[float]]) -> float | None:
    if len(groups) < 2:
        return None
    values: list[float] = []
    group_sizes: list[int] = []
    for group in groups:
        if not group:
            return None
        values.extend(group)
        group_sizes.append(len(group))
    ranks, tie_sizes = rank_data(values)
    n = len(values)
    if n == 0:
        return None
    rank_index = 0
    rank_sums: list[float] = []
    for size in group_sizes:
        rank_sums.append(sum(ranks[rank_index : rank_index + size]))
        rank_index += size
    h = 0.0
    for rank_sum, size in zip(rank_sums, group_sizes):
        h += (rank_sum**2) / size
    h = (12.0 / (n * (n + 1))) * h - 3 * (n + 1)
    tie_sum = sum(size**3 - size for size in tie_sizes)
    denom = n**3 - n
    if denom <= 0:
        return None
    tie_correction = 1.0 - tie_sum / denom
    if tie_correction <= 0:
        return None
    h /= tie_correction
    df = len(groups) - 1
    p_value = gammaincc(df / 2.0, h / 2.0)
    return min(max(p_value, 0.0), 1.0)


def compute_function_stats(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, list[Any]]] = defaultdict(lambda: {"x": [], "y": [], "tokens": []})
    for row in rows:
        grouped[row["function"]]["x"].append(row["x"])
        grouped[row["function"]]["y"].append(row["y"])
        grouped[row["function"]]["tokens"].append(row["token"])

    stats: dict[str, dict[str, Any]] = {}
    for function, data in grouped.items():
        tokens = set(data["tokens"])
        x_vals = data["x"]
        y_vals = data["y"]
        if not x_vals or not y_vals:
            continue
        stats[function] = {
            "n_participants": len(tokens),
            "n_placements": len(x_vals),
            "mean_x": mean(x_vals),
            "mean_y": mean(y_vals),
            "median_x": median(x_vals),
            "median_y": median(y_vals),
            "sd_x": stdev(x_vals) if len(x_vals) > 1 else 0.0,
            "sd_y": stdev(y_vals) if len(y_vals) > 1 else 0.0,
            "x_values": x_vals,
            "y_values": y_vals,
        }
    return stats


def compute_pair_distances(stats: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    functions = sorted(stats.keys())
    pairs: list[dict[str, Any]] = []
    for i, func_a in enumerate(functions):
        for func_b in functions[i + 1 :]:
            a = stats[func_a]
            b = stats[func_b]
            distance = math.hypot(a["mean_x"] - b["mean_x"], a["mean_y"] - b["mean_y"])
            pairs.append(
                {
                    "function_a": func_a,
                    "function_b": func_b,
                    "distance": distance,
                    "delta_x": a["mean_x"] - b["mean_x"],
                    "delta_y": a["mean_y"] - b["mean_y"],
                }
            )
    return pairs


def summarize(
    stats: dict[str, dict[str, Any]],
    pair_distances: list[dict[str, Any]],
    p_value_x: float | None,
    p_value_y: float | None,
    total_participants: int,
    total_placements: int,
    top_n: int,
) -> str:
    lines = [
        "Q13 Function Differences Summary",
        "===============================",
        f"Functions included: {len(stats)}",
        f"Participants included: {total_participants}",
        f"Placements included: {total_placements}",
        "",
        "Overall tests across participant functions",
        "------------------------------------------",
        f"Kruskal-Wallis p-value (X): {p_value_x:.4f}" if p_value_x is not None else "Kruskal-Wallis p-value (X): n/a",
        f"Kruskal-Wallis p-value (Y): {p_value_y:.4f}" if p_value_y is not None else "Kruskal-Wallis p-value (Y): n/a",
        "",
    ]

    if pair_distances:
        lines.extend(
            [
                "Top function pairs by centroid distance",
                "---------------------------------------",
            ]
        )
        for pair in sorted(pair_distances, key=lambda item: (-item["distance"], item["function_a"]))[:top_n]:
            lines.append(
                f"- {pair['function_a']} vs {pair['function_b']}: "
                f"distance {pair['distance']:.2f} (Δx {pair['delta_x']:.2f}, Δy {pair['delta_y']:.2f})"
            )
    else:
        lines.append("Not enough functions to compute pairwise distances.")

    return "\n".join(lines)


def write_csv(stats: dict[str, dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "function",
        "n_participants",
        "n_placements",
        "mean_x",
        "mean_y",
        "median_x",
        "median_y",
        "sd_x",
        "sd_y",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for function, data in sorted(stats.items(), key=lambda item: (-item[1]["n_participants"], item[0])):
            writer.writerow(
                {
                    "function": function,
                    "n_participants": data["n_participants"],
                    "n_placements": data["n_placements"],
                    "mean_x": data["mean_x"],
                    "mean_y": data["mean_y"],
                    "median_x": data["median_x"],
                    "median_y": data["median_y"],
                    "sd_x": data["sd_x"],
                    "sd_y": data["sd_y"],
                }
            )


def main() -> None:
    args = parse_args()
    seed_ids = load_seed_statement_ids(args.statements)
    token_to_function = load_functions(args.participants)
    placements = read_rows(args.input)
    deduped = deduplicate_latest(placements)
    cleaned = cleaned_rows(deduped, seed_ids, token_to_function)
    function_stats = compute_function_stats(cleaned)

    total_participants = len({row["token"] for row in cleaned})
    total_placements = len(cleaned)

    x_groups = [
        stats["x_values"]
        for stats in function_stats.values()
        if len(stats["x_values"]) >= args.min_group_size
    ]
    y_groups = [
        stats["y_values"]
        for stats in function_stats.values()
        if len(stats["y_values"]) >= args.min_group_size
    ]
    p_value_x = kruskal_wallis(x_groups) if len(x_groups) >= 2 else None
    p_value_y = kruskal_wallis(y_groups) if len(y_groups) >= 2 else None

    pair_distances = compute_pair_distances(function_stats)

    write_csv(function_stats, args.output_csv)
    summary_text = summarize(
        function_stats,
        pair_distances,
        p_value_x,
        p_value_y,
        total_participants,
        total_placements,
        args.top_n,
    )
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(summary_text, encoding="utf-8")


if __name__ == "__main__":
    main()
