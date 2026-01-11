"""Analyze participant mapping tendencies for Q9.

Computes per-participant placement averages, variance, and a high-high tendency
metric (share of placements in the high-impact/high-implementability quadrant).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, pvariance
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
    parser = argparse.ArgumentParser(description="Q9 participant mapping tendencies")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Placement.csv"),
        help="Path to Placement.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/q9_participant_tendencies.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("outputs/q9_summary.txt"),
        help="Output summary text path",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of participants to list per section",
    )
    parser.add_argument(
        "--high-threshold",
        type=float,
        default=5.0,
        help="Threshold for high-high tendency (x>=threshold and y>=threshold)",
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
        cleaned.append({"token": token, "x": x_val, "y": y_val})
    return cleaned


def participant_stats(rows: Iterable[dict[str, Any]], high_threshold: float) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"x": [], "y": []})
    for row in rows:
        token = row["token"]
        grouped[token]["x"].append(row["x"])
        grouped[token]["y"].append(row["y"])

    results: list[dict[str, Any]] = []
    for token, data in grouped.items():
        x_vals = data["x"]
        y_vals = data["y"]
        if not x_vals or not y_vals:
            continue
        high_high_count = sum(1 for x_val, y_val in zip(x_vals, y_vals) if x_val >= high_threshold and y_val >= high_threshold)
        n_placements = len(x_vals)
        results.append(
            {
                "token": token,
                "n_placements": n_placements,
                "mean_x": mean(x_vals),
                "mean_y": mean(y_vals),
                "var_x": pvariance(x_vals) if n_placements > 1 else 0.0,
                "var_y": pvariance(y_vals) if n_placements > 1 else 0.0,
                "high_high_count": high_high_count,
                "high_high_rate": high_high_count / n_placements if n_placements else 0.0,
            }
        )
    return results


def summarize(stats: list[dict[str, Any]], top_n: int, high_threshold: float) -> str:
    lines = [
        "Q9 Participant Mapping Styles Summary",
        "======================================",
        f"Participants included: {len(stats)}",
        f"High-high metric: share of placements with x>= {high_threshold:.1f} and y>= {high_threshold:.1f}",
        "",
    ]

    if stats:
        mean_x_vals = [row["mean_x"] for row in stats]
        mean_y_vals = [row["mean_y"] for row in stats]
        var_x_vals = [row["var_x"] for row in stats]
        var_y_vals = [row["var_y"] for row in stats]
        high_high_rates = [row["high_high_rate"] for row in stats]
        lines.extend(
            [
                "Overall participant averages",
                "-----------------------------",
                f"Average mean_x: {mean(mean_x_vals):.2f}",
                f"Average mean_y: {mean(mean_y_vals):.2f}",
                f"Average var_x: {mean(var_x_vals):.2f}",
                f"Average var_y: {mean(var_y_vals):.2f}",
                f"Average high-high rate: {mean(high_high_rates):.2%}",
                "",
            ]
        )

        top_optimists = sorted(
            stats,
            key=lambda row: (-row["high_high_rate"], -row["n_placements"], row["token"]),
        )
        lines.extend(
            [
                "Top high-high tendency participants",
                "-----------------------------------",
            ]
        )
        for row in top_optimists[: max(top_n, 0)]:
            lines.append(
                "{token}: high_high_rate={rate:.2%} ({count}/{total}), mean_x={mean_x:.2f}, mean_y={mean_y:.2f}".format(
                    token=row["token"],
                    rate=row["high_high_rate"],
                    count=row["high_high_count"],
                    total=row["n_placements"],
                    mean_x=row["mean_x"],
                    mean_y=row["mean_y"],
                )
            )

        variability_ranked = sorted(
            stats,
            key=lambda row: (-(row["var_x"] + row["var_y"]), -row["n_placements"], row["token"]),
        )
        lines.extend(
            [
                "",
                "Most variable participants (var_x + var_y)",
                "------------------------------------------",
            ]
        )
        for row in variability_ranked[: max(top_n, 0)]:
            total_var = row["var_x"] + row["var_y"]
            lines.append(
                "{token}: total_var={total_var:.2f}, var_x={var_x:.2f}, var_y={var_y:.2f}, placements={n}".format(
                    token=row["token"],
                    total_var=total_var,
                    var_x=row["var_x"],
                    var_y=row["var_y"],
                    n=row["n_placements"],
                )
            )
    else:
        lines.append("No participant placements available after filtering.")

    return "\n".join(lines)


def write_csv(path: Path, stats: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "token",
        "n_placements",
        "mean_x",
        "mean_y",
        "var_x",
        "var_y",
        "high_high_count",
        "high_high_rate",
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

    cleaned = cleaned_rows(deduped)
    stats = participant_stats(cleaned, args.high_threshold)

    write_csv(args.output_csv, stats)

    summary_text = summarize(stats, args.top_n, args.high_threshold)
    participants = {row["token"] for row in cleaned}

    sanity_lines = [
        "",
        "Sanity checks",
        "-------------",
        f"Rows in input: {raw_rows}",
        f"Rows after dedup: {dedup_rows}",
        f"Rows after filtering: {len(cleaned)}",
        f"Participants (unique tokens): {len(participants)}",
    ]
    summary_text = summary_text + "\n" + "\n".join(sanity_lines)

    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(summary_text, encoding="utf-8")

    print(summary_text)


if __name__ == "__main__":
    main()
