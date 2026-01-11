"""Analyze participant extremity and consistency for Q10.

Computes each participant's extremity rate (placements on grid edges) and
within-person dispersion (SD of X and Y) based on latest placements.
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
GRID_EDGES = {GRID_MIN, GRID_MAX}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q10 participant extremity analysis")
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
        "--output-csv",
        type=Path,
        default=Path("outputs/q10_participant_extremity.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("outputs/q10_summary.txt"),
        help="Output summary text path",
    )
    parser.add_argument(
        "--min-placements",
        type=int,
        default=5,
        help="Minimum placements required for summary rankings",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of participants to list in each ranking",
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


def load_participants(path: Path) -> dict[str, dict[str, str]]:
    participants: dict[str, dict[str, str]] = {}
    for row in read_rows(path):
        token = (row.get("token") or "").strip()
        if not token:
            continue
        participants[token] = {
            "role_level": (row.get("role_level") or "").strip(),
            "function": (row.get("function") or "").strip(),
            "country": (row.get("country") or "").strip(),
        }
    return participants


def cleaned_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        token = (row.get("token") or "").strip()
        if not token:
            continue
        x_val = coerce_float(row.get("x"))
        y_val = coerce_float(row.get("y"))
        if x_val is None or y_val is None:
            continue
        if not (GRID_MIN <= x_val <= GRID_MAX and GRID_MIN <= y_val <= GRID_MAX):
            continue
        cleaned.append({"token": token, "x": x_val, "y": y_val})
    return cleaned


def participant_stats(
    rows: Iterable[dict[str, Any]],
    participants: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"x": [], "y": []})
    for row in rows:
        token = row["token"]
        grouped[token]["x"].append(row["x"])
        grouped[token]["y"].append(row["y"])

    results: list[dict[str, Any]] = []
    for token, values in grouped.items():
        x_vals = values["x"]
        y_vals = values["y"]
        placements = len(x_vals)
        extremity_count = sum(1 for x_val, y_val in zip(x_vals, y_vals) if x_val in GRID_EDGES or y_val in GRID_EDGES)
        extremity_rate = extremity_count / placements if placements else 0.0
        sd_x = stdev(x_vals) if len(x_vals) > 1 else 0.0
        sd_y = stdev(y_vals) if len(y_vals) > 1 else 0.0
        profile = participants.get(token, {})
        results.append(
            {
                "token": token,
                "role_level": profile.get("role_level", ""),
                "function": profile.get("function", ""),
                "country": profile.get("country", ""),
                "placements": placements,
                "extremity_count": extremity_count,
                "extremity_rate": extremity_rate,
                "mean_x": mean(x_vals),
                "mean_y": mean(y_vals),
                "sd_x": sd_x,
                "sd_y": sd_y,
                "combined_sd": math.hypot(sd_x, sd_y),
            }
        )
    return results


def format_participant_label(row: dict[str, Any]) -> str:
    details = [row.get("role_level"), row.get("function"), row.get("country")]
    trimmed = [item for item in details if item]
    if trimmed:
        return f"{row['token']} ({' | '.join(trimmed)})"
    return row["token"]


def summarize(stats: list[dict[str, Any]], min_placements: int, top_n: int) -> str:
    eligible = [row for row in stats if row["placements"] >= min_placements]
    eligible_sorted = sorted(eligible, key=lambda row: row["extremity_rate"], reverse=True)
    consistent_sorted = sorted(eligible, key=lambda row: (row["combined_sd"], row["placements"]))
    variable_sorted = sorted(eligible, key=lambda row: (-row["combined_sd"], -row["placements"]))

    extremity_rates = [row["extremity_rate"] for row in eligible]
    sd_x_vals = [row["sd_x"] for row in eligible]
    sd_y_vals = [row["sd_y"] for row in eligible]

    lines = [
        "Q10 Participant Extremity & Consistency Summary",
        "================================================",
        f"Participants in data: {len(stats)}",
        f"Participants with >= {min_placements} placements: {len(eligible)}",
        f"Average extremity rate: {mean(extremity_rates):.2%}" if extremity_rates else "Average extremity rate: n/a",
        f"Median extremity rate: {median(extremity_rates):.2%}" if extremity_rates else "Median extremity rate: n/a",
        f"Average SD X: {mean(sd_x_vals):.2f}" if sd_x_vals else "Average SD X: n/a",
        f"Average SD Y: {mean(sd_y_vals):.2f}" if sd_y_vals else "Average SD Y: n/a",
        "",
        "Most extreme participants (highest edge-placement rate)",
        "------------------------------------------------------",
    ]

    for row in eligible_sorted[: max(top_n, 0)]:
        lines.append(
            "{label}: extremity_rate={rate:.2%} ({extreme}/{total}), sd_x={sd_x:.2f}, sd_y={sd_y:.2f}".format(
                label=format_participant_label(row),
                rate=row["extremity_rate"],
                extreme=row["extremity_count"],
                total=row["placements"],
                sd_x=row["sd_x"],
                sd_y=row["sd_y"],
            )
        )

    if not eligible_sorted:
        lines.append("No participants met the minimum placement threshold.")

    lines += [
        "",
        "Most consistent participants (lowest within-person dispersion)",
        "-------------------------------------------------------------",
    ]

    for row in consistent_sorted[: max(top_n, 0)]:
        lines.append(
            "{label}: combined_sd={combined_sd:.2f}, sd_x={sd_x:.2f}, sd_y={sd_y:.2f}, placements={total}".format(
                label=format_participant_label(row),
                combined_sd=row["combined_sd"],
                sd_x=row["sd_x"],
                sd_y=row["sd_y"],
                total=row["placements"],
            )
        )

    if not consistent_sorted:
        lines.append("No participants met the minimum placement threshold.")

    lines += [
        "",
        "Most variable participants (highest within-person dispersion)",
        "------------------------------------------------------------",
    ]

    for row in variable_sorted[: max(top_n, 0)]:
        lines.append(
            "{label}: combined_sd={combined_sd:.2f}, sd_x={sd_x:.2f}, sd_y={sd_y:.2f}, placements={total}".format(
                label=format_participant_label(row),
                combined_sd=row["combined_sd"],
                sd_x=row["sd_x"],
                sd_y=row["sd_y"],
                total=row["placements"],
            )
        )

    if not variable_sorted:
        lines.append("No participants met the minimum placement threshold.")

    return "\n".join(lines)


def write_csv(path: Path, stats: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "token",
        "role_level",
        "function",
        "country",
        "placements",
        "extremity_count",
        "extremity_rate",
        "mean_x",
        "mean_y",
        "sd_x",
        "sd_y",
        "combined_sd",
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
    participants = load_participants(args.participants)
    stats = participant_stats(cleaned, participants)

    write_csv(args.output_csv, stats)

    summary_text = summarize(stats, args.min_placements, args.top_n)
    sanity_lines = [
        "",
        "Sanity checks",
        "-------------",
        f"Rows in input: {raw_rows}",
        f"Rows after dedup: {dedup_rows}",
        f"Rows after cleaning x/y: {len(cleaned)}",
        f"Participants: {len(stats)}",
    ]
    summary_text = summary_text + "\n" + "\n".join(sanity_lines)

    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(summary_text, encoding="utf-8")

    print(summary_text)


if __name__ == "__main__":
    main()
