"""Analyze whether tenure predicts perceived implementability/impact for Q16."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from math import log, sqrt
from statistics import NormalDist, StatisticsError, mean, median, stdev
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
    parser = argparse.ArgumentParser(description="Q16 tenure vs. implementability/impact trends")
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
        default=Path("outputs/q16_tenure_metrics.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("outputs/q16_summary.txt"),
        help="Output summary text path",
    )
    parser.add_argument(
        "--extreme-values",
        type=float,
        nargs="+",
        default=[1.0, 7.0],
        help="Values that count as extreme placements",
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


def load_tenure(path: Path) -> dict[str, float]:
    rows = read_rows(path)
    token_to_tenure: dict[str, float] = {}
    for row in rows:
        token = (row.get("token") or "").strip()
        if not token:
            continue
        tenure = coerce_float(row.get("tenure_years"))
        if tenure is None:
            continue
        token_to_tenure[token] = tenure
    return token_to_tenure


def participant_metrics(rows: Iterable[dict[str, Any]], extreme_values: set[float]) -> list[dict[str, Any]]:
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
        n_placements = len(x_vals)
        extremity_count = sum(
            1 for x_val, y_val in zip(x_vals, y_vals) if x_val in extreme_values or y_val in extreme_values
        )
        results.append(
            {
                "token": token,
                "n_placements": n_placements,
                "mean_x": mean(x_vals),
                "mean_y": mean(y_vals),
                "extremity_count": extremity_count,
                "extremity_rate": extremity_count / n_placements if n_placements else 0.0,
            }
        )
    return results


def pearson_corr(x_vals: list[float], y_vals: list[float]) -> float | None:
    if len(x_vals) < 2 or len(y_vals) < 2 or len(x_vals) != len(y_vals):
        return None
    try:
        std_x = stdev(x_vals)
        std_y = stdev(y_vals)
    except StatisticsError:
        return None
    if std_x == 0 or std_y == 0:
        return None
    mean_x = mean(x_vals)
    mean_y = mean(y_vals)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_vals, y_vals)) / (len(x_vals) - 1)
    return cov / (std_x * std_y)


def corr_p_value(r_value: float | None, n_samples: int) -> float | None:
    if r_value is None or n_samples < 4:
        return None
    if abs(r_value) >= 1:
        return 0.0
    z_value = 0.5 * log((1 + r_value) / (1 - r_value)) * sqrt(n_samples - 3)
    return 2 * (1 - NormalDist().cdf(abs(z_value)))


def format_corr(label: str, value: float | None, p_value: float | None) -> str:
    if value is None:
        return f"{label}: n/a (insufficient variance)"
    if p_value is None:
        return f"{label}: r = {value:.2f}, p = n/a"
    return f"{label}: r = {value:.2f}, p = {p_value:.3f}"


def summarize(rows: list[dict[str, Any]]) -> str:
    lines = [
        "Q16 Tenure vs Implementability/Impact Summary",
        "==============================================",
    ]
    if not rows:
        lines.append("No participants with both tenure and placement data.")
        return "\n".join(lines)

    tenure_vals = [row["tenure_years"] for row in rows]
    mean_x_vals = [row["mean_x"] for row in rows]
    mean_y_vals = [row["mean_y"] for row in rows]
    extremity_rates = [row["extremity_rate"] for row in rows]
    n_placements_vals = [row["n_placements"] for row in rows]

    lines.extend(
        [
            f"Participants included: {len(rows)}",
            f"Average tenure (years): {mean(tenure_vals):.2f}",
            f"Median tenure (years): {median(tenure_vals):.2f}",
            f"Average placements per participant: {mean(n_placements_vals):.2f}",
            "",
            "Average participant tendencies",
            "-------------------------------",
            f"Mean implementability (x): {mean(mean_x_vals):.2f}",
            f"Mean impact (y): {mean(mean_y_vals):.2f}",
            f"Average extremity rate: {mean(extremity_rates):.2%}",
            "",
            "Tenure correlations (Pearson r)",
            "--------------------------------",
            format_corr(
                "Tenure vs mean implementability (x)",
                pearson_corr(tenure_vals, mean_x_vals),
                corr_p_value(pearson_corr(tenure_vals, mean_x_vals), len(rows)),
            ),
            format_corr(
                "Tenure vs mean impact (y)",
                pearson_corr(tenure_vals, mean_y_vals),
                corr_p_value(pearson_corr(tenure_vals, mean_y_vals), len(rows)),
            ),
            format_corr(
                "Tenure vs extremity rate",
                pearson_corr(tenure_vals, extremity_rates),
                corr_p_value(pearson_corr(tenure_vals, extremity_rates), len(rows)),
            ),
            "P-values use a Fisher z approximation (two-tailed).",
            "",
            "Interpretation",
            "--------------",
            "Positive r indicates higher tenure aligns with higher values; negative r indicates the opposite.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    placement_rows = read_rows(args.input)
    placement_rows = deduplicate_latest(placement_rows)
    cleaned = cleaned_rows(placement_rows)
    metrics = participant_metrics(cleaned, set(args.extreme_values))

    tenure_map = load_tenure(args.participants)
    combined: list[dict[str, Any]] = []
    for row in metrics:
        tenure = tenure_map.get(row["token"])
        if tenure is None:
            continue
        combined.append({**row, "tenure_years": tenure})

    combined.sort(key=lambda item: (-item["tenure_years"], item["token"]))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "token",
            "tenure_years",
            "n_placements",
            "mean_x",
            "mean_y",
            "extremity_count",
            "extremity_rate",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in combined:
            writer.writerow(
                {
                    "token": row["token"],
                    "tenure_years": f"{row['tenure_years']:.2f}",
                    "n_placements": row["n_placements"],
                    "mean_x": f"{row['mean_x']:.3f}",
                    "mean_y": f"{row['mean_y']:.3f}",
                    "extremity_count": row["extremity_count"],
                    "extremity_rate": f"{row['extremity_rate']:.3f}",
                }
            )

    summary_text = summarize(combined)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(summary_text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
