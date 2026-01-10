"""Analyze convergence on impact vs implementability for Q1.

Reads Placement.csv, deduplicates by latest timestamp per participant/statement,
computes per-statement dispersion, and summarizes whether impact (Y) converges
more than implementability (X).
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q1 convergence analysis")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Placement.csv"),
        help="Path to Placement.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/q1_statement_dispersion.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("outputs/q1_summary.txt"),
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


def statement_stats(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"x": [], "y": [], "tokens": []})
    for row in rows:
        statement_id = row["statement_id"]
        grouped[statement_id]["x"].append(row["x"])
        grouped[statement_id]["y"].append(row["y"])
        grouped[statement_id]["tokens"].append(row["token"])

    results: list[dict[str, Any]] = []
    for statement_id, data in grouped.items():
        tokens = set(data["tokens"])
        n_participants = len(tokens)
        if n_participants < 2:
            continue
        sd_x = stdev(data["x"]) if len(data["x"]) > 1 else float("nan")
        sd_y = stdev(data["y"]) if len(data["y"]) > 1 else float("nan")
        sd_diff = sd_y - sd_x
        if math.isnan(sd_x) or math.isnan(sd_y):
            interpretation = "Insufficient data"
        elif sd_y < sd_x:
            interpretation = "Impact more consistent"
        elif sd_y > sd_x:
            interpretation = "Implementability more consistent"
        else:
            interpretation = "Equal"

        results.append(
            {
                "statement_id": statement_id,
                "n_participants": n_participants,
                "mean_x": mean(data["x"]),
                "mean_y": mean(data["y"]),
                "sd_x": sd_x,
                "sd_y": sd_y,
                "sd_diff": sd_diff,
                "interpretation": interpretation,
            }
        )
    return results


def compute_summary(stats: list[dict[str, Any]]) -> dict[str, float | int]:
    sd_x_vals = [row["sd_x"] for row in stats]
    sd_y_vals = [row["sd_y"] for row in stats]
    sd_diff_vals = [row["sd_diff"] for row in stats]
    return {
        "mean_sd_x": mean(sd_x_vals) if sd_x_vals else float("nan"),
        "mean_sd_y": mean(sd_y_vals) if sd_y_vals else float("nan"),
        "median_sd_x": median(sd_x_vals) if sd_x_vals else float("nan"),
        "median_sd_y": median(sd_y_vals) if sd_y_vals else float("nan"),
        "mean_sd_diff": mean(sd_diff_vals) if sd_diff_vals else float("nan"),
        "statement_count": len(stats),
    }


def paired_t_test(sd_x: list[float], sd_y: list[float]) -> tuple[float, str]:
    if len(sd_x) != len(sd_y) or len(sd_x) < 2:
        return float("nan"), "insufficient data"
    diffs = [y_val - x_val for x_val, y_val in zip(sd_x, sd_y)]
    mean_diff = mean(diffs)
    sd_diff = stdev(diffs)
    if sd_diff == 0:
        return 0.0, "zero variance"
    t_stat = mean_diff / (sd_diff / math.sqrt(len(diffs)))
    if t_stat == 0:
        return 1.0, "two-tailed normal approximation"
    try:
        import importlib.util

        if importlib.util.find_spec("scipy") is not None:
            from scipy import stats  # noqa: WPS433 (optional dependency)

            p_value = float(stats.ttest_rel(sd_y, sd_x, nan_policy="omit").pvalue)
            return p_value, "paired t-test (scipy)"
    except Exception:
        pass
    # Normal approximation when scipy is unavailable.
    z = abs(t_stat)
    p_value = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
    return p_value, "two-tailed normal approximation"


def format_summary(summary: dict[str, float | int], p_value: float, p_method: str) -> str:
    lines = [
        "Q1 Convergence Summary",
        "========================",
        f"Statements included: {summary['statement_count']}",
        f"Mean SDx (Implementability): {summary['mean_sd_x']:.3f}",
        f"Mean SDy (Impact): {summary['mean_sd_y']:.3f}",
        f"Median SDx (Implementability): {summary['median_sd_x']:.3f}",
        f"Median SDy (Impact): {summary['median_sd_y']:.3f}",
        f"Mean(SDy - SDx): {summary['mean_sd_diff']:.3f}",
    ]
    lines.append(f"Paired t-test p-value (SDy vs SDx): {p_value:.4g}")
    lines.append(f"P-value method: {p_method}")
    return "\n".join(lines)


def write_statement_csv(path: Path, stats: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "statement_id",
        "n_participants",
        "mean_x",
        "mean_y",
        "sd_x",
        "sd_y",
        "sd_diff",
        "interpretation",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in stats:
            writer.writerow(row)


def write_summary(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    raw_rows = len(rows)

    deduped = deduplicate_latest(rows)
    dedup_rows = len(deduped)

    cleaned = cleaned_rows(deduped)
    stats = statement_stats(cleaned)
    summary = compute_summary(stats)
    p_value, p_method = paired_t_test(
        [row["sd_x"] for row in stats],
        [row["sd_y"] for row in stats],
    )

    participants = {row["token"] for row in cleaned}
    statements_with_data = {row["statement_id"] for row in cleaned}

    summary_text = format_summary(summary, p_value, p_method)
    sanity_lines = [
        "",
        "Sanity checks",
        "-------------",
        f"Rows in input: {raw_rows}",
        f"Rows after dedup: {dedup_rows}",
        f"Rows after cleaning x/y: {len(cleaned)}",
        f"Participants (unique tokens): {len(participants)}",
        f"Statements with any data: {len(statements_with_data)}",
        f"Statements with >=2 participants: {summary['statement_count']}",
    ]
    summary_text = summary_text + "\n" + "\n".join(sanity_lines)

    write_statement_csv(args.output_csv, stats)
    write_summary(args.output_summary, summary_text)

    print(summary_text)


if __name__ == "__main__":
    main()
