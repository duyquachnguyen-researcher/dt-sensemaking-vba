"""Analyze whether hybrid work percentage relates to placement patterns for Q17."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from math import log, sqrt
from pathlib import Path
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
    parser = argparse.ArgumentParser(description="Q17 hybrid% vs placement pattern analysis")
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
        default=Path("outputs/q17_hybrid_metrics.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("outputs/q17_summary.txt"),
        help="Output summary text path",
    )
    parser.add_argument(
        "--extreme-values",
        type=float,
        nargs="+",
        default=[1.0, 7.0],
        help="Values that count as extreme placements",
    )
    parser.add_argument(
        "--min-statement-n",
        type=int,
        default=4,
        help="Minimum placements per statement to compute correlations",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of statements to list in sensitivity summaries",
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


def load_hybrid_pct(path: Path) -> dict[str, float]:
    rows = read_rows(path)
    token_to_hybrid: dict[str, float] = {}
    for row in rows:
        token = (row.get("token") or "").strip()
        if not token:
            continue
        hybrid_pct = coerce_float(row.get("hybrid_pct"))
        if hybrid_pct is None:
            continue
        token_to_hybrid[token] = hybrid_pct
    return token_to_hybrid


def load_statement_texts(path: Path) -> dict[str, str]:
    rows = read_rows(path)
    mapping: dict[str, str] = {}
    for row in rows:
        statement_id = (row.get("id") or row.get("canonical_id") or "").strip()
        if not statement_id:
            continue
        text = (row.get("text_final") or row.get("subtitle") or "").strip()
        mapping[statement_id] = text or statement_id
    return mapping


def cleaned_rows(rows: Iterable[dict[str, str]], hybrid_map: dict[str, float]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        token = (row.get("token") or "").strip()
        statement_id = (row.get("canonical_id") or "").strip()
        if not token or not statement_id:
            continue
        hybrid_pct = hybrid_map.get(token)
        if hybrid_pct is None:
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
                "statement_id": statement_id,
                "hybrid_pct": hybrid_pct,
                "x": x_val,
                "y": y_val,
            }
        )
    return cleaned


def participant_metrics(rows: Iterable[dict[str, Any]], extreme_values: set[float]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"x": [], "y": [], "hybrid": None})
    for row in rows:
        token = row["token"]
        grouped[token]["x"].append(row["x"])
        grouped[token]["y"].append(row["y"])
        grouped[token]["hybrid"] = row["hybrid_pct"]

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
                "hybrid_pct": data["hybrid"] if data["hybrid"] is not None else 0.0,
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
        return f"{label}: insufficient data"
    if p_value is None:
        return f"{label}: r = {value:.2f} (p unavailable)"
    return f"{label}: r = {value:.2f}, p = {p_value:.3f}"


def statement_sensitivity(rows: Iterable[dict[str, Any]], min_n: int) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"hybrid": [], "x": [], "y": []})
    for row in rows:
        statement_id = row["statement_id"]
        grouped[statement_id]["hybrid"].append(row["hybrid_pct"])
        grouped[statement_id]["x"].append(row["x"])
        grouped[statement_id]["y"].append(row["y"])

    results: list[dict[str, Any]] = []
    for statement_id, values in grouped.items():
        n_samples = len(values["hybrid"])
        if n_samples < min_n:
            continue
        r_x = pearson_corr(values["hybrid"], values["x"])
        r_y = pearson_corr(values["hybrid"], values["y"])
        results.append(
            {
                "statement_id": statement_id,
                "n": n_samples,
                "r_x": r_x,
                "p_x": corr_p_value(r_x, n_samples) if r_x is not None else None,
                "r_y": r_y,
                "p_y": corr_p_value(r_y, n_samples) if r_y is not None else None,
                "mean_x": mean(values["x"]),
                "mean_y": mean(values["y"]),
            }
        )
    return results


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    fieldnames = [
        "token",
        "hybrid_pct",
        "n_placements",
        "mean_x",
        "mean_y",
        "extremity_count",
        "extremity_rate",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "token": row["token"],
                    "hybrid_pct": f"{row['hybrid_pct']:.2f}",
                    "n_placements": row["n_placements"],
                    "mean_x": f"{row['mean_x']:.3f}",
                    "mean_y": f"{row['mean_y']:.3f}",
                    "extremity_count": row["extremity_count"],
                    "extremity_rate": f"{row['extremity_rate']:.3f}",
                }
            )


def format_statement_line(entry: dict[str, Any], label: str, text: str) -> str:
    r_value = entry[label]
    p_value = entry["p_x"] if label == "r_x" else entry["p_y"]
    if r_value is None:
        return f"- {text} (n={entry['n']}): insufficient data"
    p_text = "p unavailable" if p_value is None else f"p = {p_value:.3f}"
    return f"- {text} (n={entry['n']}, r = {r_value:.2f}, {p_text})"


def write_summary(
    path: Path,
    participant_rows: list[dict[str, Any]],
    participant_corrs: dict[str, tuple[float | None, float | None]],
    statement_rows: list[dict[str, Any]],
    statement_texts: dict[str, str],
    top_n: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hybrids = [row["hybrid_pct"] for row in participant_rows]
    mean_x_vals = [row["mean_x"] for row in participant_rows]
    mean_y_vals = [row["mean_y"] for row in participant_rows]
    extremity_rates = [row["extremity_rate"] for row in participant_rows]

    statement_rows = [row for row in statement_rows if row["r_x"] is not None or row["r_y"] is not None]
    top_x = sorted(statement_rows, key=lambda r: abs(r["r_x"]) if r["r_x"] is not None else 0, reverse=True)[:top_n]
    top_y = sorted(statement_rows, key=lambda r: abs(r["r_y"]) if r["r_y"] is not None else 0, reverse=True)[:top_n]

    with path.open("w", encoding="utf-8") as handle:
        handle.write("Q17 Hybrid% vs Placement Patterns Summary\n")
        handle.write("========================================\n")
        handle.write(f"Participants included: {len(participant_rows)}\n")
        if hybrids:
            handle.write(f"Average hybrid%: {mean(hybrids):.2f}\n")
            handle.write(f"Median hybrid%: {median(hybrids):.2f}\n")
        if participant_rows:
            handle.write(f"Average placements per participant: {mean(row['n_placements'] for row in participant_rows):.2f}\n")
        handle.write("\n")

        handle.write("Average participant tendencies\n")
        handle.write("-------------------------------\n")
        if mean_x_vals:
            handle.write(f"Mean implementability (x): {mean(mean_x_vals):.2f}\n")
            handle.write(f"Mean impact (y): {mean(mean_y_vals):.2f}\n")
            handle.write(f"Average extremity rate: {mean(extremity_rates) * 100:.2f}%\n")
        else:
            handle.write("Insufficient participant placements.\n")
        handle.write("\n")

        handle.write("Hybrid% correlations (Pearson r)\n")
        handle.write("--------------------------------\n")
        handle.write(format_corr("Hybrid% vs mean implementability (x)", *participant_corrs["mean_x"]) + "\n")
        handle.write(format_corr("Hybrid% vs mean impact (y)", *participant_corrs["mean_y"]) + "\n")
        handle.write(format_corr("Hybrid% vs extremity rate", *participant_corrs["extremity_rate"]) + "\n")
        handle.write("P-values use a Fisher z approximation (two-tailed).\n")
        handle.write("\n")

        handle.write("Statement sensitivity (hybrid% vs placements)\n")
        handle.write("---------------------------------------------\n")
        if not statement_rows:
            handle.write("No statements met the minimum sample threshold.\n")
        else:
            handle.write("Top statements where hybrid% aligns with implementability (x):\n")
            for entry in top_x:
                text = statement_texts.get(entry["statement_id"], entry["statement_id"])
                handle.write(format_statement_line(entry, "r_x", text) + "\n")
            handle.write("\n")
            handle.write("Top statements where hybrid% aligns with impact (y):\n")
            for entry in top_y:
                text = statement_texts.get(entry["statement_id"], entry["statement_id"])
                handle.write(format_statement_line(entry, "r_y", text) + "\n")

        handle.write("\nInterpretation\n")
        handle.write("--------------\n")
        handle.write("Positive r indicates higher hybrid% aligns with higher values; negative r indicates the opposite.\n")


def main() -> None:
    args = parse_args()
    placements = read_rows(args.input)
    deduped = deduplicate_latest(placements)
    hybrid_map = load_hybrid_pct(args.participants)
    statement_texts = load_statement_texts(args.statements)
    cleaned = cleaned_rows(deduped, hybrid_map)

    metrics = participant_metrics(cleaned, set(args.extreme_values))
    metrics_sorted = sorted(metrics, key=lambda row: row["hybrid_pct"], reverse=True)
    write_csv(args.output_csv, metrics_sorted)

    hybrid_vals = [row["hybrid_pct"] for row in metrics]
    mean_x_vals = [row["mean_x"] for row in metrics]
    mean_y_vals = [row["mean_y"] for row in metrics]
    extremity_vals = [row["extremity_rate"] for row in metrics]

    participant_corrs = {
        "mean_x": (
            pearson_corr(hybrid_vals, mean_x_vals),
            corr_p_value(pearson_corr(hybrid_vals, mean_x_vals), len(hybrid_vals)),
        ),
        "mean_y": (
            pearson_corr(hybrid_vals, mean_y_vals),
            corr_p_value(pearson_corr(hybrid_vals, mean_y_vals), len(hybrid_vals)),
        ),
        "extremity_rate": (
            pearson_corr(hybrid_vals, extremity_vals),
            corr_p_value(pearson_corr(hybrid_vals, extremity_vals), len(hybrid_vals)),
        ),
    }

    statement_rows = statement_sensitivity(cleaned, args.min_statement_n)

    write_summary(
        args.output_summary,
        metrics,
        participant_corrs,
        statement_rows,
        statement_texts,
        args.top_n,
    )


if __name__ == "__main__":
    main()
