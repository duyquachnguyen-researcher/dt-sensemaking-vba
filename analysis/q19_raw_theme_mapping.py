"""Analyze raw statement themes and where they land on the grid for Q19.

This script assigns rule-based themes to raw statements, then aggregates
placements to compute centroid and dispersion per theme.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
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

THEME_RULES: list[tuple[str, tuple[str, ...]]] = [
    (
        "Training & Skills",
        (
            "training",
            "skills",
            "learning",
            "self learning",
            "knowledge",
            "upskill",
            "refresh",
        ),
    ),
    (
        "Communication & Change",
        (
            "communicate",
            "communication",
            "inform",
            "clarify",
            "share",
            "sharing",
            "update",
            "roadmap",
            "commitment",
            "embrace",
        ),
    ),
    (
        "Goals, KPIs & Strategy",
        (
            "goal",
            "objective",
            "kpi",
            "milestone",
            "strategic",
            "strategy",
            "roadmap",
            "target",
            "requirement",
        ),
    ),
    (
        "Process & Compliance",
        (
            "process",
            "procedure",
            "compliance",
            "regulation",
            "adherence",
            "accuracy",
            "report",
            "documentation",
            "check",
            "control",
        ),
    ),
    (
        "Technology & Infrastructure",
        (
            "technology",
            "system",
            "software",
            "infrastructure",
            "cloud",
            "security",
            "license",
            "application",
            "migrate",
            "tool",
        ),
    ),
    (
        "People & Resourcing",
        (
            "employee",
            "personnel",
            "team",
            "leadership",
            "manager",
            "reward",
            "punishment",
            "budget",
            "resource",
            "assign",
            "talent",
        ),
    ),
    (
        "Data & Integration",
        (
            "data",
            "interconnect",
            "integration",
            "share data",
            "analytics",
            "ai",
        ),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q19 raw theme mapping analysis")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Placement.csv"),
        help="Path to Placement.csv",
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
        default=Path("outputs/q19_theme_mapping.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("outputs/q19_summary.txt"),
        help="Output summary text path",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of themes to highlight",
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


def load_raw_statements(path: Path) -> dict[str, str]:
    rows = read_rows(path)
    return {row["id"].strip(): (row.get("text") or "").strip() for row in rows if row.get("id")}


def classify_theme(text: str) -> str:
    normalized = " ".join(text.lower().split())
    for theme, keywords in THEME_RULES:
        for keyword in keywords:
            if keyword in normalized:
                return theme
    return "Other / Placeholder"


def cleaned_rows(
    rows: Iterable[dict[str, str]], raw_map: dict[str, str]
) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        token = (row.get("token") or "").strip()
        statement_id = (row.get("canonical_id") or "").strip()
        if not token or not statement_id:
            continue
        statement_text = raw_map.get(statement_id)
        if not statement_text:
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
                "statement_text": statement_text,
                "theme": classify_theme(statement_text),
                "x": x_val,
                "y": y_val,
            }
        )
    return cleaned


def theme_stats(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[Any]]] = defaultdict(
        lambda: {"x": [], "y": [], "tokens": [], "statements": []}
    )
    for row in rows:
        theme = row["theme"]
        grouped[theme]["x"].append(row["x"])
        grouped[theme]["y"].append(row["y"])
        grouped[theme]["tokens"].append(row["token"])
        grouped[theme]["statements"].append(row["statement_id"])

    results: list[dict[str, Any]] = []
    for theme, data in grouped.items():
        x_vals = data["x"]
        y_vals = data["y"]
        results.append(
            {
                "theme": theme,
                "n_statements": len(set(data["statements"])),
                "n_participants": len(set(data["tokens"])),
                "n_placements": len(x_vals),
                "mean_x": mean(x_vals),
                "mean_y": mean(y_vals),
                "sd_x": stdev(x_vals) if len(x_vals) > 1 else 0.0,
                "sd_y": stdev(y_vals) if len(y_vals) > 1 else 0.0,
            }
        )
    return results


def summarize(
    stats: list[dict[str, Any]],
    cleaned: list[dict[str, Any]],
    top_n: int,
) -> str:
    lines = [
        "Q19 Raw Theme Mapping Summary",
        "==============================",
        f"Themes included: {len(stats)}",
        "",
        "Theme centroids",
        "--------------",
    ]

    for row in sorted(stats, key=lambda item: (-item["n_placements"], -item["mean_y"])):
        lines.append(
            "{theme}: mean_x={mean_x:.2f}, mean_y={mean_y:.2f}, sd_x={sd_x:.2f}, "
            "sd_y={sd_y:.2f}, placements={n_placements}, statements={n_statements}".format(**row)
        )

    theme_examples: dict[str, list[str]] = defaultdict(list)
    for row in cleaned:
        theme = row["theme"]
        if len(theme_examples[theme]) < 2:
            statement = row["statement_text"].strip()
            if statement and statement not in theme_examples[theme]:
                theme_examples[theme].append(statement)

    lines.extend(["", "Example statements", "------------------"])
    for theme, examples in sorted(theme_examples.items()):
        lines.append(f"{theme}:")
        for example in examples:
            lines.append(f"  - {example}")

    if stats:
        top_themes = sorted(stats, key=lambda item: item["n_placements"], reverse=True)[: max(top_n, 0)]
        lines.extend(["", "Top themes by placements", "------------------------"])
        for row in top_themes:
            lines.append(
                f"{row['theme']}: placements={row['n_placements']}, mean_x={row['mean_x']:.2f}, "
                f"mean_y={row['mean_y']:.2f}"
            )
    else:
        lines.append("No raw statement placements available after filtering.")

    return "\n".join(lines)


def write_csv(path: Path, stats: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "theme",
        "n_statements",
        "n_participants",
        "n_placements",
        "mean_x",
        "mean_y",
        "sd_x",
        "sd_y",
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

    raw_map = load_raw_statements(args.raw_statements)
    cleaned = cleaned_rows(deduped, raw_map)
    stats = theme_stats(cleaned)

    write_csv(args.output_csv, stats)

    summary_text = summarize(stats, cleaned, args.top_n)
    participants = {row["token"] for row in cleaned}
    statements_with_data = {row["statement_id"] for row in cleaned}

    sanity_lines = [
        "",
        "Sanity checks",
        "-------------",
        f"Rows in input: {raw_rows}",
        f"Rows after dedup: {dedup_rows}",
        f"Rows after raw filter: {len(cleaned)}",
        f"Participants (unique tokens): {len(participants)}",
        f"Raw statements with placements: {len(statements_with_data)}",
        f"Raw statements total: {len(raw_map)}",
    ]
    summary_text = summary_text + "\n" + "\n".join(sanity_lines)

    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(summary_text, encoding="utf-8")

    print(summary_text)


if __name__ == "__main__":
    main()
