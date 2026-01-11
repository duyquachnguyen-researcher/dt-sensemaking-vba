"""Analyze whether raw statements land in different regions than seed statements for Q18."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import StatisticsError, mean, median, stdev
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
    parser = argparse.ArgumentParser(description="Q18 raw vs seed placement comparison")
    parser.add_argument(
        "--input",
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
        default=Path("outputs/q18_raw_vs_seed_metrics.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("outputs/q18_summary.txt"),
        help="Output summary text path",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of top cells to list per group",
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


def load_statement_ids(path: Path) -> set[str]:
    rows = read_rows(path)
    ids: set[str] = set()
    for row in rows:
        statement_id = (row.get("id") or "").strip()
        if statement_id:
            ids.add(statement_id)
    return ids


def cleaned_rows(
    rows: Iterable[dict[str, str]],
    raw_ids: set[str],
    seed_ids: set[str],
) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        token = (row.get("token") or "").strip()
        statement_id = (row.get("canonical_id") or "").strip()
        if not token or not statement_id:
            continue
        if statement_id in raw_ids:
            statement_type = "raw"
        elif statement_id in seed_ids:
            statement_type = "seed"
        else:
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
                "statement_type": statement_type,
                "x": x_val,
                "y": y_val,
            }
        )
    return cleaned


def cell_id_for(x_val: int, y_val: int) -> int:
    return (x_val - 1) * GRID_MAX + y_val


def compute_group_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tokens = {row["token"] for row in rows}
    x_vals = [row["x"] for row in rows]
    y_vals = [row["y"] for row in rows]
    extreme_values = {GRID_MIN, GRID_MAX}
    extreme_count = sum(1 for row in rows if row["x"] in extreme_values or row["y"] in extreme_values)
    stats: dict[str, Any] = {
        "placements": len(rows),
        "participants": len(tokens),
        "mean_x": mean(x_vals) if x_vals else float("nan"),
        "mean_y": mean(y_vals) if y_vals else float("nan"),
        "median_x": median(x_vals) if x_vals else float("nan"),
        "median_y": median(y_vals) if y_vals else float("nan"),
        "sd_x": float("nan"),
        "sd_y": float("nan"),
        "extremity_rate": (extreme_count / len(rows)) if rows else float("nan"),
    }
    try:
        stats["sd_x"] = stdev(x_vals) if len(x_vals) > 1 else float("nan")
    except StatisticsError:
        stats["sd_x"] = float("nan")
    try:
        stats["sd_y"] = stdev(y_vals) if len(y_vals) > 1 else float("nan")
    except StatisticsError:
        stats["sd_y"] = float("nan")
    return stats


def cliffs_delta(values_a: list[float], values_b: list[float]) -> float:
    if not values_a or not values_b:
        return float("nan")
    greater = 0
    less = 0
    for a_val in values_a:
        for b_val in values_b:
            if a_val > b_val:
                greater += 1
            elif a_val < b_val:
                less += 1
    total = len(values_a) * len(values_b)
    if total == 0:
        return float("nan")
    return (greater - less) / total


def mann_whitney(values_a: list[float], values_b: list[float]) -> tuple[float, str]:
    if len(values_a) < 2 or len(values_b) < 2:
        return float("nan"), "insufficient data"
    try:
        import importlib.util

        if importlib.util.find_spec("scipy") is not None:
            from scipy import stats  # noqa: WPS433 (optional dependency)

            result = stats.mannwhitneyu(values_a, values_b, alternative="two-sided")
            return float(result.pvalue), "mannwhitneyu (scipy)"
    except Exception:
        pass
    return float("nan"), "scipy unavailable"


def top_cells(rows: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    counts: Counter[tuple[int, int]] = Counter()
    for row in rows:
        counts[(int(row["x"]), int(row["y"]))] += 1
    total = len(rows)
    top = counts.most_common(top_n)
    results = []
    for (x_val, y_val), count in top:
        results.append(
            {
                "x": x_val,
                "y": y_val,
                "cell_id": cell_id_for(x_val, y_val),
                "count": count,
                "percent": (count / total) * 100 if total else 0.0,
            }
        )
    return results


def write_group_csv(path: Path, metrics: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "statement_type",
        "placements",
        "participants",
        "mean_x",
        "mean_y",
        "median_x",
        "median_y",
        "sd_x",
        "sd_y",
        "extremity_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for statement_type, stats in metrics.items():
            writer.writerow(
                {
                    "statement_type": statement_type,
                    **{key: stats[key] for key in fieldnames if key != "statement_type"},
                }
            )


def format_summary(
    group_metrics: dict[str, dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
    top_raw: list[dict[str, Any]],
    top_seed: list[dict[str, Any]],
    x_p_value: float,
    x_method: str,
    y_p_value: float,
    y_method: str,
    x_delta: float,
    y_delta: float,
    sanity: dict[str, int],
) -> str:
    raw_metrics = group_metrics.get("raw", {})
    seed_metrics = group_metrics.get("seed", {})
    raw_x = [row["x"] for row in raw_rows]
    seed_x = [row["x"] for row in seed_rows]
    raw_y = [row["y"] for row in raw_rows]
    seed_y = [row["y"] for row in seed_rows]
    mean_x_diff = raw_metrics.get("mean_x", float("nan")) - seed_metrics.get("mean_x", float("nan"))
    mean_y_diff = raw_metrics.get("mean_y", float("nan")) - seed_metrics.get("mean_y", float("nan"))
    median_x_diff = raw_metrics.get("median_x", float("nan")) - seed_metrics.get("median_x", float("nan"))
    median_y_diff = raw_metrics.get("median_y", float("nan")) - seed_metrics.get("median_y", float("nan"))

    lines = [
        "Q18 Raw vs Seed Placement Summary",
        "==================================",
        f"Raw placements analyzed: {raw_metrics.get('placements', 0)}",
        f"Raw participants represented: {raw_metrics.get('participants', 0)}",
        f"Seed placements analyzed: {seed_metrics.get('placements', 0)}",
        f"Seed participants represented: {seed_metrics.get('participants', 0)}",
        "",
        "Central tendency",
        "----------------",
        (
            "Raw mean (x, y): "
            f"{raw_metrics.get('mean_x', float('nan')):.3f}, {raw_metrics.get('mean_y', float('nan')):.3f}"
        ),
        (
            "Seed mean (x, y): "
            f"{seed_metrics.get('mean_x', float('nan')):.3f}, {seed_metrics.get('mean_y', float('nan')):.3f}"
        ),
        (
            "Raw median (x, y): "
            f"{raw_metrics.get('median_x', float('nan')):.3f}, {raw_metrics.get('median_y', float('nan')):.3f}"
        ),
        (
            "Seed median (x, y): "
            f"{seed_metrics.get('median_x', float('nan')):.3f}, {seed_metrics.get('median_y', float('nan')):.3f}"
        ),
        f"Mean difference raw - seed (x, y): {mean_x_diff:.3f}, {mean_y_diff:.3f}",
        f"Median difference raw - seed (x, y): {median_x_diff:.3f}, {median_y_diff:.3f}",
        "",
        "Spread and extremity",
        "--------------------",
        f"Raw SD (x, y): {raw_metrics.get('sd_x', float('nan')):.3f}, {raw_metrics.get('sd_y', float('nan')):.3f}",
        f"Seed SD (x, y): {seed_metrics.get('sd_x', float('nan')):.3f}, {seed_metrics.get('sd_y', float('nan')):.3f}",
        f"Raw extremity rate: {raw_metrics.get('extremity_rate', float('nan')):.2%}",
        f"Seed extremity rate: {seed_metrics.get('extremity_rate', float('nan')):.2%}",
        "",
        "Nonparametric comparison",
        "------------------------",
        f"Mann-Whitney p-value for X: {x_p_value:.5f} ({x_method})",
        f"Mann-Whitney p-value for Y: {y_p_value:.5f} ({y_method})",
        f"Cliff's delta (raw - seed) for X: {x_delta:.3f}",
        f"Cliff's delta (raw - seed) for Y: {y_delta:.3f}",
        "",
        "Top cells (raw)",
        "---------------",
    ]
    if top_raw:
        for cell in top_raw:
            lines.append(
                f"Cell ({cell['x']},{cell['y']}) [ID {cell['cell_id']}]: {cell['count']} placements "
                f"({cell['percent']:.2f}%)"
            )
    else:
        lines.append("No raw placements available.")
    lines.extend(["", "Top cells (seed)", "----------------"])
    if top_seed:
        for cell in top_seed:
            lines.append(
                f"Cell ({cell['x']},{cell['y']}) [ID {cell['cell_id']}]: {cell['count']} placements "
                f"({cell['percent']:.2f}%)"
            )
    else:
        lines.append("No seed placements available.")
    lines.extend(
        [
            "",
            "Sanity checks",
            "-------------",
            f"Rows in input: {sanity.get('rows_input', 0)}",
            f"Rows after dedup: {sanity.get('rows_dedup', 0)}",
            f"Rows after raw/seed filter: {sanity.get('rows_filtered', 0)}",
            f"Rows after cleaning x/y: {sanity.get('rows_cleaned', 0)}",
            f"Raw placements after cleaning: {len(raw_rows)}",
            f"Seed placements after cleaning: {len(seed_rows)}",
            f"Raw X range: {min(raw_x) if raw_x else 'n/a'} to {max(raw_x) if raw_x else 'n/a'}",
            f"Raw Y range: {min(raw_y) if raw_y else 'n/a'} to {max(raw_y) if raw_y else 'n/a'}",
            f"Seed X range: {min(seed_x) if seed_x else 'n/a'} to {max(seed_x) if seed_x else 'n/a'}",
            f"Seed Y range: {min(seed_y) if seed_y else 'n/a'} to {max(seed_y) if seed_y else 'n/a'}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    deduped = deduplicate_latest(rows)
    seed_ids = load_statement_ids(args.seed_statements)
    raw_ids = load_statement_ids(args.raw_statements)
    filtered = cleaned_rows(deduped, raw_ids, seed_ids)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in filtered:
        grouped[row["statement_type"]].append(row)

    raw_rows = grouped.get("raw", [])
    seed_rows = grouped.get("seed", [])
    group_metrics = {
        "raw": compute_group_stats(raw_rows),
        "seed": compute_group_stats(seed_rows),
    }
    write_group_csv(args.output_csv, group_metrics)

    x_p_value, x_method = mann_whitney([row["x"] for row in raw_rows], [row["x"] for row in seed_rows])
    y_p_value, y_method = mann_whitney([row["y"] for row in raw_rows], [row["y"] for row in seed_rows])
    x_delta = cliffs_delta([row["x"] for row in raw_rows], [row["x"] for row in seed_rows])
    y_delta = cliffs_delta([row["y"] for row in raw_rows], [row["y"] for row in seed_rows])

    top_raw = top_cells(raw_rows, args.top_n)
    top_seed = top_cells(seed_rows, args.top_n)

    summary = format_summary(
        group_metrics=group_metrics,
        raw_rows=raw_rows,
        seed_rows=seed_rows,
        top_raw=top_raw,
        top_seed=top_seed,
        x_p_value=x_p_value,
        x_method=x_method,
        y_p_value=y_p_value,
        y_method=y_method,
        x_delta=x_delta,
        y_delta=y_delta,
        sanity={
            "rows_input": len(rows),
            "rows_dedup": len(deduped),
            "rows_filtered": len(filtered),
            "rows_cleaned": len(filtered),
        },
    )
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
