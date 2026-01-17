"""Draw theme centroids with spread ellipses for the four themes."""
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

THEME_COLORS = [
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw theme centroids and spread ellipses")
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
        "--theme-column",
        default="cluster",
        help="Column in StatementPub.csv that stores the theme label",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/theme_centroids_spread.svg"),
        help="Output SVG file",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=900,
        help="SVG width in pixels",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=900,
        help="SVG height in pixels",
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


def load_theme_mapping(path: Path, theme_column: str) -> dict[str, str]:
    rows = read_rows(path)
    mapping: dict[str, str] = {}
    for row in rows:
        statement_id = (row.get("id") or "").strip()
        if not statement_id:
            continue
        theme = (row.get(theme_column) or "").strip()
        if not theme and theme_column != "cluster":
            theme = (row.get("cluster") or "").strip()
        if not theme:
            theme = (row.get("tags") or "").strip()
        if not theme:
            continue
        mapping[statement_id] = theme
    return mapping


def cleaned_rows(rows: Iterable[dict[str, str]], theme_map: dict[str, str]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        token = (row.get("token") or "").strip()
        statement_id = (row.get("canonical_id") or "").strip()
        if not token or not statement_id:
            continue
        theme = theme_map.get(statement_id)
        if not theme:
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
                "theme": theme,
                "x": x_val,
                "y": y_val,
            }
        )
    return cleaned


def theme_stats(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[Any]]] = defaultdict(lambda: {"x": [], "y": []})
    for row in rows:
        theme = row["theme"]
        grouped[theme]["x"].append(row["x"])
        grouped[theme]["y"].append(row["y"])

    results: list[dict[str, Any]] = []
    for theme, data in grouped.items():
        x_vals = data["x"]
        y_vals = data["y"]
        results.append(
            {
                "theme": theme,
                "mean_x": mean(x_vals),
                "mean_y": mean(y_vals),
                "sd_x": stdev(x_vals) if len(x_vals) > 1 else 0.0,
                "sd_y": stdev(y_vals) if len(y_vals) > 1 else 0.0,
                "n": len(x_vals),
            }
        )
    return sorted(results, key=lambda item: item["theme"])


def map_point(x: float, y: float, width: int, height: int, margin: int) -> tuple[float, float]:
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin
    scale_x = plot_width / (GRID_MAX - GRID_MIN)
    scale_y = plot_height / (GRID_MAX - GRID_MIN)
    px = margin + (x - GRID_MIN) * scale_x
    py = height - margin - (y - GRID_MIN) * scale_y
    return px, py


def svg_header(width: int, height: int) -> str:
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}'>\n"
        "<defs>\n"
        "  <marker id='axis-arrow' markerWidth='12' markerHeight='12' refX='10' refY='6' orient='auto'>\n"
        "    <path d='M0,0 L12,6 L0,12 Z' fill='#000' />\n"
        "  </marker>\n"
        "</defs>"
    )


def draw_grid(width: int, height: int, margin: int) -> list[str]:
    lines: list[str] = []
    for tick in range(GRID_MIN, GRID_MAX + 1):
        x1, y1 = map_point(tick, GRID_MIN, width, height, margin)
        x2, y2 = map_point(tick, GRID_MAX, width, height, margin)
        lines.append(
            f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' "
            "stroke='#000' stroke-width='1' />"
        )
        x1, y1 = map_point(GRID_MIN, tick, width, height, margin)
        x2, y2 = map_point(GRID_MAX, tick, width, height, margin)
        lines.append(
            f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' "
            "stroke='#000' stroke-width='1' />"
        )
    return lines


def draw_axes(width: int, height: int, margin: int) -> list[str]:
    x_min, y_min = map_point(GRID_MIN, GRID_MIN, width, height, margin)
    x_max, y_max = map_point(GRID_MAX, GRID_MAX, width, height, margin)
    return [
        f"<line x1='{x_min:.1f}' y1='{y_min:.1f}' x2='{x_max + 20:.1f}' "
        f"y2='{y_min:.1f}' stroke='#000' stroke-width='6' marker-end='url(#axis-arrow)' />",
        f"<line x1='{x_min:.1f}' y1='{y_min:.1f}' x2='{x_min:.1f}' "
        f"y2='{y_max - 20:.1f}' stroke='#000' stroke-width='6' marker-end='url(#axis-arrow)' />",
    ]


def draw_labels(width: int, height: int, margin: int) -> list[str]:
    labels: list[str] = []
    for tick in range(GRID_MIN, GRID_MAX + 1):
        x, y = map_point(tick, GRID_MIN, width, height, margin)
        labels.append(
            f"<text x='{x:.1f}' y='{y + 30:.1f}' font-size='18' "
            f"fill='#000' text-anchor='middle'>{tick}</text>"
        )
        x, y = map_point(GRID_MIN, tick, width, height, margin)
        labels.append(
            f"<text x='{x - 20:.1f}' y='{y + 6:.1f}' font-size='18' "
            f"fill='#000' text-anchor='end'>{tick}</text>"
        )
    labels.append(
        f"<text x='{width / 2:.1f}' y='{height - margin / 3:.1f}' font-size='20' "
        "fill='#000' text-anchor='middle'>Implementability (1 = Hard, 7 = Easy)</text>"
    )
    labels.append(
        f"<text x='{margin / 3:.1f}' y='{height / 2:.1f}' font-size='20' "
        "fill='#000' text-anchor='middle' transform="
        f"'rotate(-90 {margin / 3:.1f} {height / 2:.1f})'>"
        "Degree of impact (1 = Low, 7 = High)</text>"
    )
    return labels


def draw_legend(stats: list[dict[str, Any]], width: int, margin: int) -> list[str]:
    entries: list[str] = []
    start_x = width - margin + 10
    start_y = margin
    for idx, row in enumerate(stats):
        color = THEME_COLORS[idx % len(THEME_COLORS)]
        y = start_y + idx * 24
        entries.append(
            f"<circle cx='{start_x}' cy='{y}' r='6' fill='{color}' />"
        )
        entries.append(
            f"<text x='{start_x + 12}' y='{y + 4}' font-size='12' fill='#333'>"
            f"{row['theme']} (n={row['n']})</text>"
        )
    return entries


def draw_themes(stats: list[dict[str, Any]], width: int, height: int, margin: int) -> list[str]:
    elements: list[str] = []
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin
    scale_x = plot_width / (GRID_MAX - GRID_MIN)
    scale_y = plot_height / (GRID_MAX - GRID_MIN)

    for idx, row in enumerate(stats):
        color = THEME_COLORS[idx % len(THEME_COLORS)]
        cx, cy = map_point(row["mean_x"], row["mean_y"], width, height, margin)
        rx = max(row["sd_x"] * scale_x, 4)
        ry = max(row["sd_y"] * scale_y, 4)
        elements.append(
            f"<ellipse cx='{cx:.1f}' cy='{cy:.1f}' rx='{rx:.1f}' ry='{ry:.1f}' "
            f"fill='{color}' fill-opacity='0.15' stroke='{color}' stroke-width='2' />"
        )
        elements.append(
            f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='6' fill='{color}' stroke='#222' stroke-width='1' />"
        )
        elements.append(
            f"<text x='{cx + 10:.1f}' y='{cy - 8:.1f}' font-size='12' fill='#111'>"
            f"{row['theme']}</text>"
        )
    return elements


def build_svg(stats: list[dict[str, Any]], width: int, height: int) -> str:
    margin = 110
    parts = [svg_header(width, height)]
    parts.extend(draw_grid(width, height, margin))
    parts.extend(draw_axes(width, height, margin))
    parts.extend(draw_themes(stats, width, height, margin))
    parts.extend(draw_labels(width, height, margin))
    parts.extend(draw_legend(stats, width, margin))
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    deduped = deduplicate_latest(rows)
    theme_map = load_theme_mapping(args.statements, args.theme_column)
    cleaned = cleaned_rows(deduped, theme_map)
    stats = theme_stats(cleaned)
    if not stats:
        raise SystemExit("No theme placements available to plot.")

    svg_text = build_svg(stats, args.width, args.height)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg_text, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
