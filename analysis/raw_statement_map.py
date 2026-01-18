"""Create a bubble count heatmap for raw statement placements."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_STATEMENT_PATH = REPO_ROOT / "StatementRaw.csv"
PLACEMENT_PATH = REPO_ROOT / "Placement.csv"
OUTPUT_PATH = REPO_ROOT / "outputs" / "raw_statement_map.svg"

WIDTH = 1200
HEIGHT = 1100
PLOT_LEFT = 130
PLOT_TOP = 120
PLOT_WIDTH = 820
PLOT_HEIGHT = 820
PLOT_RIGHT = PLOT_LEFT + PLOT_WIDTH
PLOT_BOTTOM = PLOT_TOP + PLOT_HEIGHT

GRID_MIN = 1
GRID_MAX = 7
GRID_BOUND_MIN = 0.5
GRID_BOUND_MAX = 7.5
GRID_RANGE = GRID_BOUND_MAX - GRID_BOUND_MIN

AXIS_STROKE = 8
GRID_STROKE = 2
R_MIN = 6
R_MAX = 22
LABEL_MIN_COUNT = 5
LEGEND_X = PLOT_RIGHT + 35
LEGEND_Y = PLOT_TOP + 20


def load_raw_ids() -> set[str]:
    raw_ids: set[str] = set()
    with RAW_STATEMENT_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            statement_id = row.get("id")
            if statement_id:
                raw_ids.add(statement_id)
    return raw_ids


def load_placements(raw_ids: set[str]) -> list[dict[str, str]]:
    placements: list[dict[str, str]] = []
    with PLACEMENT_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row.get("x") or not row.get("y"):
                continue
            if row.get("canonical_id") not in raw_ids:
                continue
            placements.append(row)
    return placements


def scale_x(value: float) -> float:
    return PLOT_LEFT + (value - GRID_BOUND_MIN) / GRID_RANGE * PLOT_WIDTH


def scale_y(value: float) -> float:
    return PLOT_BOTTOM - (value - GRID_BOUND_MIN) / GRID_RANGE * PLOT_HEIGHT


def svg_header() -> str:
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{WIDTH}' height='{HEIGHT}' "
        f"viewBox='0 0 {WIDTH} {HEIGHT}'>\n"
    )


def svg_defs() -> str:
    return (
        "<defs>\n"
        "  <marker id='arrow' viewBox='0 0 10 10' refX='8' refY='5' "
        "markerWidth='8' markerHeight='8' orient='auto-start-reverse'>\n"
        "    <path d='M 0 0 L 10 5 L 0 10 z' fill='black' />\n"
        "  </marker>\n"
        "</defs>\n"
    )


def svg_grid() -> str:
    lines = []
    for boundary in range(GRID_MIN, GRID_MAX + 2):
        position = boundary - 0.5
        x = scale_x(position)
        y = scale_y(position)
        lines.append(
            f"<line x1='{x:.2f}' y1='{PLOT_TOP}' x2='{x:.2f}' y2='{PLOT_BOTTOM}' "
            f"stroke='black' stroke-width='{GRID_STROKE}' />"
        )
        lines.append(
            f"<line x1='{PLOT_LEFT}' y1='{y:.2f}' x2='{PLOT_RIGHT}' y2='{y:.2f}' "
            f"stroke='black' stroke-width='{GRID_STROKE}' />"
        )
    for i in range(GRID_MIN, GRID_MAX + 1):
        x = scale_x(i)
        y = scale_y(i)
        lines.append(
            f"<text x='{x:.2f}' y='{PLOT_BOTTOM + 40}' font-size='28' "
            "text-anchor='middle' font-family='Arial'>"
            f"{i}</text>"
        )
        lines.append(
            f"<text x='{PLOT_LEFT - 30}' y='{y + 10:.2f}' font-size='28' "
            "text-anchor='middle' font-family='Arial'>"
            f"{i}</text>"
        )
    return "\n".join(lines) + "\n"


def svg_axes() -> str:
    axis = [
        (
            f"<line x1='{PLOT_LEFT}' y1='{PLOT_BOTTOM}' x2='{PLOT_RIGHT + 50}' "
            f"y2='{PLOT_BOTTOM}' stroke='black' stroke-width='{AXIS_STROKE}' "
            "marker-end='url(#arrow)' />"
        ),
        (
            f"<line x1='{PLOT_LEFT}' y1='{PLOT_BOTTOM}' x2='{PLOT_LEFT}' "
            f"y2='{PLOT_TOP - 50}' stroke='black' stroke-width='{AXIS_STROKE}' "
            "marker-end='url(#arrow)' />"
        ),
        (
            f"<text x='{(PLOT_LEFT + PLOT_RIGHT) / 2:.2f}' y='{PLOT_BOTTOM + 110}' "
            "font-size='32' text-anchor='middle' font-family='Arial'>"
            "Implementability (1 = Hard, 7 = Easy)</text>"
        ),
        (
            f"<text x='{PLOT_LEFT - 90}' y='{(PLOT_TOP + PLOT_BOTTOM) / 2:.2f}' "
            "font-size='32' text-anchor='middle' font-family='Arial'"
            " transform='rotate(-90 "
            f"{PLOT_LEFT - 90} {(PLOT_TOP + PLOT_BOTTOM) / 2:.2f})'>"
            "Degree of impact (1 = Low, 7 = High)</text>"
        ),
    ]
    return "\n".join(axis) + "\n"


def bubble_radius(count: int, max_count: int) -> float:
    if max_count <= 1:
        return float(R_MIN)
    return R_MIN + ((count**0.5 - 1) / (max_count**0.5 - 1)) * (R_MAX - R_MIN)


def svg_points(rows: list[dict[str, str]]) -> str:
    elements = []
    cell_counts: Counter[tuple[int, int]] = Counter()
    for row in rows:
        try:
            x_int = int(round(float(row["x"])))
            y_int = int(round(float(row["y"])))
        except (KeyError, TypeError, ValueError):
            continue
        if GRID_MIN <= x_int <= GRID_MAX and GRID_MIN <= y_int <= GRID_MAX:
            cell_counts[(x_int, y_int)] += 1
    if not cell_counts:
        return ""
    max_count = max(cell_counts.values())
    for (x_int, y_int), count in sorted(cell_counts.items()):
        x = scale_x(x_int)
        y = scale_y(y_int)
        radius = bubble_radius(count, max_count)
        elements.append(
            f"<circle cx='{x:.2f}' cy='{y:.2f}' r='{radius:.2f}' "
            "fill='white' fill-opacity='0.85' stroke='black' stroke-width='2' />"
        )
        if count >= LABEL_MIN_COUNT:
            elements.append(
                f"<text x='{x:.2f}' y='{y + 6:.2f}' font-size='14' "
                "text-anchor='middle' font-family='Arial' stroke='white' "
                "stroke-width='4' paint-order='stroke'>"
                f"{count}</text>"
            )
            elements.append(
                f"<text x='{x:.2f}' y='{y + 6:.2f}' font-size='14' "
                "text-anchor='middle' font-family='Arial' fill='black'>"
                f"{count}</text>"
            )
    return "\n".join(elements) + "\n"


def svg_legend(max_count: int) -> str:
    if max_count <= 0:
        return ""
    sample_counts = [1]
    if max_count >= 5:
        sample_counts.append(5)
    if max_count > 1:
        sample_counts.append(max_count)
    legend_items = [
        (
            f"<text x='{LEGEND_X:.2f}' y='{LEGEND_Y:.2f}' font-size='22' "
            "font-family='Arial'>Bubble size = # placements</text>"
        )
    ]
    for index, count in enumerate(sample_counts):
        radius = bubble_radius(count, max_count)
        cy = LEGEND_Y + 40 + index * 60
        cx = LEGEND_X + 20 + radius
        legend_items.append(
            f"<circle cx='{cx:.2f}' cy='{cy:.2f}' r='{radius:.2f}' "
            "fill='white' fill-opacity='0.85' stroke='black' stroke-width='2' />"
        )
        label = "Max" if count == max_count else str(count)
        legend_items.append(
            f"<text x='{cx + radius + 12:.2f}' y='{cy + 5:.2f}' "
            "font-size='18' font-family='Arial'>"
            f"{label}</text>"
        )
    return "\n".join(legend_items) + "\n"


def build_svg(rows: list[dict[str, str]]) -> str:
    counts: Counter[tuple[int, int]] = Counter()
    for row in rows:
        try:
            x_int = int(round(float(row["x"])))
            y_int = int(round(float(row["y"])))
        except (KeyError, TypeError, ValueError):
            continue
        if GRID_MIN <= x_int <= GRID_MAX and GRID_MIN <= y_int <= GRID_MAX:
            counts[(x_int, y_int)] += 1
    max_count = max(counts.values(), default=0)
    content = [
        svg_header(),
        svg_defs(),
        "<rect width='100%' height='100%' fill='white' />\n",
        svg_grid(),
        svg_axes(),
        svg_points(rows),
        svg_legend(max_count),
        "</svg>\n",
    ]
    return "".join(content)


def main() -> None:
    raw_ids = load_raw_ids()
    placements = load_placements(raw_ids)
    svg = build_svg(placements)
    OUTPUT_PATH.write_text(svg, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
