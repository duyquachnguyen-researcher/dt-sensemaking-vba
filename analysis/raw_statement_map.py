"""Create a map of raw statements using their placement centroids."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
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
GRID_RANGE = GRID_MAX - GRID_MIN

AXIS_STROKE = 8
GRID_STROKE = 2
POINT_RADIUS = 8
JITTER_STEP = 0.07
JITTER_MAX = 0.18


def load_placements() -> list[dict[str, str]]:
    placements: list[dict[str, str]] = []
    with PLACEMENT_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row.get("x") or not row.get("y"):
                continue
            placements.append(row)
    return placements


def scale_x(value: float) -> float:
    return PLOT_LEFT + (value - GRID_MIN) / GRID_RANGE * PLOT_WIDTH


def scale_y(value: float) -> float:
    return PLOT_BOTTOM - (value - GRID_MIN) / GRID_RANGE * PLOT_HEIGHT


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
    for i in range(GRID_MIN, GRID_MAX + 1):
        x = scale_x(i)
        y = scale_y(i)
        lines.append(
            f"<line x1='{x:.2f}' y1='{PLOT_TOP}' x2='{x:.2f}' y2='{PLOT_BOTTOM}' "
            f"stroke='black' stroke-width='{GRID_STROKE}' />"
        )
        lines.append(
            f"<line x1='{PLOT_LEFT}' y1='{y:.2f}' x2='{PLOT_RIGHT}' y2='{y:.2f}' "
            f"stroke='black' stroke-width='{GRID_STROKE}' />"
        )
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


def build_offsets(count: int) -> list[tuple[float, float]]:
    offsets: list[tuple[float, float]] = [(0.0, 0.0)]
    ring = 1
    while len(offsets) < count:
        step = ring * JITTER_STEP
        candidates = [
            (step, 0.0),
            (-step, 0.0),
            (0.0, step),
            (0.0, -step),
            (step, step),
            (step, -step),
            (-step, step),
            (-step, -step),
        ]
        offsets.extend(candidates)
        ring += 1
    return offsets[:count]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def svg_points(rows: list[dict[str, str]]) -> str:
    elements = []
    grouped: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        try:
            x_int = int(round(float(row["x"])))
            y_int = int(round(float(row["y"])))
        except (KeyError, TypeError, ValueError):
            continue
        grouped[(x_int, y_int)].append(row)
    for (x_int, y_int), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda r: (r.get("canonical_id", ""), r.get("id", "")))
        offsets = build_offsets(len(ordered))
        for row, (dx, dy) in zip(ordered, offsets):
            dx = clamp(dx, -JITTER_MAX, JITTER_MAX)
            dy = clamp(dy, -JITTER_MAX, JITTER_MAX)
            x = scale_x(x_int + dx)
            y = scale_y(y_int + dy)
            elements.append(
                f"<circle cx='{x:.2f}' cy='{y:.2f}' r='{POINT_RADIUS}' "
                "fill='white' fill-opacity='0.9' stroke='black' stroke-width='2' />"
            )
    return "\n".join(elements) + "\n"


def build_svg(rows: list[dict[str, str]]) -> str:
    content = [
        svg_header(),
        svg_defs(),
        "<rect width='100%' height='100%' fill='white' />\n",
        svg_grid(),
        svg_axes(),
        svg_points(rows),
        "</svg>\n",
    ]
    return "".join(content)


def main() -> None:
    placements = load_placements()
    svg = build_svg(placements)
    OUTPUT_PATH.write_text(svg, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
