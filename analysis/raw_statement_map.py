"""Create a map of raw statements using their placement centroids."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean

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
GRID_RANGE = GRID_MAX - GRID_MIN

AXIS_STROKE = 8
GRID_STROKE = 2
LABEL_FONT_SIZE = 12
POINT_RADIUS = 4


def load_raw_statements() -> dict[str, str]:
    statements: dict[str, str] = {}
    with RAW_STATEMENT_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            statement_id = row.get("id")
            text = row.get("text")
            if statement_id and text:
                statements[statement_id] = text.strip()
    return statements


def load_placements() -> dict[str, list[tuple[float, float]]]:
    placements: dict[str, list[tuple[float, float]]] = {}
    with PLACEMENT_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            statement_id = row.get("canonical_id")
            if not statement_id:
                continue
            try:
                x_value = float(row.get("x", ""))
                y_value = float(row.get("y", ""))
            except ValueError:
                continue
            placements.setdefault(statement_id, []).append((x_value, y_value))
    return placements


def compute_centroids(placements: dict[str, list[tuple[float, float]]]) -> dict[str, tuple[float, float]]:
    centroids: dict[str, tuple[float, float]] = {}
    for statement_id, coords in placements.items():
        if not coords:
            continue
        xs = [point[0] for point in coords]
        ys = [point[1] for point in coords]
        centroids[statement_id] = (mean(xs), mean(ys))
    return centroids


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


def wrap_text(text: str, max_len: int = 32) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        extra = len(word) + (1 if current else 0)
        if length + extra > max_len and current:
            lines.append(" ".join(current))
            current = [word]
            length = len(word)
        else:
            current.append(word)
            length += extra
    if current:
        lines.append(" ".join(current))
    return lines


def svg_points(raw_statements: dict[str, str], centroids: dict[str, tuple[float, float]]) -> str:
    elements = []
    for statement_id, text in sorted(raw_statements.items()):
        centroid = centroids.get(statement_id)
        if not centroid:
            continue
        x = scale_x(centroid[0])
        y = scale_y(centroid[1])
        elements.append(
            f"<circle cx='{x:.2f}' cy='{y:.2f}' r='{POINT_RADIUS}' "
            "fill='#1f77b4' stroke='black' stroke-width='1' />"
        )
        lines = wrap_text(text)
        label_x = min(x + 8, PLOT_RIGHT - 10)
        label_y = max(y - 8, PLOT_TOP + LABEL_FONT_SIZE)
        elements.append(
            f"<text x='{label_x:.2f}' y='{label_y:.2f}' font-size='{LABEL_FONT_SIZE}' "
            "font-family='Arial' fill='black'>"
        )
        for idx, line in enumerate(lines):
            dy = 0 if idx == 0 else LABEL_FONT_SIZE + 2
            elements.append(
                f"<tspan x='{label_x:.2f}' dy='{dy}'>{line}</tspan>"
            )
        elements.append("</text>")
    return "\n".join(elements) + "\n"


def build_svg(raw_statements: dict[str, str], centroids: dict[str, tuple[float, float]]) -> str:
    content = [
        svg_header(),
        svg_defs(),
        "<rect width='100%' height='100%' fill='white' />\n",
        svg_grid(),
        svg_axes(),
        svg_points(raw_statements, centroids),
        "</svg>\n",
    ]
    return "".join(content)


def main() -> None:
    raw_statements = load_raw_statements()
    placements = load_placements()
    centroids = compute_centroids(placements)
    svg = build_svg(raw_statements, centroids)
    OUTPUT_PATH.write_text(svg, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
