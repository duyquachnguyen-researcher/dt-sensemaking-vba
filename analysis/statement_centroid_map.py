import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

REPO_ROOT = Path(__file__).resolve().parents[1]
STATEMENT_PATH = REPO_ROOT / "StatementPub.csv"
PLACEMENT_PATH = REPO_ROOT / "Placement.csv"
OUTPUT_PATH = REPO_ROOT / "outputs" / "statement_centroid_map.svg"

WIDTH = 931
HEIGHT = 985
PLOT_LEFT = 130
PLOT_TOP = 120
PLOT_WIDTH = 700
PLOT_HEIGHT = 700
PLOT_RIGHT = PLOT_LEFT + PLOT_WIDTH
PLOT_BOTTOM = PLOT_TOP + PLOT_HEIGHT

GRID_MIN = 1
GRID_MAX = 7
GRID_RANGE = GRID_MAX - GRID_MIN

AXIS_STROKE = 8
GRID_STROKE = 2
LEGEND_X = PLOT_RIGHT + 35
LEGEND_Y = PLOT_TOP + 20

COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#393b79",
    "#637939",
    "#8c6d31",
    "#843c39",
    "#7b4173",
    "#3182bd",
    "#31a354",
    "#756bb1",
    "#636363",
]


@dataclass
class StatementStats:
    statement_id: str
    display_order: int
    mean_x: float
    mean_y: float
    ci_x: float
    ci_y: float
    n: int


def load_statement_ids():
    statements = []
    with STATEMENT_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("status") != "Published":
                continue
            display_order = row.get("display_order")
            if not display_order:
                continue
            try:
                display_order_int = int(display_order)
            except ValueError:
                continue
            if display_order_int < 1 or display_order_int > 19:
                continue
            statement_id = row.get("id")
            if not statement_id:
                continue
            statements.append((display_order_int, statement_id))
    statements.sort()
    return statements


def load_placements():
    placements = {}
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


def compute_stats(statement_ids, placements):
    stats = []
    for display_order, statement_id in statement_ids:
        coords = placements.get(statement_id, [])
        if not coords:
            continue
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        n = len(coords)
        mean_x = mean(xs)
        mean_y = mean(ys)
        if n > 1:
            std_x = stdev(xs)
            std_y = stdev(ys)
            ci_x = 1.96 * std_x / math.sqrt(n)
            ci_y = 1.96 * std_y / math.sqrt(n)
        else:
            ci_x = 0.0
            ci_y = 0.0
        stats.append(StatementStats(statement_id, display_order, mean_x, mean_y, ci_x, ci_y, n))
    return stats


def scale_x(value):
    return PLOT_LEFT + (value - GRID_MIN) / GRID_RANGE * PLOT_WIDTH


def scale_y(value):
    return PLOT_BOTTOM - (value - GRID_MIN) / GRID_RANGE * PLOT_HEIGHT


def svg_header():
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{WIDTH}' height='{HEIGHT}' "
        f"viewBox='0 0 {WIDTH} {HEIGHT}'>\n"
    )


def svg_defs():
    return (
        "<defs>\n"
        "  <marker id='arrow' viewBox='0 0 10 10' refX='8' refY='5' "
        "markerWidth='8' markerHeight='8' orient='auto-start-reverse'>\n"
        "    <path d='M 0 0 L 10 5 L 0 10 z' fill='black' />\n"
        "  </marker>\n"
        "</defs>\n"
    )


def svg_grid():
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


def svg_axes():
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


def svg_points(stats):
    elements = []
    for idx, stat in enumerate(stats):
        color = COLORS[idx % len(COLORS)]
        x = scale_x(stat.mean_x)
        y = scale_y(stat.mean_y)
        rx = stat.ci_x / GRID_RANGE * PLOT_WIDTH
        ry = stat.ci_y / GRID_RANGE * PLOT_HEIGHT
        if rx > 0 or ry > 0:
            elements.append(
                f"<ellipse cx='{x:.2f}' cy='{y:.2f}' rx='{rx:.2f}' ry='{ry:.2f}' "
                f"fill='{color}' fill-opacity='0.15' stroke='{color}' stroke-width='2' />"
            )
        elements.append(
            f"<circle cx='{x:.2f}' cy='{y:.2f}' r='6' fill='{color}' stroke='black' stroke-width='1' />"
        )
    return "\n".join(elements) + "\n"


def svg_legend():
    legend_items = [
        (
            f"<ellipse cx='{LEGEND_X + 25:.2f}' cy='{LEGEND_Y + 20:.2f}' "
            "rx='16' ry='10' fill='#4c78a8' fill-opacity='0.2' "
            "stroke='#4c78a8' stroke-width='2' />"
        ),
        (
            f"<circle cx='{LEGEND_X + 25:.2f}' cy='{LEGEND_Y + 20:.2f}' "
            "r='6' fill='#4c78a8' stroke='black' stroke-width='1' />"
        ),
        (
            f"<text x='{LEGEND_X + 55:.2f}' y='{LEGEND_Y + 25:.2f}' "
            "font-size='22' font-family='Arial'>Centroid (dot)</text>"
        ),
        (
            f"<text x='{LEGEND_X + 55:.2f}' y='{LEGEND_Y + 55:.2f}' "
            "font-size='22' font-family='Arial'>95% confidence interval (area)</text>"
        ),
    ]
    return "\n".join(legend_items) + "\n"


def build_svg(stats):
    content = [
        svg_header(),
        svg_defs(),
        "<rect width='100%' height='100%' fill='white' />\n",
        svg_grid(),
        svg_axes(),
        svg_points(stats),
        svg_legend(),
        "</svg>\n",
    ]
    return "".join(content)


def main():
    statement_ids = load_statement_ids()
    placements = load_placements()
    stats = compute_stats(statement_ids, placements)
    svg = build_svg(stats)
    OUTPUT_PATH.write_text(svg, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
