import csv
import math
from pathlib import Path

OUTPUT_PATH = Path(
    "outputs/two_profiles_decisive_vs_moderate_bubble_heatmaps.svg"
)

PARTICIPANTS = {
    "ad1b9e2e-51bb-4e90-a9ad-816022f82dc4": "Decisive mapper placements",
    "35569596-4e4f-4ef8-ba25-8e0b36d7c77c": "Moderate mapper placements",
}


def load_counts():
    counts = {pid: [[0 for _ in range(7)] for _ in range(7)] for pid in PARTICIPANTS}
    totals = {pid: 0 for pid in PARTICIPANTS}

    with open("Placement.csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            token = row.get("token")
            if token not in PARTICIPANTS:
                continue
            try:
                x = int(row.get("x") or 0)
                y = int(row.get("y") or 0)
            except ValueError:
                continue
            if not (1 <= x <= 7 and 1 <= y <= 7):
                continue
            counts[token][y - 1][x - 1] += 1
            totals[token] += 1
    return counts, totals


def svg_header(width, height):
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        '<defs>',
        '<marker id="axis-arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="strokeWidth">',
        '<path d="M0,0 L12,6 L0,12 Z" fill="#000" />',
        '</marker>',
        "</defs>",
        '<style>text { font-family: "Arial", "Helvetica", sans-serif; fill: #000; }</style>',
    ]


def svg_footer():
    return ["</svg>"]


def add_text(lines, x, y, text, size, weight="normal", rotate=None):
    safe = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate else ""
    lines.append(
        f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}"{transform}>{safe}</text>'
    )


def add_line(lines, x1, y1, x2, y2, width=1, marker_end=False):
    marker = ' marker-end="url(#axis-arrow)"' if marker_end else ""
    lines.append(
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#000" stroke-width="{width}"{marker} />'
    )


def add_rect(lines, x, y, w, h, width=1):
    lines.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="#000" stroke-width="{width}" />'
    )


def add_circle(lines, cx, cy, r, width=2):
    lines.append(
        f'<circle cx="{cx}" cy="{cy}" r="{r:.2f}" fill="none" stroke="#000" stroke-width="{width}" />'
    )


def draw_axes(
    lines,
    grid_left,
    grid_top,
    grid_right,
    grid_bottom,
    cell_size,
    axis_label_x,
    axis_label_y,
):
    axis_offset = 8
    axis_width = 5
    x_axis_y = grid_bottom + axis_offset
    y_axis_x = grid_left - axis_offset

    add_line(
        lines,
        y_axis_x,
        grid_bottom,
        y_axis_x,
        grid_top - 30,
        width=axis_width,
        marker_end=True,
    )
    add_line(
        lines,
        grid_left,
        x_axis_y,
        grid_right + 30,
        x_axis_y,
        width=axis_width,
        marker_end=True,
    )

    for i in range(7):
        value = i + 1
        x_center = grid_left + (i + 0.5) * cell_size
        y_center = grid_bottom - (i + 0.5) * cell_size
        add_text(lines, x_center - 4, x_axis_y + 28, str(value), size=16)
        add_text(lines, y_axis_x - 26, y_center + 6, str(value), size=16)

    add_text(
        lines,
        (grid_left + grid_right) / 2 - 160,
        x_axis_y + 70,
        axis_label_x,
        size=18,
    )
    add_text(
        lines,
        grid_left - 120,
        (grid_top + grid_bottom) / 2 + 80,
        axis_label_y,
        size=18,
        rotate=-90,
    )


def main():
    counts, totals = load_counts()
    cmax = max(max(max(row) for row in grid) for grid in counts.values())
    cmax = max(cmax, 1)

    width = 1500
    height = 700
    cell_size = 60
    grid_size = cell_size * 7
    grid_top = 150
    grid_lefts = [120, 120 + grid_size + 140]
    grid_bottom = grid_top + grid_size

    r_min = 6
    r_max = 26

    lines = svg_header(width, height)

    titles = list(PARTICIPANTS.values())
    participant_ids = list(PARTICIPANTS.keys())

    for idx, pid in enumerate(participant_ids):
        grid_left = grid_lefts[idx]
        grid_right = grid_left + grid_size

        title_y = 50
        add_text(lines, grid_left, title_y, titles[idx], size=22, weight="bold")
        add_text(
            lines,
            grid_left,
            title_y + 24,
            f"Total placements: {totals[pid]}",
            size=16,
        )

        draw_axes(
            lines,
            grid_left,
            grid_top,
            grid_right,
            grid_bottom,
            cell_size,
            "Implementability (1 = Hard, 7 = Easy)",
            "Degree of impact (1 = Low, 7 = High)",
        )

        add_rect(lines, grid_left, grid_top, grid_size, grid_size, width=3)
        for i in range(1, 7):
            x = grid_left + i * cell_size
            add_line(lines, x, grid_top, x, grid_bottom, width=1)
            y = grid_top + i * cell_size
            add_line(lines, grid_left, y, grid_right, y, width=1)

        grid = counts[pid]
        for y_idx in range(7):
            for x_idx in range(7):
                count = grid[y_idx][x_idx]
                if count <= 0:
                    continue
                radius = r_min + (r_max - r_min) * math.sqrt(count / cmax)
                cx = grid_left + (x_idx + 0.5) * cell_size
                cy = grid_bottom - (y_idx + 0.5) * cell_size
                add_circle(lines, cx, cy, radius, width=2)

    lines.extend(svg_footer())
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
