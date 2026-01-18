import csv
import math
import os
import struct
import zlib

PLACEMENT_PATH = "Placement.csv"
OUTPUT_DIR = "outputs"

DECISIVE_IDS = {"ad1b9e2e-51bb-4e90-a9ad-816022f82dc4"}
MODERATE_IDS = {"35569596-4e4f-4ef8-ba25-8e0b36d7c77c"}

GRID_SIZE = 7

# Simple 5x7 bitmap font for digits, dot, percent, minus, and space.
FONT_5X7 = {
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "10000", "10000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
    ".": ["00000", "00000", "00000", "00000", "00000", "00110", "00110"],
    "%": ["11001", "11010", "00100", "01000", "10110", "00110", "00000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
}


def read_placements():
    placements = []
    with open(PLACEMENT_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = int(row["x"])
                y = int(row["y"])
            except (ValueError, TypeError):
                continue
            if not (1 <= x <= GRID_SIZE and 1 <= y <= GRID_SIZE):
                continue
            placements.append({
                "participant_id": row["token"],
                "x": x,
                "y": y,
            })
    return placements


def aggregate_group(placements, participant_ids):
    participant_cells = {}
    for placement in placements:
        pid = placement["participant_id"]
        if pid not in participant_ids:
            continue
        participant_cells.setdefault(pid, [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)])
        participant_cells[pid][placement["y"] - 1][placement["x"] - 1] += 1

    participant_props = []
    raw_counts = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    total_placements = 0
    for pid, counts in participant_cells.items():
        participant_total = sum(sum(row) for row in counts)
        total_placements += participant_total
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                raw_counts[y][x] += counts[y][x]
        if participant_total == 0:
            continue
        props = [[counts[y][x] / participant_total for x in range(GRID_SIZE)] for y in range(GRID_SIZE)]
        participant_props.append(props)

    mean_props = [[0.0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    if participant_props:
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                mean_props[y][x] = sum(p[y][x] for p in participant_props) / len(participant_props)

    return {
        "mean_props": mean_props,
        "raw_counts": raw_counts,
        "participants": sorted(participant_cells.keys()),
        "total_placements": total_placements,
    }


def lerp(a, b, t):
    return int(a + (b - a) * t)


def color_scale(value, vmin, vmax):
    if vmax == vmin:
        t = 0.0
    else:
        t = (value - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    # White to blue
    return (
        lerp(255, 31, t),
        lerp(255, 119, t),
        lerp(255, 180, t),
    )


def diverging_scale(value, vmin, vmax):
    if vmax == vmin:
        t = 0.5
    else:
        t = (value - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    # Blue -> white -> red
    if t < 0.5:
        t2 = t / 0.5
        return (
            lerp(33, 255, t2),
            lerp(113, 255, t2),
            lerp(181, 255, t2),
        )
    t2 = (t - 0.5) / 0.5
    return (
        lerp(255, 178, t2),
        lerp(255, 34, t2),
        lerp(255, 34, t2),
    )


def draw_text(img, width, height, x, y, text, color=(0, 0, 0), scale=2):
    cursor_x = x
    for ch in text:
        glyph = FONT_5X7.get(ch, FONT_5X7[" "])
        for row_idx, row in enumerate(glyph):
            for col_idx, bit in enumerate(row):
                if bit == "1":
                    for sy in range(scale):
                        for sx in range(scale):
                            px = cursor_x + col_idx * scale + sx
                            py = y + row_idx * scale + sy
                            if 0 <= px < width and 0 <= py < height:
                                idx = (py * width + px) * 3
                                img[idx:idx+3] = bytes(color)
        cursor_x += (5 + 1) * scale


def _svg_color(rgb):
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _svg_text(text):
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def draw_heatmap_svg(mean_props, title, subtitle, outfile, vmin, vmax, diverging=False):
    cell_size = 80
    margin_left = 140
    margin_top = 120
    margin_right = 60
    margin_bottom = 120

    width = margin_left + GRID_SIZE * cell_size + margin_right
    height = margin_top + GRID_SIZE * cell_size + margin_bottom

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" fill="white" />',
    ]

    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            value = mean_props[y][x]
            color = diverging_scale(value, vmin, vmax) if diverging else color_scale(value, vmin, vmax)
            x0 = margin_left + x * cell_size
            y0 = margin_top + (GRID_SIZE - 1 - y) * cell_size
            svg.append(
                f'<rect x="{x0}" y="{y0}" width="{cell_size}" height="{cell_size}" fill="{_svg_color(color)}" />'
            )

    line_color = "#c8c8c8"
    for i in range(GRID_SIZE + 1):
        x = margin_left + i * cell_size
        y = margin_top + i * cell_size
        svg.append(f'<line x1="{x}" y1="{margin_top}" x2="{x}" y2="{margin_top + GRID_SIZE * cell_size}" stroke="{line_color}" />')
        svg.append(f'<line x1="{margin_left}" y1="{y}" x2="{margin_left + GRID_SIZE * cell_size}" y2="{y}" stroke="{line_color}" />')

    for x in range(1, GRID_SIZE + 1):
        label_x = margin_left + (x - 0.5) * cell_size
        label_y = margin_top + GRID_SIZE * cell_size + 30
        svg.append(
            f'<text x="{label_x}" y="{label_y}" text-anchor="middle" font-size="16" font-family="Arial">{x}</text>'
        )
    for y in range(1, GRID_SIZE + 1):
        label_x = margin_left - 20
        label_y = margin_top + (GRID_SIZE - y + 0.5) * cell_size + 6
        svg.append(
            f'<text x="{label_x}" y="{label_y}" text-anchor="end" font-size="16" font-family="Arial">{y}</text>'
        )

    svg.append(
        f'<text x="20" y="{margin_top - 60}" font-size="22" font-family="Arial">{_svg_text(title)}</text>'
    )
    svg.append(
        f'<text x="20" y="{margin_top - 30}" font-size="16" font-family="Arial" fill="#505050">{_svg_text(subtitle)}</text>'
    )
    svg.append(
        f'<text x="{margin_left + 120}" y="{height - 60}" font-size="16" font-family="Arial">Implementability</text>'
    )
    svg.append(
        f'<text x="20" y="{margin_top + 200}" font-size="16" font-family="Arial">Impact</text>'
    )

    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            value = mean_props[y][x] * 100
            label = f"{value:.1f}%"
            x0 = margin_left + x * cell_size + cell_size / 2
            y0 = margin_top + (GRID_SIZE - 1 - y) * cell_size + cell_size / 2 + 6
            color = "#000000" if value < 12 else "#ffffff"
            svg.append(
                f'<text x="{x0}" y="{y0}" text-anchor="middle" font-size="14" font-family="Arial" fill="{color}">{label}</text>'
            )

    svg.append("</svg>")
    with open(outfile, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))


def draw_placements_svg(placements, title, subtitle, outfile):
    cell_size = 80
    margin_left = 140
    margin_top = 120
    margin_right = 60
    margin_bottom = 120

    width = margin_left + GRID_SIZE * cell_size + margin_right
    height = margin_top + GRID_SIZE * cell_size + margin_bottom

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" fill="white" />',
    ]

    line_color = "#b4b4b4"
    for i in range(GRID_SIZE + 1):
        x = margin_left + i * cell_size
        y = margin_top + i * cell_size
        svg.append(f'<line x1="{x}" y1="{margin_top}" x2="{x}" y2="{margin_top + GRID_SIZE * cell_size}" stroke="{line_color}" />')
        svg.append(f'<line x1="{margin_left}" y1="{y}" x2="{margin_left + GRID_SIZE * cell_size}" y2="{y}" stroke="{line_color}" />')

    for x in range(1, GRID_SIZE + 1):
        label_x = margin_left + (x - 0.5) * cell_size
        label_y = margin_top + GRID_SIZE * cell_size + 30
        svg.append(
            f'<text x="{label_x}" y="{label_y}" text-anchor="middle" font-size="16" font-family="Arial">{x}</text>'
        )
    for y in range(1, GRID_SIZE + 1):
        label_x = margin_left - 20
        label_y = margin_top + (GRID_SIZE - y + 0.5) * cell_size + 6
        svg.append(
            f'<text x="{label_x}" y="{label_y}" text-anchor="end" font-size="16" font-family="Arial">{y}</text>'
        )

    svg.append(
        f'<text x="20" y="{margin_top - 60}" font-size="22" font-family="Arial">{_svg_text(title)}</text>'
    )
    svg.append(
        f'<text x="20" y="{margin_top - 30}" font-size="16" font-family="Arial" fill="#505050">{_svg_text(subtitle)}</text>'
    )
    svg.append(
        f'<text x="{margin_left + 80}" y="{height - 60}" font-size="16" font-family="Arial">Implementability (1 = Hard, 7 = Easy)</text>'
    )
    svg.append(
        f'<text x="20" y="{margin_top + 200}" font-size="16" font-family="Arial">Degree of impact (1 = Low, 7 = High)</text>'
    )

    point_color = "#282828"
    point_size = 12
    for placement in placements:
        x = placement["x"]
        y = placement["y"]
        center_x = margin_left + (x - 0.5) * cell_size
        center_y = margin_top + (GRID_SIZE - y + 0.5) * cell_size
        svg.append(
            f'<rect x="{center_x - point_size / 2}" y="{center_y - point_size / 2}" '
            f'width="{point_size}" height="{point_size}" fill="{point_color}" />'
        )

    svg.append("</svg>")
    with open(outfile, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))


def save_png(path, img, width, height):
    def chunk(tag, data):
        return struct.pack(">I", len(data)
        ) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = b""
    stride = width * 3
    for y in range(height):
        raw += b"\x00" + bytes(img[y * stride:(y + 1) * stride])
    compressed = zlib.compress(raw, level=9)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", compressed))
        f.write(chunk(b"IEND", b""))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    placements = read_placements()

    decisive = aggregate_group(placements, DECISIVE_IDS)
    moderate = aggregate_group(placements, MODERATE_IDS)
    decisive_placements = [p for p in placements if p["participant_id"] in DECISIVE_IDS]
    moderate_placements = [p for p in placements if p["participant_id"] in MODERATE_IDS]

    all_values = [
        value for row in decisive["mean_props"] for value in row
    ] + [
        value for row in moderate["mean_props"] for value in row
    ]
    vmin = min(all_values) if all_values else 0.0
    vmax = max(all_values) if all_values else 1.0

    decisive_subtitle = f"n={len(decisive['participants'])} participants, {decisive['total_placements']} placements"
    moderate_subtitle = f"n={len(moderate['participants'])} participants, {moderate['total_placements']} placements"

    draw_heatmap_svg(
        decisive["mean_props"],
        "Decisive mapping style",
        decisive_subtitle,
        os.path.join(OUTPUT_DIR, "heatmap_decisive.svg"),
        vmin,
        vmax,
    )

    draw_heatmap_svg(
        moderate["mean_props"],
        "Moderate mapping style",
        moderate_subtitle,
        os.path.join(OUTPUT_DIR, "heatmap_moderate.svg"),
        vmin,
        vmax,
    )

    diff = [[decisive["mean_props"][y][x] - moderate["mean_props"][y][x] for x in range(GRID_SIZE)] for y in range(GRID_SIZE)]
    diff_values = [value for row in diff for value in row]
    diff_min = min(diff_values) if diff_values else -0.1
    diff_max = max(diff_values) if diff_values else 0.1
    # Symmetric bounds around zero
    bound = max(abs(diff_min), abs(diff_max))
    draw_heatmap_svg(
        diff,
        "Decisive minus Moderate",
        "Difference in mean proportion",
        os.path.join(OUTPUT_DIR, "heatmap_diff.svg"),
        -bound,
        bound,
        diverging=True,
    )

    draw_placements_svg(
        decisive_placements,
        "Decisive mapper placements",
        f"{len(decisive_placements)} placements",
        os.path.join(OUTPUT_DIR, "placements_decisive.svg"),
    )

    draw_placements_svg(
        moderate_placements,
        "Moderate mapper placements",
        f"{len(moderate_placements)} placements",
        os.path.join(OUTPUT_DIR, "placements_moderate.svg"),
    )


if __name__ == "__main__":
    main()
