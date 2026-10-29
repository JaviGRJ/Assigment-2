"""Generate Assignment 2 SVG figures from summary.csv."""

from __future__ import annotations

import csv
import html
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read_summary() -> list[dict[str, str]]:
    """Read the benchmark summary."""
    path = ROOT / "summary.csv"

    if not path.is_file():
        raise FileNotFoundError(
            "summary.csv was not found. Run benchmark_assignment2.py first."
        )

    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def create_svg(
    rows: list[dict[str, str]],
    value_key: str,
    title: str,
    y_label: str,
    output_path: Path,
) -> None:
    """Create a labelled bar chart in SVG format."""
    width = 1400
    height = 750

    left = 90
    right = 40
    top = 70
    bottom = 220

    plot_width = width - left - right
    plot_height = height - top - bottom

    usable_rows = [
        row
        for row in rows
        if row.get(value_key, "") not in {"", None}
    ]

    if not usable_rows:
        raise ValueError(f"No values available for {value_key}")

    values = [
        float(row[value_key])
        for row in usable_rows
    ]

    maximum = max(values)

    if maximum <= 0.0:
        maximum = 1.0

    bar_width = plot_width / len(usable_rows)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
        ),
        (
            "<style>"
            "text{font-family:Arial,sans-serif;font-size:12px;fill:#17202a}"
            ".title{font-size:22px;font-weight:bold}"
            ".axis{stroke:#17202a;stroke-width:2}"
            ".grid{stroke:#d9e0e6;stroke-width:1}"
            ".bar{fill:#357edd}"
            ".bar-sparse{fill:#2ca25f}"
            "</style>"
        ),
        (
            f'<text x="{width / 2}" y="35" '
            f'text-anchor="middle" class="title">'
            f"{html.escape(title)}</text>"
        ),
        (
            f'<line x1="{left}" y1="{top}" '
            f'x2="{left}" y2="{top + plot_height}" class="axis"/>'
        ),
        (
            f'<line x1="{left}" y1="{top + plot_height}" '
            f'x2="{left + plot_width}" y2="{top + plot_height}" '
            f'class="axis"/>'
        ),
        (
            f'<text x="22" y="{top + plot_height / 2}" '
            f'transform="rotate(-90 22 {top + plot_height / 2})" '
            f'text-anchor="middle">{html.escape(y_label)}</text>'
        ),
        (
            f'<text x="{left + plot_width / 2}" '
            f'y="{height - 18}" text-anchor="middle">'
            "Configuration</text>"
        ),
    ]

    for grid_index in range(6):
        fraction = grid_index / 5
        y = top + plot_height - fraction * plot_height
        label_value = maximum * fraction

        lines.append(
            f'<line x1="{left}" y1="{y:.2f}" '
            f'x2="{left + plot_width}" y2="{y:.2f}" '
            f'class="grid"/>'
        )

        lines.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" '
            f'text-anchor="end">{label_value:.2f}</text>'
        )

    for index, row in enumerate(usable_rows):
        value = float(row[value_key])
        bar_height = value / maximum * plot_height

        x = left + index * bar_width + bar_width * 0.15
        y = top + plot_height - bar_height
        actual_width = bar_width * 0.7

        part = row.get("part", "")
        class_name = "bar-sparse" if part == "sparse" else "bar"

        label = (
            f'{row.get("method", "")} '
            f'n={row.get("size", "")} '
            f'd={row.get("density", "")}'
        )

        escaped_label = html.escape(label)

        lines.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" '
            f'width="{actual_width:.2f}" '
            f'height="{bar_height:.2f}" '
            f'class="{class_name}">'
            f'<title>{escaped_label}: {value:.6f}</title>'
            "</rect>"
        )

        label_x = x + actual_width / 2
        label_y = top + plot_height + 18

        lines.append(
            f'<text x="{label_x:.2f}" y="{label_y}" '
            f'transform="rotate(65 {label_x:.2f} {label_y})" '
            f'text-anchor="start">{escaped_label}</text>'
        )

    legend_y = 55

    lines.extend(
        [
            f'<rect x="{width - 250}" y="{legend_y - 12}" '
            'width="15" height="15" class="bar"/>',
            f'<text x="{width - 228}" y="{legend_y}" '
            ">Dense configuration</text>",
            f'<rect x="{width - 250}" y="{legend_y + 15}" '
            'width="15" height="15" class="bar-sparse"/>',
            f'<text x="{width - 228}" y="{legend_y + 27}" '
            ">Sparse configuration</text>",
            "</svg>",
        ]
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    rows = read_summary()

    create_svg(
        rows,
        value_key="median_ms",
        title="Assignment 2 runtime",
        y_label="Median time (ms)",
        output_path=ROOT / "runtime.svg",
    )

    create_svg(
        rows,
        value_key="peak_memory_mib",
        title="Assignment 2 memory",
        y_label="Peak traced memory (MiB)",
        output_path=ROOT / "memory.svg",
    )

    print("Created runtime.svg and memory.svg.")


if __name__ == "__main__":
    main()