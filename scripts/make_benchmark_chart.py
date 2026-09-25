"""Draws the README benchmark chart: Tacet Sonata against Laya, one small panel per measure.

Writes assets/benchmark-light.png and assets/benchmark-dark.png. Every number comes from the
release report; each panel has its own axis starting at zero, because the measures do not share
a scale.

    python scripts/make_benchmark_chart.py [--fonts path/to/inter/ttf/folder]
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIRECTORY = os.path.join(REPOSITORY_ROOT, "assets")

MODELS = ["Tacet Sonata", "Laya"]
PANELS = [
    {"title": "Requests per second", "note": "higher is better", "values": [211.0, 14.9], "format": "{:g}", "precise": "{:g}"},
    {"title": "Latency per request, ms", "note": "lower is better", "values": [32.5, 62.4], "format": "{:.1f}", "precise": "{:.1f}"},
    {"title": "Parameters, millions", "note": "smaller is lighter", "values": [144.0, 421.0], "format": "{:.0f}M", "precise": "{:.0f}M"},
    {"title": "Accuracy", "note": "a tie, p = 0.63", "values": [0.7625, 0.7675], "format": "{:.4f}", "precise": "{:.4f}", "axis_max": 1.0},
    {"title": "Calibration error (ECE)", "note": "lower is better", "values": [0.095, 0.215], "format": "{:.3f}", "precise": "{:.3f}"},
]
FOOTNOTE = ("Accuracy and calibration: LocalLLaMA typed decisions test split, 2,000 decisions, same script for both. "
            "Speed: one NVIDIA L4, five question requests, measured on the same 144M architecture.")

THEMES = {
    "light": {"surface": "#ffffff", "ink": "#171717", "muted": "#666666", "track": "#ececec",
              "series": ["#4f7fe0", "#b07f3e"]},
    "dark": {"surface": "#0d1117", "ink": "#ececec", "muted": "#9aa0a8", "track": "#1f2630",
             "series": ["#5a86ea", "#b3843f"]},
}


def register_fonts(font_directory):
    if not font_directory or not os.path.isdir(font_directory):
        return "DejaVu Sans"
    for file_name in os.listdir(font_directory):
        if file_name.lower().endswith(".ttf"):
            font_manager.fontManager.addfont(os.path.join(font_directory, file_name))
    return "Inter"


def horizontal_bar(axis, left, bottom, width, height, color):
    """A horizontal bar with square ends, so every panel looks alike whatever its data scale."""
    if width <= 0:
        return
    axis.add_patch(Rectangle((left, bottom), width, height, linewidth=0, facecolor=color))


def draw_panel(axis, panel, theme):
    axis_max = panel.get("axis_max", max(panel["values"]) * 1.35)
    bar_height = 0.34
    for row, (model, value) in enumerate(zip(MODELS, panel["values"])):
        center = 1 - row
        horizontal_bar(axis, 0, center - bar_height / 2, axis_max, bar_height, theme["track"])
        horizontal_bar(axis, 0, center - bar_height / 2, value, bar_height, theme["series"][row])
        axis.text(0, center + bar_height / 2 + 0.08, model, color=theme["muted"], fontsize=10.5, va="bottom")
        axis.text(axis_max, center + bar_height / 2 + 0.08, panel["precise"].format(value), color=theme["ink"],
                  fontsize=11.5, fontweight="semibold", va="bottom", ha="right")
    axis.set_xlim(0, axis_max)
    axis.set_ylim(-0.55, 1.85)
    axis.axis("off")
    axis.set_title(panel["title"], loc="left", color=theme["ink"], fontsize=13, fontweight="semibold", pad=4)
    axis.text(0, -0.5, panel["note"], color=theme["muted"], fontsize=10, va="bottom")


def draw_chart(theme_name, font_family):
    theme = THEMES[theme_name]
    plt.rcParams["font.family"] = font_family
    figure, axes = plt.subplots(2, 3, figsize=(12, 6.2), dpi=150)
    figure.patch.set_facecolor(theme["surface"])
    flat_axes = [axis for row in axes for axis in row]
    for axis in flat_axes:
        axis.set_facecolor(theme["surface"])
    for axis, panel in zip(flat_axes, PANELS):
        draw_panel(axis, panel, theme)
    summary_axis = flat_axes[-1]
    summary_axis.axis("off")
    summary_axis.text(0.02, 0.80, "Tacet Sonata vs Laya", color=theme["ink"], fontsize=15, fontweight="bold",
                      transform=summary_axis.transAxes, va="top")
    summary_axis.text(0.02, 0.58, "Same accuracy at a third of the size,\n14x the requests, better calibrated.",
                      color=theme["muted"], fontsize=11, transform=summary_axis.transAxes, linespacing=1.6, va="top")
    figure.text(0.02, 0.02, FOOTNOTE, color=theme["muted"], fontsize=8.5, wrap=True)
    figure.subplots_adjust(left=0.03, right=0.97, top=0.9, bottom=0.12, wspace=0.28, hspace=0.55)
    output_path = os.path.join(OUTPUT_DIRECTORY, f"benchmark-{theme_name}.png")
    figure.savefig(output_path, facecolor=theme["surface"])
    plt.close(figure)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fonts", default=None, help="folder with Inter .ttf files (optional)")
    arguments = parser.parse_args()
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    font_family = register_fonts(arguments.fonts)
    for theme_name in THEMES:
        print("wrote", draw_chart(theme_name, font_family))


if __name__ == "__main__":
    main()
