#!/usr/bin/env python3
"""Render all_attempts.tsv as a compact, exact dark-mode PNG report."""

from __future__ import annotations

import csv
import textwrap
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "all_attempts.tsv"
OUTPUT = ROOT.parents[1] / "assets" / "results" / "all_attempts_report.png"
IST = ZoneInfo("Asia/Kolkata")


def wrapped_description(text: str) -> str:
    return "\n".join(textwrap.wrap(text, width=74, break_long_words=False))


with SOURCE.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))

height = max(6.4, 2.4 + 0.88 * len(rows))
fig, ax = plt.subplots(figsize=(19.2, height), dpi=120)
fig.patch.set_facecolor("#0d0e0f")
ax.set_facecolor("#0d0e0f")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

card = FancyBboxPatch(
    (0.025, 0.045),
    0.95,
    0.82,
    boxstyle="round,pad=0.012,rounding_size=0.018",
    facecolor="#191a1a",
    edgecolor="#363737",
    linewidth=1.0,
)
ax.add_patch(card)

ax.text(
    0.04,
    0.94,
    "LEDGAR autoresearch — all attempts",
    color="#f0efec",
    fontsize=24,
    fontweight="semibold",
    va="center",
)
ax.text(
    0.04,
    0.895,
    f"{len(rows)} completed experiments · times shown in IST · higher macro-F1 is better",
    color="#969693",
    fontsize=12,
    va="center",
)

columns = [
    (0.047, "#"),
    (0.085, "DATE (IST)"),
    (0.205, "MACRO_F1"),
    (0.290, "ACCURACY"),
    (0.375, "UNPARSED"),
    (0.465, "DECISION"),
    (0.575, "DESCRIPTION"),
]
header_y = 0.825
for x, label in columns:
    ax.text(x, header_y, label, color="#8f8e8b", fontsize=11, fontweight="bold", va="center")

table_top = 0.785
table_bottom = 0.075
row_h = (table_top - table_bottom) / max(len(rows), 1)
line_color = "#303131"

for index, row in enumerate(rows, start=1):
    y_top = table_top - (index - 1) * row_h
    y = y_top - row_h / 2
    ax.plot([0.045, 0.955], [y_top, y_top], color=line_color, linewidth=0.8)

    stamp = datetime.fromtimestamp(int(row["timestamp"]), tz=IST).strftime("%d Jul %H:%M")
    values = [
        (0.047, str(index), "#e7e6e2"),
        (0.085, stamp, "#bdbcb8"),
        (0.205, row["macro_f1"], "#f1f0ed"),
        (0.290, row["accuracy"], "#f1f0ed"),
        (0.375, row["unparsed"].removeprefix("unparsed="), "#bdbcb8"),
    ]
    for x, value, color in values:
        ax.text(x, y, value, color=color, fontsize=12, va="center")

    decision = row["decision"].lower()
    chip_colors = {
        "kept": ("#302d45", "#aaa0e7"),
        "discarded": ("#442c2b", "#e48882"),
        "baseline": ("#363634", "#c2c0ba"),
        "pending": ("#403a28", "#d5bd75"),
    }
    chip_bg, chip_fg = chip_colors.get(decision, chip_colors["pending"])
    chip_width = 0.078 if decision != "discarded" else 0.092
    chip = FancyBboxPatch(
        (0.462, y - 0.017),
        chip_width,
        0.034,
        boxstyle="round,pad=0.004,rounding_size=0.014",
        facecolor=chip_bg,
        edgecolor="none",
    )
    ax.add_patch(chip)
    ax.text(
        0.462 + chip_width / 2,
        y,
        decision,
        color=chip_fg,
        fontsize=10.5,
        fontweight="bold",
        ha="center",
        va="center",
    )
    ax.text(
        0.575,
        y,
        wrapped_description(row["description"]),
        color="#e6e5e1",
        fontsize=10.5,
        va="center",
    )

ax.plot([0.045, 0.955], [table_bottom, table_bottom], color=line_color, linewidth=0.8)
fig.savefig(OUTPUT, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
print(OUTPUT)
