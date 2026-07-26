#!/usr/bin/env python3
"""Build the reproducible NemoCounsel article and its data-derived visuals."""

from __future__ import annotations

import base64
import csv
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import markdown
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
ASSETS = OUT / "assets"
DATA = OUT / "data"

INK = "#263238"
MUTED = "#60747b"
TEAL = "#6f9e9a"
TEAL_DARK = "#3f7773"
SAND = "#d8c7a3"
ROSE = "#c99591"
BG = "#f7f5f0"
GRID = "#dfe5e3"


def setup() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "axes.edgecolor": GRID,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.titleweight": "bold",
        }
    )


def load_attempts() -> list[dict[str, str]]:
    with (ROOT / "all_attempts.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    # The public report is the Nemotron research series only. Preserve original
    # IDs in source_iteration so every number remains traceable to the raw log.
    for source_iteration, row in enumerate(rows, start=1):
        row.setdefault("iteration", str(source_iteration))
        if not row.get("timestamp_ist"):
            timestamp = datetime.fromtimestamp(
                int(row["timestamp"]), tz=ZoneInfo("Asia/Kolkata")
            )
            row["timestamp_ist"] = timestamp.strftime("%Y-%m-%d %H:%M:%S IST")
        if row["unparsed"].startswith("unparsed="):
            row["unparsed"] = row["unparsed"].removeprefix("unparsed=")
    rows = rows[1:]
    for public_id, row in enumerate(rows, start=1):
        row["experiment"] = str(public_id)
        row["source_iteration"] = row["iteration"]
    return rows


def write_attempts(rows: list[dict[str, str]]) -> None:
    fields = [
        "experiment",
        "source_iteration",
        "timestamp_ist",
        "macro_f1",
        "accuracy",
        "unparsed",
        "decision",
        "description",
    ]
    with (DATA / "nemotron_experiments.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def extract_history() -> list[tuple[float, float, float]]:
    raw_log = ROOT / "last_run.log"
    checked_in_history = DATA / "final_training_history.tsv"
    if raw_log.exists():
        text = raw_log.read_text(errors="replace")
        pattern = re.compile(
            r"\{'loss': '([^']+)',.*?'learning_rate': '([^']+)', 'epoch': '([^']+)'\}"
        )
        history = [
            (float(epoch), float(loss), float(lr))
            for loss, lr, epoch in pattern.findall(text)
        ]
    elif checked_in_history.exists():
        with checked_in_history.open() as handle:
            rows = csv.DictReader(handle, delimiter="\t")
            history = [
                (float(row["epoch"]), float(row["loss"]), float(row["learning_rate"]))
                for row in rows
            ]
    else:
        raise RuntimeError(
            "Neither last_run.log nor data/final_training_history.tsv is available"
        )
    with (DATA / "final_training_history.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["epoch", "loss", "learning_rate"])
        writer.writerows(history)
    return history


def style_axis(ax) -> None:
    ax.set_facecolor(BG)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(length=0)


def plot_progress(rows: list[dict[str, str]]) -> None:
    x = np.arange(1, len(rows) + 1)
    f1 = np.array([float(row["macro_f1"]) for row in rows])
    kept = [row["decision"] == "kept" for row in rows]
    colors = [TEAL if value else ROSE for value in kept]

    fig, ax = plt.subplots(figsize=(11, 5.8), facecolor=BG)
    style_axis(ax)
    ax.plot(x, f1, color=SAND, linewidth=2.4, zorder=1)
    ax.scatter(x, f1, s=125, color=colors, edgecolor="white", linewidth=1.6, zorder=2)
    for xi, score in zip(x, f1):
        ax.text(xi, score + 0.027, f"{score:.4f}", ha="center", fontsize=10, weight="bold")
    ax.set_xticks(x, [f"E{i}" for i in x])
    ax.set_ylim(0.22, 0.90)
    ax.set_ylabel("Validation Macro-F1")
    ax.set_title(
        "Six Nemotron experiments: one hypothesis at a time",
        loc="left",
        fontsize=18,
        pad=16,
    )
    ax.scatter([], [], color=ROSE, label="Discarded")
    ax.scatter([], [], color=TEAL, label="Retained as new best")
    ax.legend(frameon=False, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(ASSETS / "nemotron_progress.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_loss(history: list[tuple[float, float, float]]) -> None:
    epochs = np.array([row[0] for row in history])
    loss = np.array([row[1] for row in history])
    window = 21
    smooth = np.convolve(loss, np.ones(window) / window, mode="valid")
    sx = epochs[window - 1 :]

    fig, ax = plt.subplots(figsize=(11, 5.8), facecolor=BG)
    style_axis(ax)
    ax.plot(epochs, loss, color="#b6cbc8", linewidth=0.9, alpha=0.55, label="Logged loss")
    ax.plot(sx, smooth, color=TEAL_DARK, linewidth=2.5, label=f"{window}-point moving average")
    ax.axvline(1.0, color=SAND, linestyle="--", linewidth=1.3)
    ax.text(1.01, max(smooth) * 0.78, "Epoch 2", color=MUTED, fontsize=9)
    ax.set_xlim(0, 2)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title("Final run training loss", loc="left", fontsize=18)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(ASSETS / "final_training_loss.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_config() -> None:
    fig, ax = plt.subplots(figsize=(11, 5.8), facecolor=BG)
    ax.axis("off")
    ax.text(0.04, 0.91, "Final reproducible configuration", fontsize=20, weight="bold", color=INK)
    cards = [
        ("MODEL", "Llama-3.1-Nemotron-Nano-8B-v1"),
        ("DATA", "60,000 train · 200 validation · LEDGAR"),
        ("QUANTIZATION", "4-bit NF4 · bfloat16 compute"),
        ("LORA", "r=32 · α=64 · dropout=0.05"),
        ("TARGETS", "q/k/v/o + gate/up/down projections"),
        ("OPTIMIZATION", "2 epochs · LR 2e-4 · cosine · warmup 0.1"),
        ("BATCHING", "4 × 4 accumulation = effective batch 16"),
        ("RESULT", "Macro-F1 0.8246 · accuracy 0.9150"),
    ]
    for i, (label, value) in enumerate(cards):
        col, row = i % 2, i // 2
        x, y = 0.04 + col * 0.48, 0.73 - row * 0.19
        ax.add_patch(
            plt.Rectangle((x, y), 0.44, 0.145, facecolor="white", edgecolor=GRID, linewidth=1)
        )
        ax.text(x + 0.025, y + 0.095, label, fontsize=8.5, color=TEAL_DARK, weight="bold")
        ax.text(x + 0.025, y + 0.045, value, fontsize=10.2, color=INK)
    fig.savefig(ASSETS / "final_configuration.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_html() -> None:
    md_text = (OUT / "nemocounsel-autoresearch.md").read_text()
    body = markdown.markdown(md_text, extensions=["fenced_code", "tables"])
    for image in ASSETS.glob("*.png"):
        encoded = base64.b64encode(image.read_bytes()).decode()
        body = body.replace(f'assets/{image.name}', f"data:image/png;base64,{encoded}")
    css = """
    :root{--ink:#263238;--muted:#60747b;--teal:#3f7773;--bg:#f7f5f0}
    body{margin:0;background:var(--bg);color:var(--ink);font:18px/1.72 Georgia,serif}
    main{max-width:860px;margin:0 auto;padding:64px 28px 100px}
    h1,h2,h3{font-family:Arial,sans-serif;line-height:1.18} h1{font-size:48px}
    h2{margin-top:2.2em;font-size:30px} h3{margin-top:1.8em}
    a{color:var(--teal)} img{width:100%;height:auto;margin:24px 0}
    blockquote{border-left:4px solid #6f9e9a;margin:2em 0;padding:4px 24px;color:var(--muted)}
    code{font:14px/1.5 Menlo,monospace;background:#ebece8;padding:2px 5px;border-radius:4px}
    pre{overflow:auto;background:#263238;color:#eef3f1;padding:20px;border-radius:8px}
    pre code{background:none;padding:0} table{border-collapse:collapse;width:100%;font:15px Arial,sans-serif}
    th,td{text-align:left;padding:10px;border-bottom:1px solid #dfe5e3} th{color:var(--teal)}
    """
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>NemoCounsel Autoresearch</title><style>"
        + css
        + "</style></head><body><main>"
        + body
        + "</main></body></html>"
    )
    (OUT / "nemocounsel-autoresearch.html").write_text(html)


def main() -> None:
    setup()
    attempts = load_attempts()
    if len(attempts) != 6:
        raise RuntimeError(f"Expected six Nemotron experiments, found {len(attempts)}")
    write_attempts(attempts)
    history = extract_history()
    if not history:
        raise RuntimeError("No final-run training-history records found")
    plot_progress(attempts)
    plot_loss(history)
    plot_config()
    build_html()
    print(f"Built report with {len(attempts)} experiments and {len(history)} loss records")


if __name__ == "__main__":
    main()
