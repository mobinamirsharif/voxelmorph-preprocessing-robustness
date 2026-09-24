#!/usr/bin/env python3
"""Plot the sanitized public-validation metrics from a JSON report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Metrics JSON")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG")
    return parser.parse_args()


def label_values(axis, bars, precision: int) -> None:
    for bar in bars:
        axis.annotate(
            f"{bar.get_height():.{precision}f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8,
        )


def main() -> None:
    args = parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    runs = report["runs"]
    labels = [run["scan"] for run in runs]
    x = np.arange(len(runs))
    colors = {"blue": "#3274A1", "orange": "#E1812C", "green": "#3A923A"}

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12})
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    figure.suptitle(
        "Public VoxelMorph validation on four CC0 OpenNeuro T1 scans",
        fontsize=16, fontweight="bold", y=0.985,
    )
    figure.text(
        0.5, 0.945,
        "Software/method validation only — not clinical or population validation",
        ha="center", fontsize=10, color="#555555",
    )

    mse = axes[0, 0].bar(x, [r["mse"] for r in runs], color=colors["blue"])
    axes[0, 0].set(title="A. Mean squared error", ylabel="MSE")
    label_values(axes[0, 0], mse, 4)

    width = 0.36
    corr = axes[0, 1].bar(x - width / 2, [r["correlation"] for r in runs], width, label="Correlation", color=colors["blue"])
    dice = axes[0, 1].bar(x + width / 2, [r["mask_dice"] for r in runs], width, label="Intensity-derived Mask Dice", color=colors["orange"])
    axes[0, 1].set(title="B. Similarity metrics", ylabel="Coefficient", ylim=(0, 0.85))
    axes[0, 1].legend(loc="lower right", fontsize=8)
    label_values(axes[0, 1], corr, 3); label_values(axes[0, 1], dice, 3)

    jac = axes[1, 0].bar(x, [r["jacobian_nonpositive_percent"] for r in runs], color=colors["orange"])
    axes[1, 0].set(title="C. Non-positive Jacobian determinants", ylabel="Voxels (%)")
    label_values(axes[1, 0], jac, 2)

    mean = axes[1, 1].bar(x - width / 2, [r["mean_displacement_voxels"] for r in runs], width, label="Mean", color=colors["green"])
    maximum = axes[1, 1].bar(x + width / 2, [r["max_displacement_voxels"] for r in runs], width, label="Maximum", color=colors["blue"])
    axes[1, 1].set(title="D. Displacement magnitude", ylabel="Voxels")
    axes[1, 1].legend(fontsize=8)
    label_values(axes[1, 1], mean, 2); label_values(axes[1, 1], maximum, 2)

    for axis in axes.flat:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    figure.subplots_adjust(left=0.08, right=0.98, top=0.89, bottom=0.10, hspace=0.38, wspace=0.20)
    figure.text(0.5, 0.025, "Mask Dice uses intensity-derived foreground masks, not segmentation ground truth.", ha="center", fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160, facecolor="white", metadata={"Software": "Matplotlib"})
    plt.close(figure)


if __name__ == "__main__":
    main()
