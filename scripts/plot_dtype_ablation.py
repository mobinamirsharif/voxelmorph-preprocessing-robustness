#!/usr/bin/env python3
"""Plot registration metrics from a dtype-ablation JSON report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(args.input_json.read_text(encoding="utf-8"))
    stages = report["stages"]
    order = [
        key
        for key in (
            "int16_rigid",
            "int16_affine",
            "float32_rigid",
            "float32_affine",
            "float32_affine_voxelmorph",
        )
        if key in stages
    ]
    label_map = {
        "int16_rigid": "int16\nrigid",
        "int16_affine": "int16\naffine",
        "float32_rigid": "float32\nrigid",
        "float32_affine": "float32\naffine",
        "float32_affine_voxelmorph": "float32\naffine + VXM",
    }
    labels = [label_map[key] for key in order]
    colors = ["#c0392b", "#e67e22", "#2471a3", "#17a589", "#6c3483"][: len(order)]
    x = np.arange(len(order))

    values = {
        "Foreground-mask Dice": [stages[key]["mask_dice"] for key in order],
        "Union correlation": [stages[key]["correlation"] for key in order],
        "Nonzero fraction": [
            stages[key].get("intensity_audit", {}).get("nonzero_fraction", np.nan)
            for key in order
        ],
    }

    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    for axis, (title, metric_values) in zip(axes, values.items()):
        metric_values = np.asarray(metric_values, dtype=float)
        visible = np.isfinite(metric_values)
        bars = axis.bar(x[visible], metric_values[visible], color=np.asarray(colors)[visible])
        axis.set_title(title)
        axis.set_xticks(x[visible], np.asarray(labels)[visible])
        axis.set_ylim(0, 1.02)
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, metric_values[visible]):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.015,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    figure.suptitle("Effect of NIfTI storage dtype on registration output", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    print("Figure:", args.output)


if __name__ == "__main__":
    main()
