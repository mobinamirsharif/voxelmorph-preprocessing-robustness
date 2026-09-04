#!/usr/bin/env python3
"""Convert one DICOM series to compressed NIfTI with dcm2niix."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from voxelmorph_pipeline.naming import image_stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dicom-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--session", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if shutil.which("dcm2niix") is None:
        raise RuntimeError("dcm2niix is not available on PATH")
    if not args.dicom_dir.is_dir():
        raise FileNotFoundError(args.dicom_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{image_stem(args.subject, args.session)}_T1w"
    command = [
        "dcm2niix",
        "-b",
        "y",
        "-ba",
        "y",
        "-z",
        "y",
        "-f",
        filename,
        "-o",
        str(args.output_dir),
        str(args.dicom_dir),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

