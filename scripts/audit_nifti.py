#!/usr/bin/env python3
"""Audit NIfTI storage and intensities for silent quantization risks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from voxelmorph_pipeline.quality import audit_nifti


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Return a non-zero exit code for integer storage as well as failures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audits = [audit_nifti(path).to_dict() for path in args.images]
    summary = {"images": audits}

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, indent=2))

    blocked = any(item["status"] == "fail" for item in audits)
    warned = any(item["status"] == "warn" for item in audits)
    if blocked or (args.fail_on_warning and warned):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
