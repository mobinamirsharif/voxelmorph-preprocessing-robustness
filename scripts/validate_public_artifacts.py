#!/usr/bin/env python3
"""Validate that publication files are public/synthetic and privacy-safe."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_REPORT = ROOT / "reports" / "public_demo" / "public_demo_metrics.json"
ALLOWED_REPORTS = {PUBLIC_REPORT.relative_to(ROOT).as_posix()}
MARKDOWN_FILES = (
    ROOT / "README.md",
    ROOT / "data" / "README.md",
    ROOT / "models" / "README.md",
    *sorted((ROOT / "docs").glob("*.md")),
)
LINK_PATTERN = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")
PRIVATE_PATTERNS = {
    "ADNI participant identifier": re.compile(r"\b\d{3}_?S_?\d{4}\b", re.IGNORECASE),
    "BIDS participant identifier": re.compile(r"\bsub-\d{3}[^\s/]*", re.IGNORECASE),
    "WSL absolute path": re.compile(r"/mnt/[a-z]/"),
    "Linux home path": re.compile(r"/home/[^/\s]+/"),
    "Windows user path": re.compile(r"[A-Za-z]:\\Users\\", re.IGNORECASE),
    "restricted participant label": re.compile(r"single authorized participant", re.IGNORECASE),
}
PARTICIPANT_RESULT_PATTERNS = {
    "baseline label": re.compile(r"\bbaseline\b", re.IGNORECASE),
    "follow-up label": re.compile(r"\bsix[- ]month\b", re.IGNORECASE),
}
FORBIDDEN_SUFFIXES = (
    ".nii",
    ".nii.gz",
    ".mgz",
    ".dcm",
    ".h5",
    ".keras",
    ".npz",
    ".lta",
)
FORBIDDEN_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp")


def publication_files() -> list[Path]:
    command = [
        "git",
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
    ]
    paths = subprocess.check_output(command, cwd=ROOT, text=True).splitlines()
    return [ROOT / path for path in paths]


def validate_report_allowlist(files: list[Path]) -> None:
    reports = {
        path.relative_to(ROOT).as_posix()
        for path in files
        if path.is_file() and path.relative_to(ROOT).parts[0] == "reports"
    }
    unexpected = reports - ALLOWED_REPORTS
    missing = ALLOWED_REPORTS - reports
    if unexpected or missing:
        raise ValueError(
            f"Report allowlist mismatch; unexpected={sorted(unexpected)}, "
            f"missing={sorted(missing)}"
        )


def validate_public_report() -> None:
    report = json.loads(PUBLIC_REPORT.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or not report:
        raise ValueError("Expected a non-empty public-demo report")
    provenance = report.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Public report is missing provenance")
    if provenance.get("classification") not in {"public", "synthetic"}:
        raise ValueError("Report provenance must be public or synthetic")
    for key in ("atlas", "test_scan", "model_filename", "model_sha256"):
        if not provenance.get(key):
            raise ValueError(f"Public report provenance is missing {key}")


def validate_no_binary_research_artifacts(files: list[Path]) -> None:
    for path in files:
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        lowered = relative.lower()
        if lowered.endswith(FORBIDDEN_SUFFIXES):
            raise ValueError(f"Forbidden research or model artifact: {relative}")
        if lowered.endswith(FORBIDDEN_IMAGE_SUFFIXES):
            raise ValueError(f"Image publication artifact is not allowlisted: {relative}")
        if "warp" in path.name.lower() and path.suffix.lower() != ".py":
            raise ValueError(f"Potential deformation artifact: {relative}")


def validate_relative_links() -> None:
    for markdown in MARKDOWN_FILES:
        text = markdown.read_text(encoding="utf-8")
        for target in LINK_PATTERN.findall(text):
            target = target.split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            resolved = (markdown.parent / target).resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"Broken link in {markdown}: {target}")


def validate_public_text_privacy(files: list[Path]) -> None:
    text_suffixes = {".cff", ".json", ".md", ".py", ".toml", ".txt", ".yml", ".yaml"}
    for path in files:
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8")
        for label, pattern in PRIVATE_PATTERNS.items():
            if pattern.search(text):
                raise ValueError(f"{label} found in publication file: {path}")

    for path in MARKDOWN_FILES:
        text = path.read_text(encoding="utf-8")
        for label, pattern in PARTICIPANT_RESULT_PATTERNS.items():
            if pattern.search(text):
                raise ValueError(f"{label} found in publication documentation: {path}")


def main() -> None:
    files = publication_files()
    validate_report_allowlist(files)
    validate_public_report()
    validate_no_binary_research_artifacts(files)
    validate_relative_links()
    validate_public_text_privacy(files)
    print("Public/synthetic provenance, artifacts, links, and privacy validated.")


if __name__ == "__main__":
    main()
