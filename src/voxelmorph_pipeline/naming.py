"""BIDS-inspired output naming helpers."""

from __future__ import annotations

import re


_SAFE_LABEL = re.compile(r"^[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*$")


def validate_label(value: str, label_name: str) -> str:
    """Reject path separators and unsafe subject or session labels."""
    if not _SAFE_LABEL.fullmatch(value):
        raise ValueError(f"Invalid {label_name} label: {value!r}")
    return value


def image_stem(subject: str, session: str) -> str:
    """Build a BIDS-inspired subject/session stem."""
    subject = validate_label(subject, "subject")
    session = validate_label(session, "session")
    return f"sub-{subject}_ses-{session}"

