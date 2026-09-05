import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_public_validation_artifacts():
    m = json.loads((ROOT / "reports/public_validation/public_validation_metrics.json").read_text())
    p = json.loads((ROOT / "reports/public_validation/public_validation_manifest.json").read_text())
    assert m["dataset"]["license"] == "CC0-1.0"
    assert m["dataset"] == p["dataset"] and len(m["runs"]) == 4
    text = " ".join((ROOT / x).read_text() for x in ["reports/public_validation/public_validation_metrics.json", "reports/public_validation/public_validation_manifest.json", "reports/public_validation/README.md"])
    for token in ("/mnt/", "C:\\Users\\", "single " + "authorized participant"):
        assert token.lower() not in text.lower()
    f = ROOT / "figures/public_validation/public_validation_metrics.png"
    assert f.is_file() and f.stat().st_size > 0
