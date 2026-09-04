from pathlib import Path
import os
import subprocess
import sys


def test_public_demo_cli_help_is_lightweight():
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "run_public_demo.py"), "--help"],
        check=True,
        capture_output=True,
        env=environment,
        text=True,
    )
    assert "--atlas-npz" in result.stdout
    assert "--test-scan-npz" in result.stdout
    assert "--output-dir" in result.stdout
