"""Check a submission CSV against the rules in CLAUDE.md. Usage: python src/validate_submission.py [csv]"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEAM_ID = "348471A6"
SUBMISSION = ROOT / "submissions" / f"team_{TEAM_ID}.csv"


def validate(path):
    path = Path(path)
    sub = pd.read_csv(path)
    ids = pd.read_csv(ROOT / "data" / "test_signals.csv").signal_id
    assert path.name == SUBMISSION.name, f"bad filename {path.name}, expected {SUBMISSION.name}"
    assert list(sub.columns) == ["signal_id", "ehtimollik"], f"bad columns {list(sub.columns)}"
    assert len(sub) == len(ids), f"{len(sub)} rows, expected {len(ids)}"
    assert not sub.signal_id.duplicated().any(), "duplicate signal_id"
    assert set(sub.signal_id) == set(ids), "missing or unknown signal_id"
    assert sub.ehtimollik.notna().all(), "NaN in ehtimollik"
    assert sub.ehtimollik.between(0, 1).all(), "ehtimollik outside [0, 1]"
    print(f"OK: {path.name} — {len(sub)} rows, valid")


if __name__ == "__main__":
    validate(sys.argv[1] if len(sys.argv) > 1 else SUBMISSION)
