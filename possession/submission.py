"""Validated, atomic CSV export shared by submission generation and integration tests."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "evaluation"))
import possession_metrics as pm  # noqa: E402
import provenance  # noqa: E402


def write_submission(path, labels, expected_rows):
    """Check against an independent expected size before replacing any existing output."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or expected_rows <= 0:
        raise ValueError("Expected a one-dimensional label sequence and a positive frame count")
    if not np.isin(labels, list("HADX")).all():
        raise ValueError("Submission labels must be canonical H, A, D or X")
    table = pd.DataFrame({"frame": np.arange(len(labels), dtype=np.int64), "label": labels})
    reason = pm.validate_submission(table, expected_rows)
    if reason:
        raise ValueError(reason)
    provenance.atomic_bytes(path, table.to_csv(index=False, lineterminator="\n").encode("ascii"))
    return Path(path)
