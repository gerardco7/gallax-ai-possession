"""Half boundaries and sampling blocks shared by feature extraction and decoding."""

import numpy as np


def slices(period, include_outside=False):
    """Contiguous runs of a single period; even adjacent halves remain separate."""
    period = np.asarray(period)
    if not len(period):
        return
    edges = np.r_[0, np.flatnonzero(period[1:] != period[:-1]) + 1, len(period)]
    for a, b in zip(edges[:-1], edges[1:]):
        if include_outside or period[a] > 0:
            yield int(a), int(b)


def blocks(period, step):
    """Include partial tails; no averaging block crosses a period boundary."""
    if step < 1:
        raise ValueError("step must be positive")
    starts, stops = [], []
    for a, b in slices(period, include_outside=True):
        for start in range(a, b, step):
            starts.append(start)
            stops.append(min(start + step, b))
    starts, stops = np.asarray(starts, dtype=int), np.asarray(stops, dtype=int)
    return starts, stops, (starts + stops - 1) // 2


def expand(values, period, step):
    starts, stops, _ = blocks(period, step)
    if len(values) != len(starts):
        raise ValueError("Step predictions do not match the period-safe sampling grid")
    return np.repeat(values, stops - starts, axis=0)
