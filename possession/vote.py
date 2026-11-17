"""Vote between the three methods: control rule, HMM and touch detector.

LAMBDA is chosen by tune_vote.py on the realistic external games.

Each voter labels every frame H, A or D. A state collects one vote per voter that chose it, and
changing state costs LAMBDA votes; Viterbi (hmm.viterbi) finds the sequence with the most votes
net of those costs. LAMBDA = 0 is a plain frame-by-frame majority (ties go to the first voter);
higher values change state only when the majority holds for a while, so a brief disagreement does
not become a possession of its own.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import explore as ex  # noqa: E402
import hmm  # noqa: E402
import periods  # noqa: E402

LAMBDA_SOFT = 5.0  # chosen by tune_vote.py --soft (all three folds agree)
LAMBDA = 20.0  # chosen by tune_vote.py on the realistic external games (all three folds agree)
VOTERS = ("touches", "hmm", "control")  # first voter breaks ties
STATES = np.array(["H", "A", "D"])


def decode(score, period, lam, step=1):
    """Viterbi over per-frame state scores with switching cost lam (per 25 fps frame). step > 1
    decodes blocks of `step` frames (mean score, lam scaled to match) and spreads each block's state
    over its frames: step times less work, at the cost of boundaries placed to the block."""
    out = np.full(len(period), "X", dtype="<U1")
    if step > 1:
        starts, stops, centres = periods.blocks(period, step)
        block_score = np.asarray([score[a:b].mean(0) for a, b in zip(starts, stops)]).reshape(-1, 3)
        block = decode(block_score, period[centres], lam / step)
        out = periods.expand(block, period, step)
        out[period == 0] = "X"
        return out
    log_t = np.where(np.eye(3, dtype=bool), 0.0, -lam)
    for a, b in periods.slices(period):
        out[a:b] = STATES[hmm.viterbi(score[a:b], log_t)]
    return out


def combine(labels, period, lam=LAMBDA, step=1):
    """labels: list of per-frame H/A/D/X arrays, first voter first. Returns the voted labels."""
    votes = np.stack([np.stack([lab == s for s in STATES], axis=1) for lab in labels]).astype(float)
    return decode(votes.sum(0) + 1e-3 * votes[0], period, lam, step)  # the first voter breaks ties


def voting(voters, broadcast_ball=False, lam=None):
    """A method combining already-trained voter methods, given in VOTERS order."""
    suffix = "_sync" if broadcast_ball else ""

    def run(frames):
        return combine([v(frames) for v in voters], frames["period"], LAMBDA if lam is None else lam)
    run.__name__ = "vote" + suffix
    return run


def combine_soft(probabilities, period, lam=None, step=1):
    """probabilities: list of (n, 3) [P(H), P(A), P(D)] arrays, first voter first. Each state scores
    the sum of the voters' probabilities for it; changing state costs lam, as in combine()."""
    lam = LAMBDA_SOFT if lam is None else lam
    return decode(np.sum(probabilities, axis=0) + 1e-3 * probabilities[0], period, lam, step)


def soft_voting(voters, broadcast_ball=False, lam=None):
    """A method combining trained probability voters (VOTERS order) with combine_soft."""
    def run(frames):
        return combine_soft([v(frames) for v in voters], frames["period"], lam)
    run.__name__ = "vote_soft" + ("_sync" if broadcast_ball else "")
    return run


def decode_probabilities(probabilities, period, lam, score_mode="probability", epsilon=1e-8):
    """Decode one pooled posterior sequence; log mode changes emissions, not transitions.

    Ensemble probabilities must be averaged before this call. Clipping only makes log(0)
    finite; no calibration or class-prior correction is fitted here.
    """
    probabilities = np.asarray(probabilities)
    period = np.asarray(period)
    if probabilities.shape != (len(period), 3):
        raise ValueError("Expected one H/A/D probability row per period entry")
    valid = probabilities[period > 0]
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or (probabilities > 1).any()
            or not np.allclose(valid.sum(1), 1., atol=1e-6)):
        raise ValueError("Expected finite normalised probabilities on scored steps")
    if score_mode == "probability":
        return combine_soft([probabilities], period, lam)
    if score_mode != "log" or not 0 < epsilon < 1:
        raise ValueError("Expected probability/log mode and epsilon in (0, 1)")
    return decode(np.log(np.clip(probabilities, epsilon, 1.)), period, lam)
