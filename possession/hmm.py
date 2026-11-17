"""Possession as a hidden Markov model over three states: home (H), away (A), dead ball (D).

Every frame gives evidence -- two logistic models trained on the external games: one for "is the
ball dead?" (the gap classifier's stoppage probability, whether the ball is outside the lines, how
fast the players move, how close anybody is), one for "if alive, home or away?" (how close each
team's nearest player is and by how much). Where the evidence is weak -- a ball in flight, nobody
near it, no ball at all -- it says little and the chain keeps its state.

The transitions are counted from the external games' labels: how often a frame of H is followed by
A or D, and so on, which is what makes a possession last as long as possessions do. Viterbi then
finds the most likely sequence of states.

Two knobs, chosen by tune_hmm.py on the realistic external games: ALPHA scales the per-frame
evidence (consecutive frames are far from independent, so taken at face value the evidence would be
counted many times over) and BETA scales how easily the state changes.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import explore as ex  # noqa: E402
import frames as data  # noqa: E402
import gaps  # noqa: E402
import control  # noqa: E402
import periods  # noqa: E402

FPS = 25
ALPHA = 0.2
BETA = 1.0
SAMPLE_EVERY = 5  # training frames: consecutive ones are near-duplicates
PROXIMITY_M = 1.5
STATES = np.array(["H", "A", "D"])

DEAD_FEATURES = ["seen", "p_gap", "outside", "d_line", "pace", "near", "ball_speed"]
TEAM_FEATURES = ["seen", "diff", "prox_home", "prox_away", "ball_speed"]
CONTROL_DEAD = ["ctrl_dead"]
CONTROL_TEAM = ["ctrl_home", "ctrl_away"]


def frame_features(frames, gap_model):
    """Per-frame evidence. Ball-dependent features are NaN without a ball (the logistic models fill
    them with the training mean, i.e. "no evidence") except the proximities, which are 0. The
    control rule's own decision is included as three indicators (home / away / dead)."""
    seen = ~np.isnan(frames["ball"][:, 0])
    d_home, d_away = ex.nearest_distances(frames)
    d_line = ex.distance_to_line(np.nan_to_num(frames["ball"], nan=-99.0), frames["pitch"])
    d_line[~seen] = np.nan
    v = np.full(len(seen), np.nan)
    v[1:] = np.linalg.norm(np.diff(frames["ball"], axis=0), axis=1) * FPS
    pace = ex._nanmean(np.stack([ex.team_speed(frames, "home"), ex.team_speed(frames, "away")]), 0)
    probs = gaps.kind_probabilities(frames, gap_model)
    p_gap = np.max(list(probs.values()), axis=0)
    ctrl = control.control_codes(frames["period"], gaps.decide(probs), d_home, d_away, control.ball_speed(frames["ball"]),
                                 control.CONTROL_R, control.CONTROL_T, control.CONTROL_V)
    return pd.DataFrame({
        "ctrl_home": (ctrl == control.HOME).astype(float),
        "ctrl_away": (ctrl == control.AWAY).astype(float),
        "ctrl_dead": (ctrl == control.NOBODY).astype(float),
        "seen": seen.astype(float),
        "p_gap": p_gap,
        "outside": (d_line < 0).astype(float),
        "d_line": np.clip(d_line, -5, 10),
        "pace": pace,
        "near": np.clip(np.fmin(d_home, d_away), 0, 15),
        "ball_speed": np.clip(v, 0, 40),
        "diff": np.clip(d_away - d_home, -10, 10),
        "prox_home": np.nan_to_num(np.exp(-d_home / PROXIMITY_M)),
        "prox_away": np.nan_to_num(np.exp(-d_away / PROXIMITY_M)),
    })


class PossessionHMM:
    def __init__(self, use_control=False):
        """use_control: also give the evidence models the control rule's decision (stacking)."""
        self.use_control = use_control

    def fit(self, games, stoppages=None):
        """games: list of (frames, labels) external games."""
        self.gap_model = stoppages or gaps.train(games)
        xs, ys = [], []
        for frames, truth in games:
            for copy in data.training_copies(frames, truth, gaps.TRAINING_SEEDS):
                keep = np.flatnonzero(frames["period"] > 0)[::SAMPLE_EVERY]
                xs.append(frame_features(copy, self.gap_model).iloc[keep])
                ys.append(truth[keep])
        x, y = pd.concat(xs, ignore_index=True), np.concatenate(ys)
        alive = y != "D"
        extra_dead, extra_team = (CONTROL_DEAD, CONTROL_TEAM) if self.use_control else ([], [])
        self.dead = gaps.GapModel(DEAD_FEATURES + extra_dead).fit(x, y == "D")
        self.team = gaps.GapModel(TEAM_FEATURES + extra_team).fit(x[alive], y[alive] == "H")
        self.prior = np.array([np.mean(y == s) for s in STATES])
        self.counts = self._transition_counts([t[a:b] for f, t in games for a, b in periods.slices(f["period"])])
        return self

    def evidence(self, frames):
        """(n, 3) log P(state | frame) - log P(state): the per-frame evidence for H, A, D."""
        x = frame_features(frames, self.gap_model)
        p_dead = np.clip(self.dead.predict(x), 1e-6, 1 - 1e-6)
        p_home = np.clip(self.team.predict(x), 1e-6, 1 - 1e-6)
        post = np.stack([(1 - p_dead) * p_home, (1 - p_dead) * (1 - p_home), p_dead], axis=1)
        return np.log(post) - np.log(self.prior)

    def decode(self, frames, evidence, alpha=ALPHA, beta=BETA):
        """Most likely H/A/D sequence per half; X outside play."""
        log_a = np.log(self._transitions(beta))
        out = np.full(len(frames["period"]), "X", dtype="<U1")
        for a, b in periods.slices(frames["period"]):
            out[a:b] = STATES[viterbi(alpha * evidence[a:b], log_a)]
        return out

    def posterior(self, frames, evidence, alpha=ALPHA, beta=BETA):
        """(n, 3) probability of H, A, D per frame (forward-backward); 0 outside play."""
        log_a = np.log(self._transitions(beta))
        out = np.zeros((len(frames["period"]), 3))
        for a, b in periods.slices(frames["period"]):
            out[a:b] = forward_backward(alpha * evidence[a:b], log_a)
        return out

    def prob_method(self, broadcast_ball=False):
        def run(frames):
            if broadcast_ball:
                frames = data.with_broadcast_ball(frames)
            return self.posterior(frames, self.evidence(frames))
        return run

    def both(self, frames):
        """(labels, probabilities) from one evidence computation."""
        ev = self.evidence(frames)
        return self.decode(frames, ev), self.posterior(frames, ev)

    def method(self, alpha=ALPHA, beta=BETA, broadcast_ball=False):
        def run(frames):
            if broadcast_ball:
                frames = data.with_broadcast_ball(frames)
            return self.decode(frames, self.evidence(frames), alpha, beta)
        run.__name__ = ("hmm_ctrl" if self.use_control else "hmm") + ("_sync" if broadcast_ball else "")
        return run

    def _transition_counts(self, labels):
        counts = np.ones((3, 3))  # add-one so no transition is impossible
        index = {s: i for i, s in enumerate(STATES)}
        for truth in labels:
            for prev, nxt in zip(truth[:-1], truth[1:]):
                if prev in index and nxt in index:
                    counts[index[prev], index[nxt]] += 1
        return counts

    def _transitions(self, beta):
        rates = self.counts / self.counts.sum(1, keepdims=True)
        off = rates * beta
        np.fill_diagonal(off, 0)
        np.fill_diagonal(off, 1 - off.sum(1))
        return off


def forward_backward(log_emission, log_transition):
    """(n, k) posterior state probabilities of an HMM, in log space, uniform start."""
    n, k = log_emission.shape
    fwd = np.empty((n, k))
    bwd = np.zeros((n, k))
    fwd[0] = log_emission[0] - np.log(k)
    for t in range(1, n):
        fwd[t] = log_emission[t] + _logsumexp(fwd[t - 1][:, None] + log_transition)
    for t in range(n - 2, -1, -1):
        bwd[t] = _logsumexp((log_transition + log_emission[t + 1] + bwd[t + 1]).T)
    post = fwd + bwd
    post -= post.max(1, keepdims=True)
    post = np.exp(post)
    return post / post.sum(1, keepdims=True)


def _logsumexp(m):
    """log-sum-exp over axis 0."""
    top = m.max(0)
    return top + np.log(np.exp(m - top).sum(0))


def viterbi(log_emission, log_transition):
    n, k = log_emission.shape
    back = np.zeros((n, k), dtype=np.int8)
    score = log_emission[0] - np.log(k)
    for t in range(1, n):
        cand = score[:, None] + log_transition
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], np.arange(k)] + log_emission[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = np.argmax(score)
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path
