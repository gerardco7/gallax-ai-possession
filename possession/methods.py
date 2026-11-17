"""Possession methods. Each takes a `frames` dict (see frames.py) and returns one label per frame.

Add a method by writing a function and listing it in METHODS. bench.py scores every entry on the
external games; nothing here may be tuned on the provided match or its ground truth.
"""

import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "starter"))
from baseline import smooth  # noqa: E402  (the organisers' majority vote, reused unchanged)

sys.path.insert(0, HERE)
import gaps  # noqa: E402
import hmm  # noqa: E402
import touches  # noqa: E402
import vote  # noqa: E402
from control import CONTROL_R, CONTROL_T, CONTROL_V, ball_speed, control_codes  # noqa: E402
from frames import with_broadcast_ball  # noqa: E402

HOME, AWAY, NOBODY = 0, 1, -1


def nearest_team(frames):
    """Per frame: HOME/AWAY for the team with the player nearest the ball, NOBODY without a ball."""
    ball = frames["ball"][:, None, :]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN row = no player seen
        d_home = np.nanmin(np.linalg.norm(frames["home"] - ball, axis=2), axis=1)
        d_away = np.nanmin(np.linalg.norm(frames["away"] - ball, axis=2), axis=1)
    codes = np.full(len(ball), NOBODY, dtype=np.int8)
    seen = ~np.isnan(d_home) | ~np.isnan(d_away)
    codes[seen] = np.where(np.nan_to_num(d_home[seen], nan=np.inf)
                           <= np.nan_to_num(d_away[seen], nan=np.inf), HOME, AWAY)
    return codes


def outside_play_is_x(frames, labels):
    labels = np.asarray(labels, dtype="<U1").copy()
    labels[frames["period"] == 0] = "X"
    return labels


def baseline(window):
    """The organisers' reference: nearest player, D without a ball, centred majority vote."""
    def method(frames):
        return outside_play_is_x(frames, smooth(nearest_team(frames), window))
    method.__name__ = "baseline_smooth{}".format(window)
    return method



def carry(window):
    """The baseline, except that a frame with no ball keeps the last team instead of calling it
    dead. The ball is never dead here: the point is to see what assuming play goes on costs."""
    def method(frames):
        codes = nearest_team(frames)
        last = NOBODY
        for i in range(len(codes)):
            if frames["period"][i] == 0:
                last = NOBODY
            elif codes[i] != NOBODY:
                last = codes[i]
            else:
                codes[i] = last
        return outside_play_is_x(frames, smooth(codes, window))
    method.__name__ = "carry_smooth{}".format(window)
    return method


def gap_classifier(model, window=50, broadcast_ball=False):
    """Nearest team where the ball is seen; dead inside gaps the gap model calls a stoppage; the
    last team everywhere else the ball is missing (play the tracker lost). Then the majority vote.

    broadcast_ball=True first fills the ball from the live broadcast. Whether the broadcast is live
    is never used to call a stoppage: it stays the independent check of gaps.tv_check()."""
    def method(frames):
        if broadcast_ball:
            frames = with_broadcast_ball(frames)
        codes = nearest_team(frames)
        dead = gaps.dead_mask(frames, model)
        last = NOBODY
        for i in range(len(codes)):
            if frames["period"][i] == 0:
                last = NOBODY
            elif dead[i]:
                codes[i] = NOBODY
            elif codes[i] != NOBODY:
                last = codes[i]
            else:
                codes[i] = last
        return outside_play_is_x(frames, smooth(codes, window))
    method.__name__ = "gapclf_{}{}".format("sync" if broadcast_ball else "smooth", window)
    return method


def control(model, r=None, t=None, v=None, broadcast_ball=False):
    """Gap classifier for stoppages, control rule for who has the ball."""
    r, t, v = (CONTROL_R if r is None else r), (CONTROL_T if t is None else t), (CONTROL_V if v is None else v)

    def method(frames):
        if broadcast_ball:
            frames = with_broadcast_ball(frames)
        d_home, d_away = gaps.ex.nearest_distances(frames)
        codes = control_codes(frames["period"], gaps.dead_mask(frames, model), d_home, d_away,
                              ball_speed(frames["ball"]), r, t, v)
        return outside_play_is_x(frames, np.where(codes == HOME, "H", np.where(codes == AWAY, "A", "D")))
    method.__name__ = "control_sync" if broadcast_ball else "control"
    return method


def control_both(model):
    """(labels, probabilities) of the control rule from one computation of its inputs."""
    def run(frames):
        probs = gaps.kind_probabilities(frames, model)
        d_home, d_away = gaps.ex.nearest_distances(frames)
        codes = control_codes(frames["period"], gaps.decide(probs), d_home, d_away, ball_speed(frames["ball"]),
                              CONTROL_R, CONTROL_T, CONTROL_V)
        labels = outside_play_is_x(frames, np.where(codes == HOME, "H", np.where(codes == AWAY, "A", "D")))
        p_dead = np.max(list(probs.values()), axis=0)
        home = np.where(labels == "H", 1.0, np.where(labels == "A", 0.0, 0.5))
        out = np.stack([(1 - p_dead) * home, (1 - p_dead) * (1 - home), p_dead], axis=1)
        out[frames["period"] == 0] = 0
        return labels, out
    return run


def control_probabilities(model, broadcast_ball=False):
    """(n, 3) [P(H), P(A), P(D)] for the control rule: its team, scaled by the stoppage detectors'
    probability that the ball is alive; where it calls the ball dead the team is unknown (50/50)."""
    label_method = control(model, broadcast_ball=broadcast_ball)

    def run(frames):
        labels = label_method(frames)
        f = with_broadcast_ball(frames) if broadcast_ball else frames
        p_dead = gaps.dead_probability(f, model)
        home = np.where(labels == "H", 1.0, np.where(labels == "A", 0.0, 0.5))
        out = np.stack([(1 - p_dead) * home, (1 - p_dead) * (1 - home), p_dead], axis=1)
        out[frames["period"] == 0] = 0
        return out
    return run


def baseline_sync(window):
    """The baseline plus the aligned broadcast (needs frames.add_broadcast), fixed before scoring:
      a) where the tactical ball is missing, use the live broadcast's ball unless the two balls
         disagree in that window;
      b) where nobody sees the ball but the broadcast is live, play is going on: keep the last
         team instead of calling it dead. Where the broadcast is not live, stay as the baseline.
    """
    def method(frames):
        codes = nearest_team(with_broadcast_ball(frames))
        last = NOBODY
        for i in range(len(codes)):
            if frames["period"][i] == 0:
                last = NOBODY
            elif codes[i] != NOBODY:
                last = codes[i]
            elif frames["broadcast_live"][i]:
                codes[i] = last
        return outside_play_is_x(frames, smooth(codes, window))
    method.__name__ = "baseline_sync{}".format(window)
    return method


METHODS = {m.__name__: m for m in (baseline(0), baseline(25), baseline(50), baseline(125), carry(50))}
# Need training: built per test game from the other external games, and from all of them for the
# provided match. Values take the list of (frames, labels) to train on and return a method.
TRAINABLE = {
    "gapclf_smooth50": lambda train_games: gap_classifier(gaps.train(train_games), 50),
    "gapclf_sync50": lambda train_games: gap_classifier(gaps.train(train_games), 50, broadcast_ball=True),
    "control": lambda train_games: control(gaps.train(train_games)),
    "control_sync": lambda train_games: control(gaps.train(train_games), broadcast_ball=True),
    "hmm": lambda train_games: hmm.PossessionHMM().fit(train_games).method(),
    "hmm_sync": lambda train_games: hmm.PossessionHMM().fit(train_games).method(broadcast_ball=True),
    "touches": lambda train_games: touches.TouchModel().fit(train_games).method(),
    "touches_sync": lambda train_games: touches.TouchModel().fit(train_games).method(broadcast_ball=True),
    "hmm_ctrl": lambda train_games: hmm.PossessionHMM(use_control=True).fit(train_games).method(),
    "hmm_ctrl_sync": lambda train_games: hmm.PossessionHMM(use_control=True).fit(train_games).method(broadcast_ball=True),
}
# Need the broadcast, which exists only for the provided match: never benched on external games.
MATCH_ONLY = {m.__name__: m for m in (baseline_sync(50),)}
TRAINABLE["vote"] = lambda train_games: vote.voting([TRAINABLE[v](train_games) for v in vote.VOTERS])
TRAINABLE["vote_sync"] = lambda train_games: vote.voting(
    [TRAINABLE[v + "_sync"](train_games) for v in vote.VOTERS], broadcast_ball=True)
# Probability voters, in vote.VOTERS order, for the soft vote.
PROBABILITY = {
    "touches": lambda g, b=False: touches.TouchModel().fit(g).prob_method(broadcast_ball=b),
    "hmm": lambda g, b=False: hmm.PossessionHMM().fit(g).prob_method(broadcast_ball=b),
    "control": lambda g, b=False: control_probabilities(gaps.train(g), broadcast_ball=b),
}
TRAINABLE["vote_soft"] = lambda train_games: vote.soft_voting([PROBABILITY[v](train_games) for v in vote.VOTERS])
TRAINABLE["vote_soft_sync"] = lambda train_games: vote.soft_voting(
    [PROBABILITY[v](train_games, True) for v in vote.VOTERS], broadcast_ball=True)
NEEDS_BROADCAST = {"baseline_sync50", "gapclf_sync50", "control_sync", "hmm_sync", "hmm_ctrl_sync", "touches_sync", "vote_sync", "vote_soft_sync"}
