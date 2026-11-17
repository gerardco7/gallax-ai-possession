"""Stoppage detection: per-stretch classifiers for gaps (no ball) and for still balls.

A gap -- a stretch of play with no ball -- is either a stoppage or play the tracker lost. A still
ball -- seen, but hardly moving for a while -- is either a stoppage (held for a throw-in, placed
for a free kick or a corner) or play. Each kind has its own classifier with the same features, and a
frame is dead when either calls the stretch it lies in a stoppage.

    python possession/gaps.py          # leave-one-game-out on the external games, then the TV check

Trained only on external games. A gap's features are the ones both the external games and the
provided match have: how long it lasts, how the players move before, during and after it, and where
the ball was when it was lost and where it reappears. The broadcast and the Smart Tagging exist only
for the provided match, so they are not features: the broadcast is kept back as an independent
check (does the classifier's dead ball coincide with the TV cutting away?), and nothing is tuned on
it.

The training set per external game: its stretches as they are, plus those of realistic copies
(frames.degrade), which add play the tracker "lost" and a visible ball in most stoppages (the
external games remove it). A stretch is dead when most of its frames are labelled D.

The model is an L2-regularised logistic regression, written out in numpy rather than imported: the
whole thing is a dozen lines, it is deterministic, and its weights read as "this feature pushes
towards a stoppage".
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import explore as ex  # noqa: E402
import external_metrica  # noqa: E402
import frames as data  # noqa: E402

FPS = 25
MIN_GAP_S = 1          # shorter gaps are left to the possession logic: there is no stoppage that short
PACE_WINDOW_S = 3
CORNER_M = 3.0
CENTRE_M = 3.0
GOAL_AREA = (5.5, 9.16)  # depth, half width (m): where a goal kick is taken
L2 = 1.0
TRAINING_SEEDS = (100, 101, 102)  # realistic copies per external game, kept apart from the bench's
STILL_WINDOW = 25      # frames: ball speed for "still" is measured over one second, above tracking drift
STILL_V = 1.5          # m/s: slower than this the ball is still (a placed ball drifts ~0.8 m/s in the noise)
MIN_STILL_S = 1
# How sure each classifier must be to call a stretch a stoppage. The still-ball one is chosen by
# tune_control.py: a slow ball in open play costs a false stoppage, so it may need to be surer.
THRESHOLDS = {"gap": 0.5, "still": 0.6}

FEATURES = ["log_seconds", "pace_before", "pace_in", "pace_after", "pace_drop", "pace_rise",
            "d_last", "last_outside", "d_projected", "d_first", "first_at_corner",
            "first_in_goal_area", "first_at_centre", "ball_speed_before"]


def gap_spans(frames):
    """(starts, stops) of every stretch of play with no ball lasting MIN_GAP_S or more."""
    missing = np.isnan(frames["ball"][:, 0]) & (frames["period"] > 0)
    starts, stops = ex.runs(missing)
    keep = (stops - starts) >= MIN_GAP_S * FPS
    return starts[keep], stops[keep]


def still_spans(frames):
    """(starts, stops) of every stretch where the ball is seen and hardly moving for MIN_STILL_S."""
    ball = frames["ball"]
    v = np.full(len(ball), np.nan)
    v[STILL_WINDOW:] = np.linalg.norm(ball[STILL_WINDOW:] - ball[:-STILL_WINDOW], axis=1) * FPS / STILL_WINDOW
    with np.errstate(invalid="ignore"):
        still = (v < STILL_V) & (frames["period"] > 0)
    starts, stops = ex.runs(still)
    starts = np.maximum(starts - STILL_WINDOW, 0)  # the speed looks back a second: so does the stretch
    keep = (stops - starts) >= MIN_STILL_S * FPS
    return starts[keep], stops[keep]


SPANS = {"gap": gap_spans, "still": still_spans}


def gap_features(frames, starts, stops):
    n, (length, width) = len(frames["period"]), frames["pitch"]
    pace = ex._nanmean(np.stack([ex.team_speed(frames, "home"), ex.team_speed(frames, "away")]), 0)
    ball, w = frames["ball"], PACE_WINDOW_S * FPS
    last = ball[np.maximum(starts - 1, 0)]
    first = ball[np.minimum(stops, n - 1)]
    earlier = ball[np.maximum(starts - 6, 0)]
    velocity = (last - earlier) / (5 / FPS)  # m/s over the last 0.2 s before the gap
    rows = pd.DataFrame({
        "log_seconds": np.log((stops - starts) / FPS),
        "pace_before": [ex._nanmean(pace[max(a - w, 0):a], 0) for a in starts],
        "pace_in": [ex._nanmean(pace[a:b], 0) for a, b in zip(starts, stops)],
        "pace_after": [ex._nanmean(pace[b:b + w], 0) for b in stops],
        "d_last": ex.distance_to_line(last, frames["pitch"]),
        "d_projected": ex.distance_to_line(last + velocity, frames["pitch"]),  # where it was heading, 1 s on
        "d_first": ex.distance_to_line(first, frames["pitch"]),
        "ball_speed_before": np.linalg.norm(velocity, axis=1),
    })
    rows["pace_drop"] = rows["pace_before"] - rows["pace_in"]
    rows["pace_rise"] = rows["pace_after"] - rows["pace_in"]
    rows["last_outside"] = (rows["d_last"] < 0).astype(float)
    corners = np.array([[0, 0], [length, 0], [0, width], [length, width]])
    rows["first_at_corner"] = (np.min(np.linalg.norm(first[:, None] - corners[None], axis=2), 1)
                               < CORNER_M).astype(float)
    rows["first_in_goal_area"] = (((first[:, 0] < GOAL_AREA[0]) | (first[:, 0] > length - GOAL_AREA[0]))
                                  & (np.abs(first[:, 1] - width / 2) < GOAL_AREA[1])).astype(float)
    rows["first_at_centre"] = (np.linalg.norm(first - [length / 2, width / 2], axis=1) < CENTRE_M).astype(float)
    # A ball never seen on one side of the gap (start of a half, a tracker that never found it)
    # leaves NaN; the model fills it with the training mean, i.e. "no evidence".
    return rows[FEATURES]


def labelled_gaps(frames, truth, kind="gap"):
    starts, stops = SPANS[kind](frames)
    dead = np.array([np.mean(truth[a:b] == "D") > 0.5 for a, b in zip(starts, stops)], dtype=bool)
    return gap_features(frames, starts, stops), dead


def training_set(games, kind="gap"):
    """Features and dead/alive labels from each external game and its realistic copies."""
    xs, ys = [], []
    for frames, truth in games:
        for copy in data.training_copies(frames, truth, TRAINING_SEEDS):
            x, y = labelled_gaps(copy, truth, kind)
            xs.append(x)
            ys.append(y)
    return pd.concat(xs, ignore_index=True), np.concatenate(ys)


class GapModel:
    """Standardised L2 logistic regression fitted by Newton's method. Also used by hmm.py with its
    own feature list."""

    def __init__(self, features=FEATURES):
        self.features = list(features)

    def fit(self, x, y):
        x = x[self.features]
        self.mean = x.mean()
        self.std = x.std().replace(0, 1).fillna(1)
        z = self._design(x)
        self.w = np.zeros(z.shape[1])
        penalty = L2 * np.eye(z.shape[1])
        penalty[0, 0] = 0  # the intercept is not shrunk
        for _ in range(50):
            p = 1 / (1 + np.exp(-z @ self.w))
            step = np.linalg.solve(z.T @ (z * (p * (1 - p))[:, None]) + penalty,
                                   z.T @ (p - y) + penalty @ self.w)
            self.w -= step
            if np.abs(step).max() < 1e-8:
                break
        return self

    def predict(self, x):
        """Probability of the positive class (for the gap model: that the gap is a stoppage)."""
        return 1 / (1 + np.exp(-self._design(x) @ self.w))

    def weights(self):
        return pd.Series(self.w[1:], index=self.features).sort_values()

    def _design(self, x):
        x = x[self.features]
        z = ((x.fillna(self.mean) - self.mean) / self.std).to_numpy(dtype=float)
        return np.hstack([np.ones((len(z), 1)), z])


class Stoppages:
    """One classifier per kind of stretch (gap, still ball)."""

    def __init__(self, models):
        self.models = models


def train(games):
    return Stoppages({kind: GapModel().fit(*training_set(games, kind)) for kind in SPANS})


def kind_probabilities(frames, model):
    """Per kind of stretch, per frame: the stoppage probability of the stretch it lies in, else 0."""
    out = {}
    for kind, kind_model in model.models.items():
        prob = np.zeros(len(frames["period"]))
        starts, stops = SPANS[kind](frames)
        if len(starts):
            for a, b, p in zip(starts, stops, kind_model.predict(gap_features(frames, starts, stops))):
                prob[a:b] = np.maximum(prob[a:b], p)
        out[kind] = prob
    return out


def decide(probabilities, thresholds=None):
    """Dead where any kind's probability clears its threshold."""
    thresholds = thresholds or THRESHOLDS
    return np.any([p > thresholds[k] for k, p in probabilities.items()], axis=0)


def dead_probability(frames, model):
    """Per frame: the highest stoppage probability of any stretch it lies in; 0 outside them."""
    return np.max(list(kind_probabilities(frames, model).values()), axis=0)


def dead_mask(frames, model):
    """Per frame: inside a stretch its classifier calls a stoppage."""
    return decide(kind_probabilities(frames, model))


def _scores(y, p):
    pred = p > 0.5
    tp, fp, fn = np.sum(pred & y), np.sum(pred & ~y), np.sum(~pred & y)
    return {"gaps": len(y), "dead %": 100 * y.mean(), "accuracy %": 100 * np.mean(pred == y),
            "dead precision %": 100 * tp / max(tp + fp, 1), "dead recall %": 100 * tp / max(tp + fn, 1)}


def leave_one_game_out():
    games = {g: data.load_metrica(g) for g in external_metrica.GAMES}
    rows = []
    for test in games:
        model = train([games[g] for g in games if g != test])
        frames, truth = games[test]
        for name, copy in (("clean", frames), ("realistic", data.degrade(frames, truth, seed=test))):
            for kind in SPANS:
                x, y = labelled_gaps(copy, truth, kind)
                if len(y):
                    rows.append(dict(_scores(y, model.models[kind].predict(x)), test_game=test, bench=name, kind=kind))
    print(pd.DataFrame(rows).set_index(["kind", "test_game", "bench"]).round(1))
    full = train(list(games.values()))
    for kind in SPANS:
        print("\n{} weights (trained on every external game; > 0 pushes towards a stoppage):".format(kind))
        print(full.models[kind].weights().round(2).to_string())


def tv_check():
    """On the provided match: does the classifier's dead ball coincide with the TV cutting away?
    Aggregates only; the broadcast is not a feature, so this is an independent check."""
    model = train([data.load_metrica(g) for g in external_metrica.GAMES])
    frames = data.add_broadcast(data.load_match())
    graded = frames["period"] > 0
    no_ball = graded & np.isnan(frames["ball"][:, 0])
    dead = dead_mask(frames, model)
    starts, stops = gap_spans(frames)
    tv_off = ~frames["broadcast_live"]
    print("\n== provided match: TV as an independent check ==")
    print("  gaps >= {} s: {}   called stoppage: {:.0%}".format(MIN_GAP_S, len(starts),
          np.mean([dead[a] for a in starts]) if len(starts) else 0))
    print("  graded time called dead: {:.1%}   (no-ball time: {:.1%})".format(dead[graded].mean(), no_ball[graded].mean()))
    print("  TV not live -- in gaps called stoppage: {:.1%}   in gaps called play: {:.1%}   with the ball seen: {:.1%}".format(
        tv_off[dead].mean(), tv_off[no_ball & ~dead].mean(), tv_off[graded & ~no_ball].mean()))


if __name__ == "__main__":
    leave_one_game_out()
    if os.path.isfile(data.SYNC_PATH):
        tv_check()
