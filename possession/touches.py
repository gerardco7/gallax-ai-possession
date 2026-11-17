"""Possession from detected touches: a third method, built on what the ball does, not where it is.

    python possession/touches.py      # touch detection quality on the external games, leave-one-game-out

When somebody touches the ball -- a pass, a reception, a clearance, a shot -- its velocity changes
abruptly. The tracker's error drifts slowly, so a touch is a sharp change on top of it. Two logistic
models trained on the external games read each frame: "is there a touch here?" from the change of
the ball's velocity at three time scales and from who is near it, and "whose touch?" from how near
each team's nearest player is. Touches are the peaks of the first. Possession is then built from
them exactly as the external labels are built from events: the team of the last touch has the ball,
and the stoppage detectors of gaps.py say when it is dead.

Training labels come from the external games' events: the start of every pass, shot, recovery,
loss, carry and set piece, and the end of every completed pass (the reception), with the team that
touched. Nothing about the provided match is used.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import control  # noqa: E402
import explore as ex  # noqa: E402
import external_metrica  # noqa: E402
import frames as data  # noqa: E402
import external_pff  # noqa: E402
import gaps  # noqa: E402

FPS = 25
SCALES = (2, 4, 8)          # frames before and after a frame over which the ball's velocity is measured
TOUCH_TYPES = {"PASS", "SHOT", "RECOVERY", "BALL LOST", "CARRY", "SET PIECE"}
POSITIVE_TOL = 2            # frames either side of an event frame labelled as a touch
NEGATIVE_GAP = 6            # frames from any event before a frame may serve as "no touch"
NEGATIVE_EVERY = 10         # negatives are subsampled: they outnumber touches ~40 to 1
THRESHOLD = 0.85            # chosen on the realistic bench (0.480; 0.95 gives 0.491 but relies on sharper kinks than the match ball has)
MIN_SEPARATION_S = 0.3      # two touches closer than this are one
MATCH_TOL_S = 0.2           # a detected touch this close to a true one, same team, is a hit

TOUCH_FEATURES = (["speed_before_%d" % k for k in SCALES] + ["speed_after_%d" % k for k in SCALES]
                  + ["dv_%d" % k for k in SCALES] + ["turn_%d" % k for k in SCALES]
                  + ["near", "crowd", "seen"])
TEAM_FEATURES = ["diff", "prox_home", "prox_away", "crowd_home", "crowd_away"]


def event_touches(events, n):
    """(frames, teams) of every touch the events record; team 0 home, 1 away."""
    ev = events.sort_values("Start Frame", kind="stable").reset_index(drop=True)
    team = np.where(ev["Team"] == "Home", control.HOME, control.AWAY)
    frames, teams = [], []
    for i, row in ev.iterrows():
        if row["Type"] in TOUCH_TYPES:
            frames.append(int(row["Start Frame"]) - 1)
            teams.append(team[i])
        if row["Type"] == "PASS" and row["End Frame"] and i + 1 < len(ev) and team[i + 1] == team[i]:
            frames.append(int(row["End Frame"]) - 1)  # the reception
            teams.append(team[i])
    frames, teams = np.array(frames), np.array(teams)
    keep = (frames >= 0) & (frames < n)
    order = np.argsort(frames[keep], kind="stable")
    return frames[keep][order], teams[keep][order]


def touch_features(frames):
    ball = frames["ball"]
    n = len(ball)
    cols = {"seen": (~np.isnan(ball[:, 0])).astype(float)}
    for k in SCALES:
        before = np.full((n, 2), np.nan)
        after = np.full((n, 2), np.nan)
        before[k:] = (ball[k:] - ball[:-k]) * FPS / k
        after[:-k] = (ball[k:] - ball[:-k]) * FPS / k
        sb, sa = np.linalg.norm(before, axis=1), np.linalg.norm(after, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos = np.sum(before * after, axis=1) / (sb * sa)
        cols["speed_before_%d" % k] = np.clip(sb, 0, 40)
        cols["speed_after_%d" % k] = np.clip(sa, 0, 40)
        cols["dv_%d" % k] = np.clip(np.linalg.norm(after - before, axis=1), 0, 60)
        cols["turn_%d" % k] = np.arccos(np.clip(cos, -1, 1))
    d_home, d_away = ex.nearest_distances(frames)
    b = ball[:, None, :]
    with np.errstate(invalid="ignore"):
        crowd_home = (np.linalg.norm(frames["home"] - b, axis=2) < 2).sum(1).astype(float)
        crowd_away = (np.linalg.norm(frames["away"] - b, axis=2) < 2).sum(1).astype(float)
    cols.update({
        "near": np.clip(np.fmin(d_home, d_away), 0, 15),
        "crowd": crowd_home + crowd_away,
        "crowd_home": crowd_home, "crowd_away": crowd_away,
        "diff": np.clip(d_away - d_home, -10, 10),
        "prox_home": np.nan_to_num(np.exp(-d_home / 1.5)),
        "prox_away": np.nan_to_num(np.exp(-d_away / 1.5)),
    })
    return pd.DataFrame(cols)


class TouchModel:
    def fit(self, games, stoppages=None):
        self.stoppages = stoppages or gaps.train(games)
        xs, ys, ts = [], [], []
        for frames, truth in games:
            if str(frames["game"]).startswith("pff"):
                tf, tt = external_pff.event_touches(int(frames["game"][3:]))
            else:
                tf, tt = event_touches(external_metrica.load_events(frames["game"]), len(truth))
            label = np.full(len(truth), -1)  # -1 negative, -2 ignored, else team
            near_event = np.zeros(len(truth), bool)
            for f in tf:
                near_event[max(0, f - NEGATIVE_GAP):f + NEGATIVE_GAP + 1] = True
            label[near_event] = -2
            for f, t in zip(tf, tt):
                label[max(0, f - POSITIVE_TOL):f + POSITIVE_TOL + 1] = t
            pos = np.flatnonzero(label >= 0)
            neg = np.flatnonzero((label == -1) & (frames["period"] > 0))[::NEGATIVE_EVERY]
            keep = np.concatenate([pos, neg])
            for copy in data.training_copies(frames, truth, gaps.TRAINING_SEEDS):
                xs.append(touch_features(copy).iloc[keep])
                ys.append(label[keep] >= 0)
                ts.append(label[keep])
        x, y, t = pd.concat(xs, ignore_index=True), np.concatenate(ys), np.concatenate(ts)
        self.touch = gaps.GapModel(TOUCH_FEATURES).fit(x, y)
        self.team = gaps.GapModel(TEAM_FEATURES).fit(x[y], t[y] == control.HOME)
        return self

    def detect(self, frames, threshold=None):
        """(frames, teams) of detected touches."""
        peaks, p_home = self.detect_soft(frames, threshold)
        return peaks, np.where(p_home > 0.5, control.HOME, control.AWAY)

    def detect_soft(self, frames, threshold=None):
        """(frames, probability that each touch is the home team's) of detected touches."""
        threshold = THRESHOLD if threshold is None else threshold
        x = touch_features(frames)
        p = np.nan_to_num(self.touch.predict(x))
        p[frames["period"] == 0] = 0
        sep = int(MIN_SEPARATION_S * FPS)
        peaks = []
        for i in np.flatnonzero(p > threshold):
            lo, hi = max(0, i - sep), min(len(p), i + sep + 1)
            if p[i] == p[lo:hi].max() and (not peaks or i - peaks[-1] > sep):
                peaks.append(i)
        peaks = np.array(peaks, dtype=int)
        return peaks, (self.team.predict(x.iloc[peaks]) if len(peaks) else np.array([]))

    def method(self, broadcast_ball=False, threshold=None):
        def run(frames):
            if broadcast_ball:
                frames = data.with_broadcast_ball(frames)
            peaks, teams = self.detect(frames, threshold)
            return possession_from_touches(frames, peaks, teams, gaps.dead_mask(frames, self.stoppages))
        run.__name__ = "touches_sync" if broadcast_ball else "touches"
        return run

    def both(self, frames):
        """(labels, probabilities) from one detection and one stoppage computation."""
        peaks, p_home = self.detect_soft(frames)
        probs = gaps.kind_probabilities(frames, self.stoppages)
        dead = gaps.decide(probs)
        teams = np.where(p_home > 0.5, control.HOME, control.AWAY)
        return (possession_from_touches(frames, peaks, teams, dead),
                probabilities_from_touches(frames, peaks, p_home, self.stoppages, (np.max(list(probs.values()), axis=0), dead)))

    def prob_method(self, broadcast_ball=False):
        """(n, 3) probability of H, A, D: the team confidence of the last touch, scaled by the
        stoppage detectors' probability that the ball is alive."""
        def run(frames):
            if broadcast_ball:
                frames = data.with_broadcast_ball(frames)
            peaks, p_home = self.detect_soft(frames)
            return probabilities_from_touches(frames, peaks, p_home, self.stoppages)
        return run


def possession_from_touches(frames, peaks, teams, dead):
    """The team of the last touch has the ball; dead frames are dead; after a stoppage the first
    touch's team has it from the end of the stoppage."""
    n = len(frames["period"])
    team_at = np.full(n, control.NOBODY, dtype=np.int8)
    team_at[peaks] = teams
    out = np.full(n, "X", dtype="<U1")
    state, open_from = control.NOBODY, 0
    for i in range(n):
        if i and frames["period"][i] != frames["period"][i - 1]:
            state, open_from = control.NOBODY, i
        if frames["period"][i] == 0 or dead[i]:
            out[i] = "X" if frames["period"][i] == 0 else "D"
            state, open_from = control.NOBODY, i + 1
            continue
        if team_at[i] != control.NOBODY and team_at[i] != state:
            if state == control.NOBODY:
                out[open_from:i] = "H" if team_at[i] == control.HOME else "A"
            state = team_at[i]
        out[i] = "D" if state == control.NOBODY else ("H" if state == control.HOME else "A")
    return out


def probabilities_from_touches(frames, peaks, p_home, stoppages, dead_info=None):
    """Per frame [P(H), P(A), P(D)] from the last touch's team confidence; after a stoppage, the
    first touch's confidence reaches back to its end, as in possession_from_touches. dead_info:
    (stoppage probability, dead mask) when already computed."""
    n = len(frames["period"])
    p_dead, dead = dead_info if dead_info is not None else (gaps.dead_probability(frames, stoppages),
                                                            gaps.dead_mask(frames, stoppages))
    conf = np.full(n, np.nan)
    conf[peaks] = p_home
    home = np.full(n, 0.5)
    last, open_from = np.nan, 0
    for i in range(n):
        if i and frames["period"][i] != frames["period"][i - 1]:
            last, open_from = np.nan, i
        if frames["period"][i] == 0 or dead[i]:
            last, open_from = np.nan, i + 1
            continue
        if not np.isnan(conf[i]):
            if np.isnan(last):
                home[open_from:i] = conf[i]
            last = conf[i]
        if not np.isnan(last):
            home[i] = last
    out = np.stack([(1 - p_dead) * home, (1 - p_dead) * (1 - home), p_dead], axis=1)
    out[frames["period"] == 0] = 0
    return out


def detection_scores(found, true, tol):
    """Precision and recall of detected touches (frame, team) against true ones."""
    (pf, pt), (tf, tt) = found, true
    used = np.zeros(len(tf), bool)
    hits = 0
    for f, t in zip(pf, pt):
        cand = np.flatnonzero(~used & (np.abs(tf - f) <= tol) & (tt == t))
        if len(cand):
            used[cand[np.argmin(np.abs(tf[cand] - f))]] = True
            hits += 1
    return hits / max(len(pf), 1), hits / max(len(tf), 1)


if __name__ == "__main__":
    games = {g: data.load_metrica(g) for g in external_metrica.GAMES}
    rows = []
    for test in games:
        model = TouchModel().fit([games[g] for g in games if g != test])
        fr, tr = games[test]
        true = event_touches(external_metrica.load_events(test), len(tr))
        for bench_name, f in (("clean", fr), ("realistic", data.degrade(fr, tr, seed=test))):
            for thr in (0.3, 0.5, 0.7, 0.85):
                found = model.detect(f, thr)
                prec, rec = detection_scores(found, true, int(MATCH_TOL_S * FPS))
                rows.append({"bench": bench_name, "threshold": thr, "game": test, "true touches": len(true[0]),
                             "detected": len(found[0]), "precision %": 100 * prec, "recall %": 100 * rec})
    print(pd.DataFrame(rows).groupby(["bench", "threshold"])[["true touches", "detected", "precision %", "recall %"]]
          .mean().round(1).to_string())
