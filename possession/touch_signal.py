"""Is the ball's acceleration a usable touch signal? Its peaks against the touches PFF's events record.

    python possession/touch_signal.py

On the 64 World Cup games: peaks of temporal._ball_change's acceleration (local maxima over a
threshold, at most one per 0.2 s), on the raw ball and on the ball smoothed first (Savitzky-Golay,
7 frames, order 2), against the start and end of every on-the-ball event. Precision and recall
are set beside the same peaks moved 2 s later, which is what chance gives at the events' density,
and at a matched peak the team of the nearest player is checked against the event's. Aggregates.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402
import external_pff as ep  # noqa: E402
import temporal  # noqa: E402
THRESH = (10, 20, 30, 50, 80)
TOL = (5, 10)  # frames: 0.2 s, 0.4 s

def savgol(window=7, order=2):
    """Least-squares polynomial smoothing weights for the centre of a window."""
    x = np.arange(window) - window // 2
    return np.linalg.pinv(np.vander(x, order + 1, increasing=True))[0]


def smooth(ball, window=7):
    out = ball.copy()
    w = savgol(window)
    seen = ~np.isnan(ball[:, 0])
    edges = np.flatnonzero(np.diff(np.r_[0, seen.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a >= window:
            h = window // 2
            for d in range(2):  # the ends of a seen stretch keep their raw positions
                out[a + h:b - h, d] = np.convolve(ball[a:b, d], w[::-1], mode="valid")
    return out

def peaks(accel, thr):
    m = accel >= thr
    local = np.r_[False, (accel[1:-1] >= accel[:-2]) & (accel[1:-1] >= accel[2:]), False]
    idx = np.flatnonzero(m & local)
    keep, last = [], -99
    for i in idx:  # one peak per 0.2 s
        if i - last > 5: keep.append(i); last = i
    return np.array(keep, int)

def near(a, b, tol):
    """for each a, is there a b within tol"""
    if not len(b): return np.zeros(len(a), bool)
    j = np.clip(np.searchsorted(b, a), 1, len(b) - 1)
    return np.minimum(np.abs(b[j - 1] - a), np.abs(b[j] - a)) <= tol

def one(g):
    frames, _ = ep.load(g)
    ball = frames["ball"]
    seen = ~np.isnan(ball[:, 0])
    tf, tt = ep.event_touches(g)
    tf, first = np.unique(tf, return_index=True)
    tt = tt[first]
    touch_seen = seen[tf]
    rows = []
    for name, b in (("raw", ball), ("smoothed", smooth(ball))):
        accel, _ = temporal._ball_change(b)
        d = [np.nanmin(np.linalg.norm(frames[t] - b[:, None, :], axis=2), axis=1) for t in ("home", "away")]
        for thr in THRESH:
            p = peaks(accel, thr)
            for tol in TOL:
                rec = near(tf, p, tol)
                prec = near(p, tf, tol)
                # team of the nearest player at matched peaks vs the event's team
                j = np.clip(np.searchsorted(tf, p), 1, len(tf) - 1)
                k = np.where(np.abs(tf[j - 1] - p) <= np.abs(tf[j] - p), j - 1, j)
                ok = prec & ~np.isnan(d[0][p]) & ~np.isnan(d[1][p])
                team_right = (np.where(d[0][p] <= d[1][p], 0, 1) == tt[k])[ok]
                ps = p + 50  # the same peaks 2 s later: what chance gives with this density
                ps = ps[ps < len(seen)]
                ps = ps[seen[ps]]
                rows.append({"shift_found": int((near(tf, ps, tol) & touch_seen).sum()), "shift_true": int(near(ps, tf, tol).sum()), "shift_peaks": len(ps),
                             "game": g, "ball": name, "thr": thr, "tol_s": tol / 25, "touches": len(tf),
                             "touches_seen": int(touch_seen.sum()), "found": int(rec.sum()), "found_seen": int((rec & touch_seen).sum()),
                             "peaks": len(p), "true_peaks": int(prec.sum()), "team_n": len(team_right), "team_right": int(team_right.sum()),
                             "seen_minutes": seen[frames["period"] > 0].sum() / 1500})
    return rows

if __name__ == "__main__":
    df = pd.DataFrame([r for rs in bench.map_games(one, ep.games()) for r in rs])
    g = df.groupby(["ball", "tol_s", "thr"]).sum(numeric_only=True)
    out = pd.DataFrame({
        "recall (all touches)": g.found / g.touches,
        "recall (ball seen)": g.found_seen / g.touches_seen,
        "precision": g.true_peaks / g.peaks,
        "peaks per seen min": g.peaks / g.seen_minutes,
        "precision by chance": g.shift_true / g.shift_peaks,
        "recall seen by chance": g.shift_found / g.touches_seen,
        "nearest team right": g.team_right / g.team_n}).round(3)
    print("touches per game {:.0f}, with the ball seen {:.0%}".format(df.touches.iloc[0::20].mean(), (g.touches_seen / g.touches).iloc[0]))
    print("event touches per seen minute: {:.1f}".format((g.touches_seen / g.seen_minutes).iloc[0]))
    print(out.to_string())
