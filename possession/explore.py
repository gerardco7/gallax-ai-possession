"""Exploratory analysis of any game in the `frames` format (see frames.py).

Every function takes the same dict, so the Metrica notebook and the match notebook run identical
code and their sections can be compared side by side. Nothing here labels anything or is tuned on
anything: it describes.
"""

import os
import warnings

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FPS = 25
MAX_BALL_SPEED = 40.0  # m/s -- the hardest shots are ~35 m/s, so anything faster is a tracking jump


def runs(mask):
    """(starts, stops) of the True runs of a boolean array, stops exclusive."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def playing(frames):
    return frames["period"] > 0


def speed(xy):
    """Speed in m/s from consecutive positions; NaN where either frame is missing."""
    v = np.full(xy.shape[:-1], np.nan)
    v[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=-1) * FPS
    return v


def _nanmean(a, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows are expected
        return np.nanmean(a, axis=axis)


def _nanmin(a, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmin(a, axis=axis)


# -- 1. overview ---------------------------------------------------------------------------------

def summary(frames):
    """One row per half: length, ball presence and how many players are tracked."""
    ball = ~np.isnan(frames["ball"][:, 0])
    home = (~np.isnan(frames["home"][:, :, 0])).sum(1)
    away = (~np.isnan(frames["away"][:, :, 0])).sum(1)
    rows = []
    for half in (1, 2):
        m = frames["period"] == half
        rows.append({"half": half, "frames": int(m.sum()), "minutes": m.sum() / FPS / 60,
                     "ball present %": 100 * ball[m].mean(),
                     "home tracked (mean)": home[m].mean(), "away tracked (mean)": away[m].mean()})
    return pd.DataFrame(rows).set_index("half").round(2)


# -- 2. players ----------------------------------------------------------------------------------

def players_per_frame(frames):
    m = playing(frames)
    fig, ax = plt.subplots(figsize=(8, 3))
    for team in ("home", "away"):
        n = (~np.isnan(frames[team][m, :, 0])).sum(1)
        ax.hist(n, bins=np.arange(-0.5, 16.5), alpha=0.6, label=team)
    ax.set(xlabel="players tracked in a frame", ylabel="frames", title="Players tracked per frame")
    ax.legend()
    return fig


def tracklets(frames):
    """Every continuous stretch a track slot is present: (team, slot, start, seconds)."""
    rows = []
    for team in ("home", "away"):
        present = ~np.isnan(frames[team][:, :, 0])
        for slot in range(present.shape[1]):
            for a, b in zip(*runs(present[:, slot])):
                rows.append((team, slot, a, (b - a) / FPS))
    return pd.DataFrame(rows, columns=["team", "slot", "start", "seconds"])


def plot_tracklets(table):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(table["seconds"], bins=np.logspace(-2, 4, 50))
    ax.set(xscale="log", xlabel="tracklet length (s, log)", ylabel="tracklets",
           title="How long a track slot stays continuous ({} tracklets)".format(len(table)))
    return fig


# -- 3. ball presence ----------------------------------------------------------------------------

def ball_gaps(frames):
    """Stretches of play with no ball: how many, how long, and which lengths hold the missing time."""
    missing = np.isnan(frames["ball"][:, 0]) & playing(frames)
    starts, stops = runs(missing)
    seconds = (stops - starts) / FPS
    buckets = pd.cut(seconds, [0, 0.5, 2, 5, 10, np.inf],
                     labels=["<0.5 s", "0.5-2 s", "2-5 s", "5-10 s", ">10 s"])
    table = pd.DataFrame({"gaps": pd.Series(buckets).value_counts(sort=False),
                          "missing time %": pd.Series(seconds).groupby(buckets, observed=False)
                          .sum() / seconds.sum() * 100}).round(1)
    return starts, stops, table


def plot_ball_presence_over_time(frames, window_s=60):
    present = (~np.isnan(frames["ball"][:, 0])).astype(float)
    present[~playing(frames)] = np.nan
    rolling = pd.Series(present).rolling(window_s * FPS, center=True, min_periods=FPS).mean()
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(np.arange(len(present)) / FPS / 60, 100 * rolling)
    ax.set(xlabel="minute of the video", ylabel="ball present %", ylim=(0, 100),
           title="Ball presence, rolling {} s".format(window_s))
    return fig


def pitch_heatmap(ax, xy, pitch, title, bins=(42, 27)):
    ok = ~np.isnan(xy[:, 0])
    ax.hist2d(xy[ok, 0], xy[ok, 1], bins=bins, range=[[0, pitch[0]], [0, pitch[1]]], cmap="viridis")
    ax.set(title=title, aspect="equal", xticks=[], yticks=[])


def plot_ball_heatmaps(frames, gaps):
    """Where the ball is seen, and where it was last seen before a gap of 1 s or more."""
    starts, stops = gaps
    long_gap = (stops - starts) >= FPS
    last_seen = frames["ball"][np.maximum(starts[long_gap] - 1, 0)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    pitch_heatmap(axes[0], frames["ball"][playing(frames)], frames["pitch"], "Ball seen")
    pitch_heatmap(axes[1], last_seen, frames["pitch"], "Last seen before a gap >= 1 s")
    return fig


# -- 4. ball movement ----------------------------------------------------------------------------

def ball_speed(frames):
    v = speed(frames["ball"])
    v[~playing(frames)] = np.nan
    ok = v[~np.isnan(v)]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(np.clip(ok, 0, 60), bins=120)
    ax.axvline(MAX_BALL_SPEED, color="red", ls="--", label="{} m/s".format(MAX_BALL_SPEED))
    ax.set(yscale="log", xlabel="ball speed (m/s, clipped at 60)", ylabel="frames (log)",
           title="Ball speed from consecutive positions")
    ax.legend()
    stats = pd.Series({"frames with a speed": len(ok), "median m/s": np.median(ok),
                       "p95 m/s": np.percentile(ok, 95),
                       "jumps > {:.0f} m/s %".format(MAX_BALL_SPEED):
                           100 * np.mean(ok > MAX_BALL_SPEED)}).round(2)
    return fig, stats


def ball_out_of_pitch(frames, margin=0.0):
    """Share of frames with a ball whose position lies outside the pitch lines."""
    b = frames["ball"][playing(frames)]
    b = b[~np.isnan(b[:, 0])]
    length, width = frames["pitch"]
    out = (b[:, 0] < -margin) | (b[:, 0] > length + margin) | \
          (b[:, 1] < -margin) | (b[:, 1] > width + margin)
    return 100 * out.mean()


# -- 5. ball and players -------------------------------------------------------------------------

def nearest_distances(frames):
    """Per frame: distance (m) from the ball to the nearest home and nearest away player."""
    ball = frames["ball"][:, None, :]
    return (_nanmin(np.linalg.norm(frames["home"] - ball, axis=2), 1),
            _nanmin(np.linalg.norm(frames["away"] - ball, axis=2), 1))


def plot_nearest(frames, d_home, d_away):
    m = playing(frames) & ~np.isnan(d_home) & ~np.isnan(d_away)
    nearest = np.minimum(d_home[m], d_away[m])
    margin = np.abs(d_home[m] - d_away[m])
    fig, axes = plt.subplots(1, 2, figsize=(12, 3))
    axes[0].hist(np.clip(nearest, 0, 20), bins=80)
    axes[0].set(xlabel="ball to nearest player (m, clipped at 20)", ylabel="frames",
                title="How close is anyone to the ball")
    axes[1].hist(np.clip(margin, 0, 20), bins=80)
    axes[1].set(xlabel="|nearest home - nearest away| (m)", ylabel="frames",
                title="How clear is the nearest team")
    return fig


# -- 6. teams ------------------------------------------------------------------------------------

def team_speed(frames, team):
    """Mean speed (m/s) of a team's tracked players per frame -- low during stoppages."""
    v = speed(frames[team])
    v[v > 12] = np.nan  # faster than a sprinter: an identity switch, not a run
    return _nanmean(v, 1)


def plot_team_speed(frames):
    fig, ax = plt.subplots(figsize=(8, 3))
    m = playing(frames)
    for team in ("home", "away"):
        s = team_speed(frames, team)[m]
        ax.hist(s[~np.isnan(s)], bins=np.linspace(0, 6, 80), alpha=0.6, label=team)
    ax.set(xlabel="mean player speed (m/s)", ylabel="frames", title="How fast each team moves")
    ax.legend()
    return fig


def plot_centroids(frames):
    """Each team's mean x over the match: who is pushed up and who is pinned back."""
    m = playing(frames).astype(float)
    m[m == 0] = np.nan
    fig, ax = plt.subplots(figsize=(10, 3))
    for team in ("home", "away"):
        cx = _nanmean(frames[team][:, :, 0], 1) * m
        ax.plot(np.arange(len(cx)) / FPS / 60,
                pd.Series(cx).rolling(30 * FPS, center=True, min_periods=FPS).mean(), label=team)
    ax.set(xlabel="minute of the video", ylabel="team centroid x (m)", ylim=(0, frames["pitch"][0]),
           title="Team centroid along the pitch, rolling 30 s")
    ax.legend()
    return fig


# -- 7. around restarts --------------------------------------------------------------------------

def around(frames, moments, before_s=15, after_s=10):
    """Mean ball presence and mean player speed in a window around each moment (a frame index).

    The restarts are where a dead ball ends, so this shows what a stoppage looks like in the data
    without needing a single label for it.
    """
    offsets = np.arange(-before_s * FPS, after_s * FPS + 1)
    present = (~np.isnan(frames["ball"][:, 0])).astype(float)
    pace = _nanmean(np.stack([team_speed(frames, "home"), team_speed(frames, "away")]), 0)
    idx = np.asarray(moments)[:, None] + offsets[None, :]
    idx = idx[(idx.min(1) >= 0) & (idx.max(1) < len(present))]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(offsets / FPS, 100 * present[idx].mean(0), label="ball present %")
    ax2 = ax.twinx()
    ax2.plot(offsets / FPS, _nanmean(pace[idx], 0), color="tab:orange", label="player speed")
    ax.axvline(0, color="grey", ls="--")
    ax.set(xlabel="seconds from the restart", ylabel="ball present %", ylim=(0, 100),
           title="Around {} restarts".format(len(idx)))
    ax2.set_ylabel("mean player speed (m/s)")
    fig.legend(loc="lower right")
    return fig


# -- 8. labels (external games only) -------------------------------------------------------------

def label_stats(labels):
    """Durations of possessions and of dead-ball stretches in a labelled game."""
    rows = []
    for label in "HAD":
        a, b = runs(labels == label)
        rows.append({"label": label, "runs": len(a), "median s": np.median(b - a) / FPS,
                     "mean s": np.mean(b - a) / FPS, "share of play %":
                         100 * (labels == label).sum() / (labels != "X").sum()})
    return pd.DataFrame(rows).set_index("label").round(2)


def nearest_vs_label(labels, d_home, d_away, max_m=(1, 2, 3, 5, 10, np.inf)):
    """When the ball is seen during a possession, how often the nearest team is the possessor,
    by how close that nearest player is."""
    alive = np.isin(labels, ["H", "A"]) & ~np.isnan(d_home) & ~np.isnan(d_away)
    nearest = np.where(d_home <= d_away, "H", "A")
    dist = np.minimum(d_home, d_away)
    rows, low = [], 0
    for high in max_m:
        m = alive & (dist >= low) & (dist < high)
        rows.append({"nearest player within": "{}-{} m".format(low, high), "frames": int(m.sum()),
                     "nearest team = possessor %": 100 * np.mean(nearest[m] == labels[m])})
        low = high
    return pd.DataFrame(rows).set_index("nearest player within").round(1)


def ball_presence_by_label(frames, labels):
    present = ~np.isnan(frames["ball"][:, 0])
    return pd.Series({label: 100 * present[labels == label].mean() for label in "HAD"},
                     name="ball present %").round(1)


# -- 9. long ball gaps: stoppage or lost play? ---------------------------------------------------

def distance_to_line(xy, pitch):
    """Distance (m) from each point to the nearest touchline or goal line; negative outside."""
    length, width = pitch
    return np.min(np.stack([xy[:, 0], length - xy[:, 0], xy[:, 1], width - xy[:, 1]]), axis=0)


def gap_table(frames, min_s=5, moments=None, labels=None, before_s=5):
    """One row per stretch of play with no ball lasting at least `min_s`.

    pace_in / pace_before: mean player speed (both teams) inside the gap and in the seconds before
    it -- a stoppage slows everybody down, lost play does not. d_last / d_first: how far from a
    line the ball was last seen before the gap and first seen after it -- a ball that went out is
    lost at a line, and a throw-in or corner starts at one. restart_inside: a restart moment falls
    inside the gap or within 2 s after it. dead_share: the share of the gap labelled D, only for
    labelled (external) games. Meant to be summarised, not listed.
    """
    n = len(frames["period"])
    missing = np.isnan(frames["ball"][:, 0]) & playing(frames)
    starts, stops = runs(missing)
    keep = (stops - starts) >= min_s * FPS
    starts, stops = starts[keep], stops[keep]
    pace = _nanmean(np.stack([team_speed(frames, "home"), team_speed(frames, "away")]), 0)
    ball = frames["ball"]
    last = ball[np.maximum(starts - 1, 0)]
    first = ball[np.minimum(stops, n - 1)]
    table = pd.DataFrame({
        "seconds": (stops - starts) / FPS,
        "pace_in": [_nanmean(pace[a:b], 0) for a, b in zip(starts, stops)],
        "pace_before": [_nanmean(pace[max(a - before_s * FPS, 0):a], 0) for a in starts],
        "d_last": distance_to_line(last, frames["pitch"]),
        "d_first": distance_to_line(first, frames["pitch"]),
    })
    if moments is not None:
        moments = np.sort(np.asarray(moments))
        lo = np.searchsorted(moments, starts)
        hi = np.searchsorted(moments, stops + 2 * FPS, side="right")
        table["restart_inside"] = hi > lo
    if labels is not None:
        table["dead_share"] = [np.mean(labels[a:b] == "D") for a, b in zip(starts, stops)]
    return table


def gap_summary(table, line_m=2.0):
    s = {"long gaps": len(table), "median length s": table["seconds"].median(),
         "pace inside (median m/s)": table["pace_in"].median(),
         "pace before (median m/s)": table["pace_before"].median(),
         "ball lost within {:.0f} m of a line %".format(line_m): 100 * np.mean(table["d_last"] < line_m),
         "ball found within {:.0f} m of a line %".format(line_m): 100 * np.mean(table["d_first"] < line_m)}
    if "restart_inside" in table:
        s["with a restart inside %"] = 100 * table["restart_inside"].mean()
    if "dead_share" in table:
        s["mostly dead (>50% D) %"] = 100 * np.mean(table["dead_share"] > 0.5)
    return pd.Series(s).round(2)


def plot_gaps(frames, table):
    """Player speed inside long gaps against player speed while the ball is seen, and where the
    ball is lost. With labels, also how the speed separates dead gaps from live ones."""
    pace = _nanmean(np.stack([team_speed(frames, "home"), team_speed(frames, "away")]), 0)
    seen = pace[playing(frames) & ~np.isnan(frames["ball"][:, 0])]
    n = 3 if "dead_share" in table else 2
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 3.2))
    bins = np.linspace(0, 4, 40)
    axes[0].hist(seen[~np.isnan(seen)], bins=bins, density=True, alpha=0.5, label="ball seen (frames)")
    axes[0].hist(table["pace_in"].dropna(), bins=bins, density=True, alpha=0.6, label="long gaps (mean)")
    axes[0].set(xlabel="mean player speed (m/s)", ylabel="density", title="Do players slow down in a gap?")
    axes[0].legend()
    axes[1].hist(np.clip(table["d_last"].dropna(), -5, 35), bins=40)
    axes[1].set(xlabel="distance to nearest line (m, <0 outside)", ylabel="gaps",
                title="Where the ball is last seen before a gap")
    if n == 3:
        axes[2].scatter(table["pace_in"], table["dead_share"], s=12, alpha=0.6)
        axes[2].set(xlabel="mean player speed in gap (m/s)", ylabel="share of gap labelled D",
                    title="Speed vs dead ball (labelled games)")
    fig.tight_layout()
    return fig


# -- 10. smart tagging mirrors (match only) ------------------------------------------------------

def tag_mirrors(tags):
    """For each code, the other code most often tagged over the exact same interval, and how often.
    Codes that come in pairs are the same phase seen from each team."""
    key = tags["start_s"].round(2).astype(str) + "-" + tags["end_s"].round(2).astype(str)
    rows = []
    for code, group in tags.groupby("code"):
        partners = tags[key.isin(key[group.index]) & (tags["code"] != code)]["code"].value_counts()
        partner = partners.index[0] if len(partners) else "-"
        share = partners.iloc[0] / len(group) * 100 if len(partners) else 0.0
        rows.append({"code": code, "instances": len(group), "same-interval partner": partner,
                     "share with partner %": round(share, 1)})
    return pd.DataFrame(rows).set_index("code").sort_values("instances", ascending=False)


# -- 11. comparing possession timelines (notebooks) ----------------------------------------------

LABEL_COLOURS = {"H": "tab:blue", "A": "tab:red", "D": "lightgrey", "X": "white"}


def segments(labels, start, stop):
    """[(label, a, b)] maximal runs of one label inside [start, stop), b exclusive."""
    lab = labels[start:stop]
    if not len(lab):
        return []
    e = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    a = np.concatenate(([0], e))
    b = np.concatenate((e, [len(lab)]))
    return [(lab[i], start + i, start + j) for i, j in zip(a, b)]


def plot_timelines(rows, start, stop, title, origin=None):
    """One bar per (name, labels) in rows, coloured H / A / D, between frames start and stop. The
    first row is drawn outlined (the truth); thin lines mark its changes. x in seconds from origin."""

    origin = start if origin is None else origin
    fig, ax = plt.subplots(figsize=(15, 0.62 * len(rows) + 1.2))
    for k, (name, lab) in enumerate(rows):
        y = len(rows) - 1 - k
        for l, a, b in segments(lab, start, stop):
            ax.barh(y, (b - a) / FPS, left=(a - origin) / FPS, height=0.6, color=LABEL_COLOURS[l],
                    edgecolor="black" if k == 0 else "none", linewidth=0.3)
    for _, a, _ in segments(rows[0][1], start, stop)[1:]:
        ax.axvline((a - origin) / FPS, color="black", lw=0.4, alpha=0.4)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([n for n, _ in rows][::-1])
    ax.set_xlim((start - origin) / FPS, (stop - origin) / FPS)
    ax.set_xlabel("seconds")
    ax.set_title(title)
    ax.legend(handles=[Patch(color=LABEL_COLOURS[s], label=n) for s, n in (("H", "home"), ("A", "away"), ("D", "dead ball"))],
              loc="upper right", bbox_to_anchor=(1, 1.22), ncol=3, fontsize=8)
    return fig


def possessions(labels, start, stop):
    return [s for s in segments(labels, start, stop) if s[0] in "HA"]


def _iou(x, y):
    inter = max(0, min(x[2], y[2]) - max(x[1], y[1]))
    return inter / (max(x[2], y[2]) - min(x[1], y[1]))


def found_by_duration(truth, predictions, start, stop, bins=(0, 2, 5, 10, 20, np.inf), threshold=0.5):
    """Share of true possessions each prediction finds (same team, IoU >= threshold), by duration."""
    true_poss = possessions(truth, start, stop)
    dur = np.array([(b - a) / FPS for _, a, b in true_poss])
    labels = ["{}-{} s".format(lo, hi) if np.isfinite(hi) else ">{} s".format(lo) for lo, hi in zip(bins[:-1], bins[1:])]
    bucket = pd.cut(dur, bins, labels=labels, right=False)
    table = pd.DataFrame({"duration": bucket})
    for name, lab in predictions:
        pred = possessions(lab, start, stop)
        table[name] = [max((_iou(t, p) for p in pred if p[0] == t[0]), default=0) >= threshold for t in true_poss]
    out = table.groupby("duration", observed=False).mean().mul(100).round(0)
    out.insert(0, "true possessions", table.groupby("duration", observed=False).size())
    return out


DURATION_BINS = (0, 2, 5, 10, 20, np.inf)


def _duration_bucket(seconds):
    names = ["{}-{} s".format(lo, hi) if np.isfinite(hi) else ">{} s".format(lo)
             for lo, hi in zip(DURATION_BINS[:-1], DURATION_BINS[1:])]
    return pd.cut(np.asarray(seconds), DURATION_BINS, labels=names, right=False)


def miss_table(truth, pred, start, stop, ball_seen=None, threshold=0.5):
    """One row per true possession in [start, stop): found (same team, IoU >= threshold), and if not,
    why -- merged into a longer possession of the same team, its boundaries shifted, given to the
    other team (a missed change), or called dead. Also how it began (after a dead ball or a change),
    the share of its frames with the ball seen, and the start/end error of the best same-team match."""
    true_poss, pred_poss = possessions(truth, start, stop), possessions(pred, start, stop)
    rows = []
    for t in true_poss:
        same = [p for p in pred_poss if p[0] == t[0]]
        best = max(same, key=lambda p: _iou(t, p), default=None)
        iou = _iou(t, best) if best else 0.0
        seg = pred[t[1]:t[2]]
        share = {k: float(np.mean(seg == k)) for k in ("H", "A", "D")}
        other = "A" if t[0] == "H" else "H"
        if iou >= threshold:
            reason = "found"
        elif share[t[0]] >= 0.5:
            reason = "merged" if best and (best[2] - best[1]) > 1.5 * (t[2] - t[1]) else "shifted"
        elif share[other] >= share["D"]:
            reason = "wrong team"
        else:
            reason = "called dead"
        before = truth[t[1] - 1] if t[1] > 0 else "D"
        rows.append({"label": t[0], "start": t[1], "stop": t[2], "seconds": (t[2] - t[1]) / FPS, "iou": iou,
                     "reason": reason, "began": "after dead ball" if before in "DX" else "after a change",
                     "ball_seen": float(np.mean(ball_seen[t[1]:t[2]])) if ball_seen is not None else np.nan,
                     "start_error_s": (best[1] - t[1]) / FPS if best else np.nan,
                     "end_error_s": (best[2] - t[2]) / FPS if best else np.nan})
    out = pd.DataFrame(rows)
    out["duration"] = _duration_bucket(out["seconds"])
    return out


def extra_table(truth, pred, start, stop, threshold=0.5):
    """One row per predicted possession that matches no true one: what the truth is under it -- the
    same team (a split), the other team (an invented change) or a dead ball."""
    true_poss = possessions(truth, start, stop)
    rows = []
    for p in possessions(pred, start, stop):
        if max((_iou(t, p) for t in true_poss if t[0] == p[0]), default=0) >= threshold:
            continue
        seg = truth[p[1]:p[2]]
        share = {k: float(np.mean(seg == k)) for k in ("H", "A", "D")}
        under = max(share, key=share.get)
        rows.append({"label": p[0], "start": p[1], "stop": p[2], "seconds": (p[2] - p[1]) / FPS,
                     "under": "same team" if under == p[0] else ("dead ball" if under == "D" else "other team")})
    out = pd.DataFrame(rows, columns=["label", "start", "stop", "seconds", "under"])
    out["duration"] = _duration_bucket(out["seconds"])
    return out
