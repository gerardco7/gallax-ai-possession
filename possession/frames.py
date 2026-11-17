"""One data format for every game, so a method runs unchanged on Metrica's samples and on ours.

A `frames` dict holds, for n frames at 25 fps:
    period  (n,)       1 or 2 while a half is being played, 0 otherwise
    ball    (n, 2)     metres, NaN where the ball is not tracked
    home    (n, p, 2)  metres, one column per track slot of the home team, NaN where absent
    away    (n, q, 2)  the same for the away team
    pitch   (length, width) in metres

Row i is video frame i (the ATD's frame_id - 1), the index a submission uses.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import DEFAULT_SOURCE, GAME, GAMES, half_video_blocks, require  # noqa: E402

import external_metrica  # noqa: E402

METRICA_PITCH = (105.0, 68.0)

# The realistic copy of an external game. Generic choices about automatic ball tracking, none
# measured on the provided match: the ball is lost in blocks of play rather than frame by frame,
# it is lost a little before a stoppage rather than seen leaving the pitch, and a detection never
# sits exactly on the carrier the way the sample data's corrected ball does.
DEGRADE_LIVE_SHARE = 0.15      # share of possession time whose ball is removed
DEGRADE_BLOCK_S = (4, 16)      # length range of a removed block
DEGRADE_ERODE_S = 3            # a natural gap starts up to this much earlier
DEGRADE_NOISE_M = 0.7          # per axis, ~0.9 m in distance
DEGRADE_NOISE_TAU_S = 1.0      # the error drifts slowly, as a tracker's does, rather than jittering every frame
DEGRADE_SMOOTH_S = 0.4        # an automatic tracker filters the ball: kinks at touches come out rounded
DEGRADE_DEAD_VISIBLE = 0.7     # share of stoppages whose ball is seen, placed where play restarts...
DEGRADE_DEAD_VISIBLE_PART = (0.3, 1.0)  # ...for this final part of the stoppage
DEGRADE_DESCRIPTION = ("ball removed in {}-{} s blocks over {:.0%} of play, natural gaps start up to {} s "
                       "earlier, +/-{} m noise drifting over ~{} s, {:.0%} of stoppages show the ball placed at the restart, ball smoothed over {} s").format(
    *DEGRADE_BLOCK_S, DEGRADE_LIVE_SHARE, DEGRADE_ERODE_S, DEGRADE_NOISE_M, DEGRADE_NOISE_TAU_S, DEGRADE_DEAD_VISIBLE,
    DEGRADE_SMOOTH_S)


def moving_average(xy, width):
    """Centred moving average over `width` frames, per axis; frames with no value stay missing."""
    out = pd.DataFrame(xy).rolling(width, center=True, min_periods=width // 2 + 1).mean().to_numpy(copy=True)
    out[np.isnan(xy[:, 0])] = np.nan
    return out


def smooth_noise(rng, shape, sigma, tau_s):
    """Gaussian noise of standard deviation sigma whose value drifts with time constant tau_s
    (a first-order autoregressive process), instead of being drawn afresh every frame."""
    a = np.exp(-1.0 / (tau_s * 25))
    shocks = rng.normal(0, sigma * np.sqrt(1 - a * a), shape)
    out = np.empty(shape)
    out[0] = rng.normal(0, sigma, shape[1:])
    for i in range(1, shape[0]):
        out[i] = a * out[i - 1] + shocks[i]
    return out  # the sample data's normalised coordinates span a 105 x 68 m pitch


def _metres(xy, pitch):
    return xy * np.asarray(pitch, dtype=np.float64)


def load_metrica(game):
    """(frames, truth) for one Metrica sample game; truth is the event-derived label array."""
    t = external_metrica.load_tracking(game)
    frames = {
        "period": t["period"].astype(np.int8),
        "ball": _metres(t["ball"], METRICA_PITCH),
        "home": _metres(t["home"], METRICA_PITCH),
        "away": _metres(t["away"], METRICA_PITCH),
        "pitch": METRICA_PITCH,
        "game": game,
    }
    truth = external_metrica.possession_labels(external_metrica.load_events(game), t["period"])
    return frames, truth


SYNC_PATH = os.path.join(ROOT, "data", "derived", "broadcast_sync.npz")


def add_broadcast(frames, game=GAME):
    """Put the aligned broadcast on the tactical timeline (see broadcast_sync.py):
        broadcast_live      (n,)    a live broadcast frame shows this tactical frame
        broadcast_ball      (n, 2)  the broadcast's ball there, NaN where it has none
        broadcast_ball_ok   (n,)    False inside windows where the two balls disagree
    """
    sync = np.load(SYNC_PATH)
    mapping = sync["tactical_to_broadcast"]
    live = mapping >= 0
    ball = np.full_like(frames["ball"], np.nan)
    ball[live] = load_match(game, source="broadcast")["ball"][mapping[live]]
    ok = np.ones(len(live), dtype=bool)
    for (a, b, t0), kind, bad in zip(sync["windows"][:, :3], sync["kind"], sync["ball_disagrees"]):
        if kind == 1 and bad:
            ok[int(t0):int(t0) + int(b - a)] = False
    frames.update(broadcast_live=live, broadcast_ball=ball, broadcast_ball_ok=ok)
    return frames


def degrade(frames, truth, seed=0):
    """A copy of an external game whose ball is as hard to follow as an automatic tracker's.

    Only the input changes, never the labels. Removed blocks start on frames with a possession, so
    the ball taken away is ball an algorithm would have needed.
    """
    rng = np.random.default_rng(seed)
    ball = frames["ball"].copy()
    # A stoppage's ball is usually still visible: somebody holds it or places it where play
    # restarts. The external games remove it instead, so put it back, still, at the restart point.
    dead = (truth == "D") & (frames["period"] > 0)
    edges = np.diff(np.concatenate(([0], dead.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        seen_after = np.flatnonzero(~np.isnan(frames["ball"][b:b + 2 * 25, 0]))
        if rng.random() < DEGRADE_DEAD_VISIBLE and len(seen_after):
            k = int((b - a) * rng.uniform(*DEGRADE_DEAD_VISIBLE_PART))
            ball[b - k:b] = frames["ball"][b + seen_after[0]]
    ball = ball + smooth_noise(rng, ball.shape, DEGRADE_NOISE_M, DEGRADE_NOISE_TAU_S)
    ball = moving_average(ball, int(round(DEGRADE_SMOOTH_S * 25)))
    missing = np.isnan(ball[:, 0]).astype(np.int8)
    for start in np.flatnonzero(np.diff(np.concatenate(([0], missing))) == 1):
        ball[max(0, start - int(rng.uniform(0, DEGRADE_ERODE_S) * 25)):start] = np.nan
    alive = np.flatnonzero(np.isin(truth, ["H", "A"]))
    budget = int(len(alive) * DEGRADE_LIVE_SHARE)
    while budget > 0:
        length = int(rng.uniform(*DEGRADE_BLOCK_S) * 25)
        start = rng.choice(alive)
        ball[start:start + length] = np.nan
        budget -= length
    return dict(frames, ball=ball)


def with_broadcast_ball(frames):
    """The tactical ball, filled from the live broadcast where only the broadcast sees it and the
    two balls do not disagree in that window (needs frames.add_broadcast)."""
    ball = frames["ball"].copy()
    fill = (np.isnan(ball[:, 0]) & frames["broadcast_live"] & frames["broadcast_ball_ok"]
            & ~np.isnan(frames["broadcast_ball"][:, 0]))
    ball[fill] = frames["broadcast_ball"][fill]
    return dict(frames, ball=ball)


def training_copies(frames, truth, seeds):
    """The versions of an external game to train on: as it is, plus realistic copies -- unless the
    game's tracking is already automatic and uncorrected (frames["real"], the PFF games), in which
    case degrading it further would only simulate defects it already has."""
    if frames.get("real"):
        return [frames]
    return [frames] + [degrade(frames, truth, seed=s) for s in seeds]


def load_match(game=GAME, source=DEFAULT_SOURCE):
    """The provided match's ATD in the same format. No labels: there are none to have.

    source="broadcast" reads the TV recording's ATD, which is on its own timeline: its `period`
    is 1 everywhere, because which of its frames fall in which half is not known until it has
    been aligned to the tactical video (broadcast_sync.py).
    """
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, source=source)
    root = ET.parse(meta_path).getroot()
    home_id = [team.get("id") for team in root.iter("Team")][0]  # first team is home (EPTS)
    is_home = np.array([p.get("teamId") == home_id for p in root.iter("Player")])

    # Each line is "frame:x,y,s;x,y,s;...:x,y,s" -- one flat row once both separators are commas.
    with open(raw_path) as handle:
        table = np.loadtxt((line.replace(":", ",").replace(";", ",") for line in handle),
                           delimiter=",", dtype=np.float32)
    n_slots = len(is_home)
    if table.shape[1] != 1 + 3 * n_slots + 3:
        raise ValueError("unexpected ATD width {} for {} slots".format(table.shape[1], n_slots))

    pitch = GAMES[game]["pitch"]
    players = _metres(table[:, 1:1 + 3 * n_slots].reshape(-1, n_slots, 3)[:, :, :2], pitch)
    period = np.zeros(len(table), dtype=np.int8)
    if source == "tactical":
        for half, first, last in half_video_blocks(game):
            period[first:last + 1] = half
    else:
        period[:] = 1
    return {
        "period": period,
        "ball": _metres(table[:, -3:-1], pitch),
        "home": players[:, is_home],
        "away": players[:, ~is_home],
        "pitch": pitch,
    }
