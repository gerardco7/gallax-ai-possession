"""Align the TV broadcast to the tactical video, from the tracking alone.

    python possession/broadcast_sync.py          # align, save data/derived/broadcast_sync.npz, report
    python possession/broadcast_sync.py --reuse  # reclassify the saved matches (seconds, not minutes)

No mapping between the two recordings is provided, and the broadcast has cuts, replays and
close-ups. Both trackings are in the same pitch coordinates, though, so a stretch of broadcast is
placed by finding the tactical moment where its players stand in the same places:

  1. Cut the broadcast into shots: runs of valid homography, split wherever the point the camera
     looks at jumps further than a camera can pan in one frame (CUT_JUMP_M).
  2. Split every shot into windows of WINDOW_S and place each window on its own, so a shot that is
     part live and part slow-motion replay loses only the part that does not fit. For every window,
     score every tactical start frame by how far each
     broadcast player is from the nearest tactical player of the same team (truncated at CAP_M so
     a missed or phantom detection costs a bounded amount). Coarse over every 2nd frame from a few
     sample frames, then refined frame by frame over the whole window.
  3. Accept a window whose best score is good and clearly better than any other moment. A window
     that lands before what has already been broadcast live is a replay; the rest are live.

The ball is left out of the matching, so ball agreement afterwards is an independent check that
the alignment is right. Where both recordings see the ball and still disagree by more than
BALL_DISAGREE_M while the players agree, the window is kept and flagged instead: a homography
projects the image onto the ground, so a ball in the air lands on different ground points seen from
two cameras. Rejecting those windows was measured to drop correctly placed ones (their player score
was no worse than the rest), so the flag records it rather than vetoing. The constants are physical bounds, not values fitted to this
match. Everything printed is aggregate.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "starter"))
import frames as data  # noqa: E402
import load_homography  # noqa: E402

FPS = 25
IMAGE_CENTRE = (960.0, 540.0)
CUT_JUMP_M = 3.0      # a broadcast camera following play moves its aim ~1 m per frame at most
MIN_SHOT_S = 2        # shorter shots hold too few frames to place reliably
MIN_PLAYERS = 3       # a sample frame needs at least this many broadcast players
N_SAMPLES = 8
CAP_M = 5.0
COARSE_STEP = 2
REFINE = 3            # frames either side of the coarse best
ACCEPT_M = 1.5        # mean truncated player distance of an accepted match
MIN_MARGIN = 1.3      # best must beat the best moment > 5 s away by this factor
REPLAY_TOLERANCE_S = 2
WINDOW_S = 5
BALL_MIN_FRAMES = 25  # frames where both see the ball before their disagreement counts
BALL_DISAGREE_M = 3.0
FAR = 1e4
OUT = os.path.join(ROOT, "data", "derived", "broadcast_sync.npz")


def camera_aim(matrices, valid, pitch):
    """Pitch point (metres) under the image centre, per frame; NaN without a homography."""
    aim = np.full((len(valid), 2), np.nan)
    inverse = np.linalg.inv(matrices[valid])
    p = inverse @ np.array([IMAGE_CENTRE[0], IMAGE_CENTRE[1], 1.0])
    aim[valid] = p[:, :2] / p[:, 2:] * np.asarray(pitch)
    return aim


def shots(valid, aim):
    """(start, stop) of every continuous camera shot of at least MIN_SHOT_S, stop exclusive."""
    jump = np.zeros(len(valid), bool)
    jump[1:] = np.linalg.norm(np.diff(aim, axis=0), axis=1) > CUT_JUMP_M  # NaN compares False
    is_start = valid.copy()
    is_start[1:] &= ~valid[:-1] | jump[1:]
    starts = np.flatnonzero(is_start)
    invalid = np.flatnonzero(~valid)
    out = []
    for i, a in enumerate(starts):
        next_start = starts[i + 1] if i + 1 < len(starts) else len(valid)
        k = np.searchsorted(invalid, a)
        next_invalid = invalid[k] if k < len(invalid) else len(valid)
        stop = min(next_start, next_invalid)
        if stop - a >= MIN_SHOT_S * FPS:
            out.append((int(a), int(stop)))
    return out


def windows(shot_list):
    """Each shot cut into WINDOW_S pieces; a short tail joins the piece before it."""
    size = WINDOW_S * FPS
    out = []
    for a, b in shot_list:
        edges = list(range(a, b, size)) + [b]
        if len(edges) > 2 and edges[-1] - edges[-2] < size // 2:
            del edges[-2]
        out += list(zip(edges[:-1], edges[1:]))
    return out


def ball_agreement(a, b, t0, tactical, broadcast):
    """(frames where both see the ball, median distance in m) for window [a, b) placed at t0."""
    bb = broadcast["ball"][a:b]
    tb = tactical["ball"][t0:t0 + (b - a)]
    both = ~np.isnan(bb[:, 0]) & ~np.isnan(tb[:, 0])
    if not both.any():
        return 0, np.nan
    return int(both.sum()), float(np.median(np.linalg.norm(bb[both] - tb[both], axis=1)))


def packed(xy):
    """Positions as float32 with missing players moved far away, so a plain min ignores them."""
    return np.nan_to_num(xy.astype(np.float32), nan=FAR)


def score_all(points, tactical):
    """Sum over `points` (k, 2) of the truncated distance to the nearest tactical player, for
    every tactical frame. tactical is (n, p, 2) packed."""
    dx = tactical[:, :, None, 0] - points[None, None, :, 0]
    dy = tactical[:, :, None, 1] - points[None, None, :, 1]
    return np.minimum(np.sqrt(dx * dx + dy * dy).min(1), CAP_M).sum(1)


def visible(xy):
    return xy[~np.isnan(xy[:, 0])].astype(np.float32)


def match_shot(a, b, broadcast, tac_home, tac_away):
    """(tactical start, score, margin) for broadcast shot [a, b), or None if it cannot be placed."""
    usable = [f for f in range(a, b) if
              len(visible(broadcast["home"][f])) + len(visible(broadcast["away"][f])) >= MIN_PLAYERS]
    if len(usable) < 3:
        return None
    samples = [usable[i] for i in np.linspace(0, len(usable) - 1, N_SAMPLES).astype(int)]
    n = len(tac_home)
    coarse_home, coarse_away = tac_home[::COARSE_STEP], tac_away[::COARSE_STEP]
    starts = np.arange(0, n - (b - a), COARSE_STEP)
    total, count = np.zeros(len(starts)), 0
    for f in sorted(set(samples)):
        h, w = visible(broadcast["home"][f]), visible(broadcast["away"][f])
        s = np.zeros(len(coarse_home))
        if len(h):
            s += score_all(h, coarse_home)
        if len(w):
            s += score_all(w, coarse_away)
        shift = (f - a) // COARSE_STEP
        total += s[shift:shift + len(starts)]
        count += len(h) + len(w)
    total /= count
    best = int(np.argmin(total))
    far = np.abs(starts - starts[best]) > 5 * FPS
    margin = total[far].min() / max(total[best], 1e-6) if far.any() else np.inf

    def exact(t0):
        num = den = 0.0
        for f in usable[::max(1, len(usable) // 40)]:
            t = t0 + f - a
            for team, tac in (("home", tac_home), ("away", tac_away)):
                pts = visible(broadcast[team][f])
                if len(pts):
                    num += score_all(pts, tac[t:t + 1])[0]
                    den += len(pts)
        return num / den

    candidates = range(max(0, starts[best] - REFINE), min(n - (b - a), starts[best] + REFINE + 1))
    scored = [(exact(t0), t0) for t0 in candidates]
    score, t0 = min(scored)
    return t0, score, margin


def match_all(tactical, broadcast, pieces):
    """One row per piece: start, stop, tactical start, score, margin, ball frames, ball median."""
    tac_home, tac_away = packed(tactical["home"]), packed(tactical["away"])
    rows = []
    for i, (a, b) in enumerate(pieces):
        m = match_shot(a, b, broadcast, tac_home, tac_away)
        if m:
            rows.append((a, b) + m + ball_agreement(a, b, m[0], tactical, broadcast))
        else:
            rows.append((a, b, -1, np.inf, 0.0, 0, np.nan))
        if (i + 1) % 100 == 0:
            print("  {} / {} windows".format(i + 1, len(pieces)), flush=True)
    return np.array(rows, dtype=float)


def classify(table):
    """(kind, ball_disagrees): kind 0 unplaced, 1 live, 2 replay, per window."""
    ball_disagrees = (table[:, 5] >= BALL_MIN_FRAMES) & (table[:, 6] > BALL_DISAGREE_M)
    accepted = (table[:, 3] < ACCEPT_M) & (table[:, 4] > MIN_MARGIN)
    kind = np.where(accepted, 1, 0)
    live_end = -np.inf
    for i in np.flatnonzero(accepted):
        a, b, t0 = table[i, :3]
        if t0 + (b - a) < live_end - REPLAY_TOLERANCE_S * FPS:
            kind[i] = 2
        else:
            live_end = max(live_end, t0 + (b - a))
    return kind, ball_disagrees


def tactical_to_broadcast(table, kind, n_tactical):
    """For each tactical frame, the live broadcast frame showing it, or -1."""
    out = np.full(n_tactical, -1, dtype=np.int64)
    for (a, b, t0), k in zip(table[:, :3], kind):
        if k == 1:
            t = np.arange(int(t0), int(t0) + int(b - a))
            free = out[t] < 0
            out[t[free]] = np.arange(int(a), int(b))[free]
    return out


def speed_by_piece(broadcast, table):
    """Median player speed (m/s) of the broadcast tracking inside each piece."""
    xy = np.concatenate([broadcast["home"], broadcast["away"]], axis=1)
    v = np.full(xy.shape[:2], np.nan)
    v[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=-1) * FPS
    v[v > 12] = np.nan
    return np.array([np.nanmedian(v[int(a) + 1:int(b)]) for a, b in table[:, :2]])


def report(tactical, broadcast, table, kind, mapping, ball_disagrees):
    graded = tactical["period"] > 0
    t_ball = ~np.isnan(tactical["ball"][:, 0])
    b_ball = ~np.isnan(broadcast["ball"][:, 0])
    covered = mapping >= 0
    shot_frames = (table[:, 1] - table[:, 0])
    print("\n== windows of {} s ==".format(WINDOW_S))
    print("  windows: {}   placed live {}   replays {}   unplaced {}   placed with the ball disagreeing {}".format(
        len(table), int((kind == 1).sum()), int((kind == 2).sum()), int((kind == 0).sum()),
        int((ball_disagrees & (kind > 0)).sum())))
    by_piece = speed_by_piece(broadcast, table)
    print("  median player speed in broadcast windows (m/s): placed {:.2f}   unplaced {:.2f}".format(
              np.nanmedian(by_piece[kind > 0]), np.nanmedian(by_piece[kind == 0]) if (kind == 0).any() else np.nan))
    print("  broadcast frames in windows: {:.1%}   in placed windows: {:.1%}".format(
        shot_frames.sum() / len(b_ball), shot_frames[kind > 0].sum() / len(b_ball)))
    placed = kind > 0
    if placed.any():
        print("  match score of placed windows (m): median {:.2f}   p90 {:.2f}".format(
            np.median(table[placed, 3]), np.percentile(table[placed, 3], 90)))
        print("  replays: {:.1f} min in total".format(shot_frames[kind == 2].sum() / FPS / 60))

    print("\n== independent check: the ball (not used for matching) ==")
    both = covered & t_ball
    both[covered] &= b_ball[mapping[covered]]
    if both.any():
        d = np.linalg.norm(tactical["ball"][both] - broadcast["ball"][mapping[both]], axis=1)
        print("  frames where both see the ball: {}   distance median {:.2f} m   within 2 m {:.1%}".format(
            int(both.sum()), np.median(d), np.mean(d < 2)))

    print("\n== what the broadcast adds ==")
    no_ball = graded & ~t_ball
    fill = no_ball & covered
    fill[covered] &= b_ball[mapping[covered]]
    print("  graded tactical frames shown live by the broadcast: {:.1%}".format(covered[graded].mean()))
    print("  graded frames with no tactical ball: {}   shown live: {:.1%}   broadcast sees the ball: {:.1%}".format(
        int(no_ball.sum()), covered[no_ball].mean(), fill.sum() / no_ball.sum()))
    not_live = graded & ~covered
    print("  graded frames NOT shown live (replay, close-up, cut): {:.1%}   tactical ball missing there: {:.1%}"
          "   vs shown live: {:.1%}".format(not_live.mean() / graded.mean(), (~t_ball[not_live]).mean(),
                                            (~t_ball[graded & covered]).mean()))


if __name__ == "__main__":
    reuse = "--reuse" in sys.argv  # reclassify the saved matches instead of matching again
    print("loading tracking (tactical + broadcast) ...", flush=True)
    tactical = data.load_match(source="tactical")
    broadcast = data.load_match(source="broadcast")
    if reuse:
        table = np.load(OUT)["windows"]
    else:
        matrices, _, valid = load_homography.load(source="broadcast")
        shot_list = shots(valid, camera_aim(matrices, valid, broadcast["pitch"]))
        pieces = windows(shot_list)
        print("aligning {} windows from {} shots ...".format(len(pieces), len(shot_list)), flush=True)
        table = match_all(tactical, broadcast, pieces)
    kind, ball_disagrees = classify(table)
    mapping = tactical_to_broadcast(table, kind, len(tactical["period"]))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, windows=table, kind=kind, ball_disagrees=ball_disagrees, tactical_to_broadcast=mapping)
    print("saved {}".format(os.path.relpath(OUT, ROOT)))
    report(tactical, broadcast, table, kind, mapping, ball_disagrees)
