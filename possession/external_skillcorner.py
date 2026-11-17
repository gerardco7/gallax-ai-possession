"""SkillCorner open data (MIT): 20 Australian A-League 2024/25 games of broadcast tracking.

    git clone --depth 1 https://github.com/SkillCorner/opendata external/skillcorner
    # the tracking files are Git LFS objects; without git-lfs, fetch each from
    # https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches/<id>/<id>_tracking_extrapolated.jsonl
    python possession/external_skillcorner.py    # writes external/skillcorner/frames/<id>.npz

Tracking at 10 fps, resampled to our 25 fps grid by linear interpolation. The ball only where
SkillCorner detected it (like PFF's visible ball); players detected or extrapolated (like PFF's
estimated ones). Labels follow external_pff.possession_labels(second_touch=True) on SkillCorner's
player possessions (dynamic events): in live play the other team takes the ball only when its next
player possession is also its own, or the player held it HOLD_S or more; after a stoppage the first
possession gives the ball. Dead ball from the moment the ball leaves the pitch after a possession
followed by a game interruption (the end of that possession if the tracking never shows it out), and
from the end of the last possession before a restart.
"""

import csv
import json
import multiprocessing
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from external_pff import HOLD_S  # noqa: E402

SC_DIR = os.path.join(os.path.dirname(HERE), "external", "skillcorner")
MATCH_DIR = os.path.join(SC_DIR, "data", "matches")
OUT_DIR = os.path.join(SC_DIR, "frames")
FPS, SRC_FPS = 25, 10


def games():
    return sorted(int(d) for d in os.listdir(MATCH_DIR) if d.isdigit())


def _read(game):
    base = os.path.join(MATCH_DIR, str(game), str(game))
    meta = json.load(open(base + "_match.json"))
    home_id = meta["home_team"]["id"]
    side = {p["id"]: ("home" if p["team_id"] == home_id else "away") for p in meta["players"]}
    records = [json.loads(line) for line in open(base + "_tracking_extrapolated.jsonl")]
    events = [e for e in csv.DictReader(open(base + "_dynamic_events.csv")) if e["event_type"] == "player_possession"]
    return meta, side, home_id, records, events


def _tracks(records, side, length, width):
    n = max(r["frame"] for r in records) + 1
    period = np.zeros(n, dtype=np.int8)
    ball = np.full((n, 2), np.nan)
    slots = {"home": {}, "away": {}}
    for r in records:
        for p in r["player_data"]:
            s = side.get(p["player_id"])
            if s is not None:
                slots[s].setdefault(p["player_id"], len(slots[s]))
    teams = {s: np.full((n, max(len(v), 1), 2), np.nan) for s, v in slots.items()}
    for r in records:
        f = r["frame"]
        period[f] = r["period"] or 0
        b = r["ball_data"]
        if b["is_detected"] and b["x"] is not None:
            ball[f] = (b["x"] + length / 2, b["y"] + width / 2)
        for p in r["player_data"]:
            s = side.get(p["player_id"])
            if s is not None and p["x"] is not None:
                teams[s][f, slots[s][p["player_id"]]] = (p["x"] + length / 2, p["y"] + width / 2)
    any_ball = np.full(n, np.nan)  # detected or extrapolated, only to see the ball leave the pitch
    for r in records:
        b = r["ball_data"]
        if b["x"] is not None:
            any_ball[r["frame"]] = max(abs(b["x"]) - length / 2, abs(b["y"]) - width / 2)
    return period, ball, teams["home"], teams["away"], any_ball


def labels_10fps(events, home_id, period, out_margin):
    """Per 10 fps frame: 'H'/'A'/'D' inside a half, 'X' outside (see the module docstring)."""
    pps = sorted(events, key=lambda e: int(e["frame_start"]))
    team_of = lambda e: "H" if int(e["team_id"]) == home_id else "A"
    changes, state, prev_end = [], "D", None
    for i, e in enumerate(pps):
        start, end, team = int(e["frame_start"]), int(e["frame_end"]), team_of(e)
        if e["game_interruption_before"] and state != "D" and prev_end is not None:
            changes.append((prev_end, "D"))
            state = "D"
        after = bool(e["game_interruption_after"])
        confirmed = True
        if state not in ("D", team):
            nxt = pps[i + 1] if i + 1 < len(pps) else None
            confirmed = (end - start) / SRC_FPS >= HOLD_S or (not after and nxt is not None and team_of(nxt) == team
                                                               and not nxt["game_interruption_before"])
        if confirmed:
            changes.append((start, team))
            state = team
        prev_end = end
        if after:
            stop = int(pps[i + 1]["frame_start"]) if i + 1 < len(pps) else len(period)
            out = np.flatnonzero(out_margin[end:stop] > 0)
            changes.append((end + int(out[0]) if len(out) else end, "D"))
            state, prev_end = "D", None
    labels = np.full(len(period), "X", dtype="<U1")
    for half in (1, 2):
        rows = np.flatnonzero(period == half)
        if not len(rows):
            continue
        first, last = rows[0], rows[-1]
        inside = [(first, "D")] + sorted((c for c in changes if first <= c[0] <= last), key=lambda c: c[0])
        for (begin, label), (stop, _) in zip(inside, inside[1:] + [(last + 1, None)]):
            if begin < stop:
                labels[begin:stop] = label
    return labels


def convert(game):
    meta, side, home_id, records, events = _read(game)
    length, width = float(meta["pitch_length"]), float(meta["pitch_width"])
    period, ball, home, away, out_margin = _tracks(records, side, length, width)
    labels = labels_10fps(events, home_id, period, out_margin)
    t = np.arange(0, len(period) - 1, SRC_FPS / FPS)  # 25 fps grid in 10 fps frame units
    i0 = np.floor(t).astype(int)
    w = (t - i0)[:, None]
    lerp = lambda x: (1 - w.reshape((-1,) + (1,) * (x.ndim - 1))) * x[i0] + w.reshape((-1,) + (1,) * (x.ndim - 1)) * x[i0 + 1]
    nearest = np.rint(t).astype(int)
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(OUT_DIR, "{}.npz".format(game)), period=period[nearest], ball=lerp(ball),
                        home=lerp(home), away=lerp(away), pitch=np.array([length, width]), labels=labels[nearest])
    return game, len(t), float(np.mean(labels != "X"))


def load(game):
    """(frames, second-touch labels) in the format of external_pff.load."""
    z = np.load(os.path.join(OUT_DIR, "{}.npz".format(game)))
    return ({"period": z["period"], "ball": z["ball"], "home": z["home"], "away": z["away"],
             "pitch": tuple(z["pitch"]), "game": "sc{}".format(game), "real": True}, z["labels"])


if __name__ == "__main__":
    with multiprocessing.Pool(4) as pool:
        for game, n, share in pool.imap_unordered(convert, games()):
            print(game, n, "frames, {:.0%} in play".format(share), flush=True)
