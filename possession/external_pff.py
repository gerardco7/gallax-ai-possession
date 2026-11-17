"""PFF FC's 2022 World Cup data (64 games) in our `frames` format, with possession labels.

    python possession/external_pff.py          # convert every game into external/pff_wc2022/frames/
    python possession/external_pff.py 10502    # just these

LICENCE: the files are marked "PFFA Acquisition LLC proprietary information" and no terms of use
were published with them. Confirm with PFF (fchelp@pff.com) that they may be used in the
competition before relying on them. They are read from external/, which is never committed.

Tracking. Broadcast tracking by Sportlogiq at 29.97 fps, resampled to 25 fps by taking the record
nearest in video time. Only the *raw* positions are used. The broadcast camera shows part of the
pitch, so only 2-4 players per team are "VISIBLE" at a time and the rest are estimated from context;
both are kept (the provided match tracks ~10 per team over the whole pitch, and the players near the
ball, which the camera follows, are the visible ones), with which were visible saved beside them.
The ball is kept only where it was seen: an estimated ball is filled in from context, mostly in the
air. The smoothed ball is not used: it is placed on the carrier between events, which is the
corrected-ball artefact the Metrica sample games have. Coordinates are moved
from a pitch-centre origin to a corner origin, in metres.

Labels, from the event file, with the rules used for the Metrica games:
  * an on-the-ball event (OTB: a player in control) gives the ball to that player's team;
  * the ball is dead from an OUT or an END, and before a restart (throw-in, free kick, corner, goal
    kick, kick-off, penalty) from the end of the previous on-the-ball event until the restart, which
    covers fouls and goals, neither of which has an OUT of its own.
"""

import bz2
import json
import multiprocessing
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PFF_DIR = os.path.join(ROOT, "external", "pff_wc2022")
OUT_DIR = os.path.join(PFF_DIR, "frames")
FPS = 25
RESTARTS = {"T", "F", "G", "C", "K", "P"}  # set-piece types of an OTB event that restarts play
MAX_SLOTS = 30                              # jersey numbers 1..30 per team


def games():
    return sorted(int(f.split(".")[0]) for f in os.listdir(os.path.join(PFF_DIR, "Tracking Data"))
                  if f.endswith(".jsonl.bz2"))


def _metadata(game):
    meta = json.load(open(os.path.join(PFF_DIR, "Metadata", "{}.json".format(game))))
    return meta[0] if isinstance(meta, list) else meta


def _pitch(meta):
    pitch = meta["stadium"]["pitches"][0]
    return float(pitch["length"]), float(pitch["width"])


def convert(game):
    """Write external/pff_wc2022/frames/<game>.npz: period, ball, home, away, pitch, labels."""
    meta = _metadata(game)
    length, width = _pitch(meta)
    times, periods, balls, homes, aways = [], [], [], [], []
    with bz2.open(os.path.join(PFF_DIR, "Tracking Data", "{}.jsonl.bz2".format(game)), "rt") as handle:
        for line in handle:
            r = json.loads(line)
            times.append(r["videoTimeMs"] / 1000.0)
            periods.append(r["period"] or 0)
            ball = next((b for b in (r.get("balls") or []) if b.get("visibility") == "VISIBLE" and b.get("x") is not None), None)
            balls.append((ball["x"] + length / 2, ball["y"] + width / 2) if ball else (np.nan, np.nan))
            for team, store in (("homePlayers", homes), ("awayPlayers", aways)):
                row = np.full((MAX_SLOTS, 3), np.nan)
                for p in r.get(team) or []:
                    try:
                        jersey = int(p.get("jerseyNum"))
                    except (TypeError, ValueError):
                        continue
                    if 1 <= jersey <= MAX_SLOTS and p.get("x") is not None:
                        row[jersey - 1] = (p["x"] + length / 2, p["y"] + width / 2,
                                           p.get("visibility") == "VISIBLE")
                store.append(row)
    times = np.asarray(times)
    order = np.argsort(times, kind="stable")
    times = times[order]
    grid = np.arange(times[0], times[-1], 1.0 / FPS)
    nearest = np.clip(np.searchsorted(times, grid), 1, len(times) - 1)
    nearest = np.where(np.abs(times[nearest - 1] - grid) <= np.abs(times[nearest] - grid), nearest - 1, nearest)
    pick = order[nearest]
    home, away = np.asarray(homes)[pick], np.asarray(aways)[pick]
    used = ~np.all(np.isnan(np.concatenate([home, away], axis=1)[:, :, 0]), axis=0)
    hu, au = used[:MAX_SLOTS], used[MAX_SLOTS:]
    period = np.zeros(len(grid), dtype=np.int8)
    for half in (1, 2):
        start, end = meta.get("startPeriod%d" % half), meta.get("endPeriod%d" % half)
        if start is not None and end is not None:
            period[(grid >= start) & (grid <= end)] = half
        else:  # five games' metadata omits the halves; each tracking record carries its own
            period[np.asarray(periods)[pick] == half] = half
    labels = possession_labels(game, grid, period)
    np.savez_compressed(os.path.join(OUT_DIR, "{}.npz".format(game)), period=period,
                        ball=np.asarray(balls)[pick], home=home[:, hu, :2], away=away[:, au, :2],
                        home_visible=home[:, hu, 2] == 1, away_visible=away[:, au, 2] == 1,
                        pitch=np.array([length, width]), labels=labels, video_time=grid)
    return game, len(grid), np.mean(labels != "X")


HOLD_S = 1.0  # second_touch: holding the ball this long confirms a single on-the-ball event


def possession_labels(game, grid, period, second_touch=False):
    """Per frame on the 25 fps grid: 'H'/'A'/'D' inside a half, 'X' outside.

    second_touch=True follows the challenge's definition, where a touch without control does not
    change possession: in live play the other team takes the ball only when its next on-the-ball
    event is also its own, or its player holds the ball HOLD_S or more, and the change is placed at
    the first touch. After a stoppage the first on-the-ball event gives possession as before."""
    events = json.load(open(os.path.join(PFF_DIR, "Event Data", "{}.json".format(game))))
    events = sorted(events, key=lambda e: e["startTime"])
    frame = lambda t: int(np.clip(np.searchsorted(grid, t), 0, len(grid) - 1))
    changes = []
    prev_end, state = None, "D"
    chain = [(e.get("gameEvents") or {}) for e in events]
    for i, e in enumerate(events):
        ge = chain[i]
        kind, setpiece = ge.get("gameEventType"), ge.get("setpieceType")
        team = "H" if ge.get("homeTeam") else "A"
        if kind in ("OUT", "END"):
            changes.append((frame(e["startTime"]), "D"))
            state, prev_end = "D", None
        elif kind in ("FIRSTKICKOFF", "SECONDKICKOFF"):
            changes.append((frame(e["startTime"]), "D"))  # the kick-off's own OTB event gives the ball
            state = "D"
        elif kind == "OTB":
            if setpiece in RESTARTS and state != "D" and prev_end is not None:
                changes.append((frame(prev_end), "D"))
                state = "D"
            end = e.get("endTime") or e["startTime"]
            if second_touch and state not in ("D", team):
                nxt = next((g for g in chain[i + 1:] if g.get("gameEventType") in ("OTB", "OUT", "END")), {})
                confirmed = end - e["startTime"] >= HOLD_S or (nxt.get("gameEventType") == "OTB"
                                                               and ("H" if nxt.get("homeTeam") else "A") == team)
                if not confirmed:
                    prev_end = end
                    continue
            changes.append((frame(e["startTime"]), team))
            state, prev_end = team, end
    labels = np.full(len(grid), "X", dtype="<U1")
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


def load(game, labels="events"):
    """(frames, labels) in the same format as frames.load_metrica. labels="second_touch" gives the
    labels with possession_labels(second_touch=True), cached beside the converted game."""
    z = np.load(os.path.join(OUT_DIR, "{}.npz".format(game)))
    frames = {"period": z["period"], "ball": z["ball"], "home": z["home"], "away": z["away"],
              "pitch": tuple(z["pitch"]), "game": "pff{}".format(game), "real": True}
    if labels == "events":
        return frames, z["labels"]
    path = os.path.join(OUT_DIR, "{}_second_touch.npy".format(game))
    if not os.path.isfile(path):
        np.save(path, possession_labels(game, z["video_time"], z["period"], second_touch=True))
    return frames, np.load(path)


def event_touches(game):
    """(frames, teams) of the touches the events record: the start of every on-the-ball event (a
    player takes control) and its end (he releases the ball), with the player's team (0 home)."""
    grid = np.load(os.path.join(OUT_DIR, "{}.npz".format(game)))["video_time"]
    events = json.load(open(os.path.join(PFF_DIR, "Event Data", "{}.json".format(game))))
    frames, teams = [], []
    for e in events:
        ge = e.get("gameEvents") or {}
        if ge.get("gameEventType") != "OTB":
            continue
        team = 0 if ge.get("homeTeam") else 1
        for t in (e["startTime"], e.get("endTime")):
            if t is not None:
                frames.append(int(np.clip(np.searchsorted(grid, t), 0, len(grid) - 1)))
                teams.append(team)
    order = np.argsort(frames, kind="stable")
    return np.asarray(frames)[order], np.asarray(teams)[order]


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    todo = [int(g) for g in sys.argv[1:]] or games()
    with multiprocessing.Pool(min(12, len(todo))) as pool:
        for game, n, played in pool.imap_unordered(convert, todo):
            print("  {}: {} frames, {:.0%} in play".format(game, n, played), flush=True)
