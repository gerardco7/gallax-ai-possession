"""Metrica Sports' public sample games, turned into possession labels for tuning.

    python possession/external_metrica.py            # download games 1-2, write labels
    python possession/external_metrica.py --selftest

The rules require every tuned parameter to be tuned on external data. These are two full matches
with tracking (players and ball, 25 fps, 0-1 coordinates -- the same conventions as the provided
ATD) and human event data, from which a per-frame H/A/D label is derived mechanically below.

LICENCE: github.com/metrica-sports/sample-data declares no licence. Confirm with the organisers
(Metrica is one of them) before relying on it, and document the answer. The data is downloaded
on demand into external/ and never committed.

How the labels are derived (the possession rules of CHALLENGE.md, applied to events):
  * a controlled action -- PASS, SET PIECE, SHOT, RECOVERY, BALL LOST -- gives its team the ball
    from its start frame. BALL LOST counts because the team had the ball when it was lost.
  * CHALLENGE never changes possession: a duel or touch without control does not count, and a
    loose ball stays with the last team that had it.
  * the ball is dead from a BALL OUT, a shot that went out or in, an offside or a foul (FAULT
    RECEIVED) until the next controlled action, which is the restart.
  * CARRY (game 3 only) is a controlled action too.
These are event-based, while the challenge's ground truth comes from corrected tracking, so the
two will disagree by a few frames around every change. Good enough to tune durations and rates;
not a frame-exact stand-in.
"""

import argparse
import json
import os
import sys
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXTERNAL_DIR = os.path.join(ROOT, "external", "metrica")
BASE_URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
GAMES = (1, 2, 3)
EPTS_GAMES = (3,)  # game 3 ships as FIFA EPTS tracking with JSON events instead of CSVs

HOLD_S = 1.0  # as external_pff.HOLD_S
OWNER_EVENTS = {"PASS", "SET PIECE", "SHOT", "RECOVERY", "BALL LOST", "CARRY"}
TEAM_LABEL = {"Home": "H", "Away": "A"}


def _name(game, kind):
    return "Sample_Game_{0}_{1}.csv".format(game, kind)


def download(game, dest=EXTERNAL_DIR):
    """The files of one sample game, fetched once."""
    folder = os.path.join(dest, "Sample_Game_{}".format(game))
    os.makedirs(folder, exist_ok=True)
    if game in EPTS_GAMES:
        names = {k: "Sample_Game_{}_{}".format(game, k) for k in ("metadata.xml", "tracking.txt", "events.json")}
    else:
        names = {k: _name(game, k) for k in
                 ("RawEventsData", "RawTrackingData_Home_Team", "RawTrackingData_Away_Team")}
    paths = {}
    for kind, name in names.items():
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            url = "{}/Sample_Game_{}/{}".format(BASE_URL, game, name)
            print("  downloading {}".format(url))
            urllib.request.urlretrieve(url, path)
        paths[kind] = path
    return paths


def _load_epts(paths):
    """EPTS tracking: one line per frame, 22 players then the ball, each "x,y". Which player sits in
    which slot changes with every substitution, so each DataFormatSpecification block says it."""
    root = ET.parse(paths["metadata.xml"]).getroot()
    home_id = [team.get("id") for team in root.iter("Team")][0]
    team_of_player = {p.get("id"): p.get("teamId") for p in root.iter("Player")}
    player_of_channel = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    params = {p.findtext("Name"): p.findtext("Value") for p in root.iter("ProviderParameter")}
    with open(paths["tracking.txt"]) as handle:
        table = np.loadtxt((line.replace(":", ",").replace(";", ",") for line in handle),
                           delimiter=",", dtype=np.float64)
    frame, xy = table[:, 0].astype(int), table[:, 1:].reshape(len(table), -1, 2)
    n_players = xy.shape[1] - 1
    home = np.full((len(table), n_players, 2), np.nan)
    away = np.full((len(table), n_players, 2), np.nan)
    for spec in root.iter("DataFormatSpecification"):
        refs = [r.get("playerChannelId") for r in spec.iter("PlayerChannelRef")][::2]
        is_home = np.array([team_of_player[player_of_channel[r]] == home_id for r in refs])
        rows = (frame >= int(spec.get("startFrame"))) & (frame <= int(spec.get("endFrame")))
        home[rows, :is_home.sum()] = xy[rows][:, :n_players][:, is_home]
        away[rows, :(~is_home).sum()] = xy[rows][:, :n_players][:, ~is_home]
    period = np.zeros(len(table), dtype=np.int64)
    for half, key in ((1, "first_half"), (2, "second_half")):
        period[(frame >= int(params[key + "_start"])) & (frame <= int(params[key + "_end"]))] = half
    keep = np.isfinite(home).any(axis=(0, 2)) | np.isfinite(away).any(axis=(0, 2))
    return period, frame, home[:, keep[:home.shape[1]]], away[:, keep[:away.shape[1]]], xy[:, -1]


def _events_from_json(paths):
    """The JSON events as the same table the CSV games give, subtypes joined with '-'."""
    root = ET.parse(paths["metadata.xml"]).getroot()
    home_name = [team.findtext("Name") for team in root.iter("Team")][0]
    rows = []
    for e in json.load(open(paths["events.json"]))["data"]:
        sub = e["subtypes"]
        sub = "-".join(s["name"] for s in sub) if isinstance(sub, list) else (sub or {}).get("name")
        rows.append({"Team": "Home" if e["team"]["name"] == home_name else "Away",
                     "Type": e["type"]["name"], "Subtype": sub, "Period": e["period"],
                     "Start Frame": e["start"]["frame"], "End Frame": e["end"]["frame"] or 0})
    return pd.DataFrame(rows)


def _read_team(path):
    """(period, frame, players (n, p, 2), ball (n, 2)) from one Metrica tracking CSV.

    The file has three header rows; the third names Period, Frame, Time, then one column per
    player x (its y is the unnamed column after it), and Ball last.
    """
    table = pd.read_csv(path, skiprows=2)
    xy = table.iloc[:, 3:].to_numpy(dtype=np.float64).reshape(len(table), -1, 2)
    return table["Period"].to_numpy(), table["Frame"].to_numpy(), xy[:, :-1], xy[:, -1]


def load_tracking(game):
    """Both teams and the ball on one frame index. Missing positions are NaN, as in the ATD."""
    paths = download(game)
    if game in EPTS_GAMES:
        period, frame, home, away, ball = _load_epts(paths)
    else:
        period, frame, home, ball = _read_team(paths["RawTrackingData_Home_Team"])
        period_a, frame_a, away, _ = _read_team(paths["RawTrackingData_Away_Team"])
        if not (np.array_equal(frame, frame_a) and np.array_equal(period, period_a)):
            raise ValueError("game {}: home and away tracking are not on the same frames".format(game))
    if not np.array_equal(frame, np.arange(1, len(frame) + 1)):
        raise ValueError("game {}: tracking frames are not 1..n".format(game))
    return {"period": period, "frame": frame, "home": home, "away": away, "ball": ball}


def load_events(game):
    paths = download(game)
    return _events_from_json(paths) if game in EPTS_GAMES else pd.read_csv(paths["RawEventsData"])


def _is_dead_ball(event_type, subtype):
    subtype = subtype if isinstance(subtype, str) else ""
    if event_type in ("BALL OUT", "FAULT RECEIVED") or "OFFSIDE" in subtype:
        return True
    return event_type == "SHOT" and ("OUT" in subtype or "GOAL" in subtype)


def possession_labels(events, period, second_touch=False):
    """Per-frame 'H'/'A'/'D' for every tracking frame, 'X' where no period is being played.

    Row i is tracking frame i + 1, the same off-by-one the provided ATD has. second_touch=True applies
    external_pff's rule: in live play the other team takes the ball only when its next owner event is
    also its own (before any dead ball), or the event lasts HOLD_S or more.
    """
    labels = np.full(len(period), "X", dtype="<U1")
    for half in np.unique(period[period > 0]):
        rows = np.flatnonzero(period == half)
        first, last = rows[0], rows[-1]
        changes = [(first, "D")]  # before the kickoff the ball is dead
        for event in events[events["Period"] == half].itertuples(index=False):
            start = int(event[events.columns.get_loc("Start Frame")]) - 1
            end = int(event[events.columns.get_loc("End Frame")]) - 1
            kind, subtype, team = event.Type, event.Subtype, event.Team
            if _is_dead_ball(kind, subtype):
                changes.append((max(start, end), "D"))
            elif kind in OWNER_EVENTS:
                changes.append((start, TEAM_LABEL[team]))
                if second_touch:
                    changes[-1] += (end,)
        if second_touch:
            changes = _second_touch(changes, events[events["Period"] == half])
        changes.sort(key=lambda change: change[0])  # stable: same-frame order is file order
        for (begin, label), (stop, _) in zip(changes, changes[1:] + [(last + 1, None)]):
            begin, stop = max(begin, first), min(stop, last + 1)
            if begin < stop:
                labels[begin:stop] = label
    return labels


def _second_touch(changes, events):
    """Drop live-play team changes not confirmed by the team's next owner event or a 1 s hold."""
    ordered = sorted(changes, key=lambda change: change[0])
    out, state = [], "D"
    for i, change in enumerate(ordered):
        label = change[1]
        if label == "D":
            out.append(change[:2])
            state = "D"
            continue
        if state not in ("D", label):
            nxt = next((c for c in ordered[i + 1:]), None)
            held = len(change) > 2 and change[2] - change[0] >= HOLD_S * 25
            if not held and not (nxt is not None and nxt[1] == label):
                continue
        out.append(change[:2])
        state = label
    return out


def spells(labels):
    """[(label, start, stop)] maximal runs of H or A -- the possessions spellF1 counts."""
    edges = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [len(labels)]))
    return [(labels[a], a, b) for a, b in zip(starts, stops) if labels[a] in ("H", "A")]


def write_labels(path, labels):
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))


def selftest():
    events = pd.DataFrame(
        [("Home", "SET PIECE", "KICK OFF", 1, 3, 0),
         ("Home", "PASS", np.nan, 1, 3, 5),
         ("Away", "CHALLENGE", "AERIAL-WON", 1, 6, 6),   # no control: stays H
         ("Away", "RECOVERY", "INTERCEPTION", 1, 8, 8),
         ("Away", "BALL OUT", np.nan, 1, 10, 11),        # dead from frame 11
         ("Home", "SET PIECE", "THROW IN", 1, 14, 0)],
        columns=["Team", "Type", "Subtype", "Period", "Start Frame", "End Frame"])
    period = np.array([1] * 16 + [0] * 2)
    got = "".join(possession_labels(events, period))
    assert got == "DDHHHHHAAADDDHHHXX", got
    assert [s[0] for s in spells(np.array(list(got)))] == ["H", "A", "H"]
    print("selftest ok")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        sys.exit()

    for game in GAMES:
        tracking = load_tracking(game)
        labels = possession_labels(load_events(game), tracking["period"])
        out = os.path.join(EXTERNAL_DIR, "Sample_Game_{}_possession.csv".format(game))
        write_labels(out, labels)
        played = labels != "X"
        runs = spells(labels)
        durations = np.array([stop - start for _, start, stop in runs]) / 25.0
        ball_seen = np.mean(~np.isnan(tracking["ball"][played, 0]))
        print("game {}: {} frames played  H {:.1%}  A {:.1%}  D {:.1%}  ball tracked {:.1%}".format(
            game, played.sum(), *(np.mean(labels[played] == c) for c in "HAD"), ball_seen))
        print("        {} possessions, mean {:.1f} s, median {:.1f} s  -> {}".format(
            len(runs), durations.mean(), np.median(durations), os.path.relpath(out, ROOT)))
