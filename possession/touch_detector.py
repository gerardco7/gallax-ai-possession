"""A learned touch detector: per 25 fps frame, does a player of H or of A touch the ball?

    python possession/touch_detector.py                          # World Cup only, its own 25 fps
    python possession/touch_detector.py --sources wc+sc --res10  # World Cup + SkillCorner, both at 10 fps

Supervised by PFF's on-the-ball events, which are in sync with the tracking (docs/experiments.md, "Are
the World Cup labels in sync"): the start of every OTB event is a touch by that team, and so is its end
when the player held the ball 0.4 s or more (the release); each marks +-2 frames. Input: per-frame
features made for contact (the ball's speed, acceleration and turn; per team the two nearest players'
distances, how fast the nearest closes in, its speed relative to the ball, players within 1 and 2 m).
Model: a small TCN at 25 fps seeing +-2.5 s, trained on the folds of bench_temporal.py, home/away
swapped at random. Out-of-fold probabilities are cached under external/pff_wc2022/touch/.

With --sources wc+sc it also learns from SkillCorner's player possessions (each one's start, and its end
when held RELEASE_S or more), whose ball is seen in live play as often as the match's; fold k holds out
SkillCorner games [k::4] as bench_temporal_sc.py does. SkillCorner marks no deflections or failed
challenges, so its touches are fewer than PFF's. --res10 puts the World Cup through SkillCorner's 10 fps
(sampled every 2.5 frames, interpolated back) so both sources show the ball's changes alike; the match
would get the same treatment.

Evaluated as detection (peaks matched to true touches within +-3 frames) against a hand-made rule
(ball acceleration peaks given to the nearest team within 1.5 m), and as a kill test: around the short
possessions the possession ensemble misses with the ball seen, does it see that team's touches as
often as around the short ones it finds? Aggregates only.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import explore  # noqa: E402
import external_pff as ep  # noqa: E402
import external_skillcorner as sc  # noqa: E402
import periods  # noqa: E402
import temporal  # noqa: E402
from bench_pff import folds  # noqa: E402

FPS = 25
SOURCES = sys.argv[sys.argv.index("--sources") + 1] if "--sources" in sys.argv else "wc"
RES10 = "--res10" in sys.argv
CACHE = os.path.join(ep.PFF_DIR, "touch", SOURCES + ("_res10" if RES10 else ""))
SC_RUNS = [os.path.join(ep.PFF_DIR, "sc_runs", "tcn{}_second_touch_sw2_channels128_sc1".format(s), "oof.npz")
           for s in ["", "_s1", "_s2", "_s3", "_s4"]]  # the submission's networks, out of fold on both datasets
SUB_COST = 7.0
MARK = 2          # frames either side of an event that count as its touch
RELEASE_S = 0.4   # an event's end is a second touch (the release) when the ball was held this long
CROP, BATCH, STEPS = 750, 32, 60
MAX_EPOCHS, PATIENCE = 80, 8
HP = {"channels": 64, "layers": 6, "kernel": 3, "dropout": 0.1}
NB, NT = 5, 7     # ball features, per-team features


def _nearest(players, ball):
    with np.errstate(invalid="ignore"):
        d = np.linalg.norm(players - ball[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    order = np.argsort(d, axis=1)
    return d, order


def touch_features(frames):
    """(n_frames, NB + 2 NT): ball block, home block, away block."""
    ball = frames["ball"]
    n = len(ball)
    seen = ~np.isnan(ball[:, 0])
    length, width = frames["pitch"]
    accel, turn = temporal._ball_change(ball)
    vb = np.full((n, 2), np.nan)
    vb[1:-1] = (ball[2:] - ball[:-2]) * FPS / 2
    speed = np.linalg.norm(vb, axis=1)
    out = [seen.astype(float), np.nan_to_num(np.clip(speed, 0, 40) / 10), np.clip(np.nan_to_num(accel) / 50, 0, 2),
           np.nan_to_num(turn), np.where(seen, np.abs(ball[:, 1] - width / 2) / (width / 2), 0)]
    for team in ("home", "away"):
        xy = frames[team]
        d, order = _nearest(xy, ball)
        rows = np.arange(n)
        d1, d2 = d[rows, order[:, 0]], d[rows, order[:, 1]] if xy.shape[1] > 1 else np.full(n, np.inf)
        p = order[:, 0]
        dp_before, dp_after = np.full(n, np.nan), np.full(n, np.nan)  # the same player 2 frames before and after
        dp_before[2:] = d[rows[:-2], p[2:]]
        dp_after[:-2] = d[rows[2:], p[:-2]]
        closing = (dp_before - dp_after) * FPS / 4
        vp = np.full((n, 2), np.nan)
        vp[1:-1] = (xy[rows[2:], p[1:-1]] - xy[rows[:-2], p[1:-1]]) * FPS / 2
        rel = np.linalg.norm(vp - vb, axis=1)
        with np.errstate(invalid="ignore"):
            closing = np.where(np.isfinite(closing), closing, 0)
            block = [np.clip(np.where(np.isfinite(d1), d1, 10), 0, 10) / 5, np.clip(np.where(np.isfinite(d2), d2, 10), 0, 10) / 5,
                     np.clip(closing, -10, 10) / 5,
                     np.clip(np.nan_to_num(rel, nan=10.0), 0, 20) / 10,
                     (d < 1).sum(1) / 2.0, (d < 2).sum(1) / 3.0, (np.isfinite(d1) & seen).astype(float)]
        out += block
    return np.nan_to_num(np.stack(out, axis=1)).astype(np.float32)


def touch_targets(game, grid, n):
    """(n,) 0 none, 1 a touch by H, 2 by A."""
    events = json.load(open(os.path.join(ep.PFF_DIR, "Event Data", "{}.json".format(game))))
    y = np.zeros(n, dtype=np.int64)
    for e in events:
        ge = e.get("gameEvents") or {}
        if ge.get("gameEventType") != "OTB" or e.get("startTime") is None:
            continue
        c = 1 if ge.get("homeTeam") else 2
        times = [e["startTime"]] + ([e["endTime"]] if e.get("endTime") and e["endTime"] - e["startTime"] >= RELEASE_S else [])
        for t in times:
            f = int(np.searchsorted(grid, t))
            y[max(f - MARK, 0):min(f + MARK + 1, n)] = c
    return y


def _lerp(x, t):
    i0 = np.clip(np.floor(t).astype(int), 0, len(x) - 2)
    w = (t - i0).reshape((-1,) + (1,) * (x.ndim - 1))
    return (1 - w) * x[i0] + w * x[i0 + 1]


def at_10fps(frames):
    """The same frames as SkillCorner would give them: sampled at 10 fps, interpolated back to 25."""
    n = len(frames["period"])
    src = np.arange(0, n - 1, FPS / 10)
    back = np.clip(np.arange(n) * 10 / FPS, 0, len(src) - 1)
    out = dict(frames)
    for k in ("ball", "home", "away"):
        out[k] = _lerp(_lerp(frames[k], src), back)
    return out


def sc_targets(game, n):
    """SkillCorner's player possessions as touches: each start, and each end held RELEASE_S or more."""
    meta, _, home_id, _, events = sc._read(game)
    y = np.zeros(n, dtype=np.int64)
    for e in events:
        c = 1 if int(e["team_id"]) == home_id else 2
        a, b = int(e["frame_start"]), int(e["frame_end"])
        for f10 in [a] + ([b] if (b - a) / sc.SRC_FPS >= RELEASE_S else []):
            f = int(round(f10 * FPS / sc.SRC_FPS))
            y[max(f - MARK, 0):min(f + MARK + 1, n)] = c
    return y


def load_game(key):
    """(frames, second-touch labels, touch targets) of ("wc", game) or ("sc", game)."""
    source, game = key
    if source == "sc":
        frames, labels = sc.load(game)
        return frames, labels, sc_targets(game, len(labels))
    z = np.load(os.path.join(ep.OUT_DIR, "{}.npz".format(game)))
    frames, labels = ep.load(game, "second_touch")
    return (at_10fps(frames) if RES10 else frames), labels, touch_targets(game, z["video_time"], len(labels))


def game_data(key):
    frames, _, y = load_game(key)
    x = touch_features(frames)
    halves = [np.flatnonzero(frames["period"] == h) for h in (1, 2)]
    return [(x[r[0]:r[-1] + 1], y[r[0]:r[-1] + 1]) for r in halves if len(r) > CROP]


class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        c = HP["channels"]
        self.inp = nn.Conv1d(NB + 2 * NT, c, 1)
        self.blocks = nn.Sequential(*[temporal.Block(c, 2 ** i, HP["kernel"], HP["dropout"]) for i in range(HP["layers"])])
        self.out = nn.Conv1d(c, 3, 1)

    def forward(self, x):
        return self.out(self.blocks(self.inp(x.transpose(1, 2)))).transpose(1, 2)


SWAP = list(range(NB)) + list(range(NB + NT, NB + 2 * NT)) + list(range(NB, NB + NT))


def _crops(seqs, rng, device):
    for _ in range(STEPS):
        picks = rng.integers(0, len(seqs), BATCH)
        xs, ys = [], []
        for i in picks:
            x, y = seqs[i]
            a = int(rng.integers(0, len(x) - CROP))
            x, y = x[a:a + CROP], y[a:a + CROP]
            if rng.random() < 0.5:
                x, y = x[:, SWAP], np.where(y == 0, 0, 3 - y)
            xs.append(x)
            ys.append(y)
        yield torch.tensor(np.stack(xs), device=device), torch.tensor(np.stack(ys), device=device)


def predict(model, x):
    with torch.no_grad():
        p = torch.softmax(model(torch.tensor(x[None], device=temporal._device())), -1)[0].cpu().numpy()
    return p


def train(fit, val, log=print):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    device = temporal._device()
    model = Detector().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    best, best_state, waited = np.inf, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for x, y in _crops(fit, rng, device):
            loss = nn.functional.cross_entropy(model(x).reshape(-1, 3), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = np.mean([nn.functional.cross_entropy(model(torch.tensor(x[None], device=device))[0], torch.tensor(y, device=device)).item()
                         for x, y in val])
        if v < best - 1e-4:
            best, waited, best_state = v, 0, {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            waited += 1
        if waited >= PATIENCE:
            break
    log("    stopped at epoch {} (best validation loss {:.4f})".format(epoch, best))
    model.load_state_dict(best_state)
    model.eval()
    return model


def peaks(p, threshold, window=5):
    """Frames where p is the maximum of +-window and at least threshold, at most one per window."""
    pad = np.r_[np.full(window, -1.0), p, np.full(window, -1.0)]
    local = np.lib.stride_tricks.sliding_window_view(pad, 2 * window + 1).max(1)
    out, last = [], -10 ** 9
    for f in np.flatnonzero((p >= threshold) & (p >= local)):
        if f - last > window:
            out.append(f)
            last = f
    return np.array(out, dtype=int)


def detection(events, found_peaks, tol=3):
    """Precision and recall of peaks against true touch frames (same team), within tol frames."""
    if not len(events) or not len(found_peaks):
        return 0, len(found_peaks), 0
    hits = np.abs(found_peaks[:, None] - events[None]) <= tol
    return int(hits.any(1).sum()), len(found_peaks), int(hits.any(0).sum())


def rule_scores(frames):
    """The hand-made rule as a score per team: ball acceleration where that team's nearest is within 1.5 m."""
    accel, _ = temporal._ball_change(frames["ball"])
    out = []
    for team in ("home", "away"):
        d, order = _nearest(frames[team], frames["ball"])
        d1 = d[np.arange(len(d)), order[:, 0]]
        out.append(np.where(d1 < 1.5, np.clip(np.nan_to_num(accel) / 50, 0, 1), 0))
    return np.stack(out, 1)


if __name__ == "__main__":
    os.makedirs(CACHE, exist_ok=True)
    wc, scg = ep.games(), sc.games()
    keys = [("wc", g) for g in wc] + [("sc", g) for g in scg]
    t0 = time.time()
    data = dict(zip(keys, bench.map_games(game_data, keys, processes=4)))
    print("data ready in {:.0f} s; touch frames WC {:.1%}, SC {:.1%}".format(time.time() - t0,
          *[np.mean([np.mean(y > 0) for k, v in data.items() if k[0] == s for _, y in v]) for s in ("wc", "sc")]), flush=True)
    for k, test, _ in folds(wc):
        sc_test = scg[k::4]
        held = [("wc", g) for g in test] + [("sc", g) for g in sc_test]
        if all(os.path.isfile(os.path.join(CACHE, "{}_{}.npz".format(*h))) for h in held):
            continue
        rest = [g for g in wc if g not in test]
        val, fit = [("wc", g) for g in rest[::8]], [("wc", g) for i, g in enumerate(rest) if i % 8]
        if "sc" in SOURCES:
            sc_rest = [g for g in scg if g not in sc_test]
            val += [("sc", g) for g in sc_rest[::8]]
            fit += [("sc", g) for i, g in enumerate(sc_rest) if i % 8]
        model = train([s for g in fit for s in data[g]], [s for g in val for s in data[g]])
        for h in held:
            np.savez_compressed(os.path.join(CACHE, "{}_{}.npz".format(*h)), p=predict(model, touch_features(load_game(h)[0])).astype(np.float16))
        torch.save(model.state_dict(), os.path.join(CACHE, "fold{}.pt".format(k)))
        print("fold {} done".format(k), flush=True)

    oofs = [dict(np.load(p)) for p in SC_RUNS]  # read before forking: workers sharing one open npz corrupt its reads

    def possession_probs(key, frames):
        name = "{}_{}_sc".format("worldcup" if key[0] == "wc" else "skillcorner", key[1])
        per = frames["period"][periods.blocks(frames["period"], temporal.STEP)[2]]
        return np.mean([z[name] for z in oofs], axis=0), per

    def evaluate(key):
        frames, truth, y = load_game(key)
        p = np.load(os.path.join(CACHE, "{}_{}.npz".format(*key)))["p"].astype(float)
        seen = ~np.isnan(frames["ball"][:, 0])
        rule = rule_scores(frames)
        res = {}
        for c in (1, 2):
            ev = np.flatnonzero((y == c) & (np.r_[0, y[:-1]] != c)) + MARK  # each touch's centre
            ev = ev[ev < len(seen)]
            for name, score, ths in (("detector", p[:, c], (0.3, 0.5)), ("rule", rule[:, c - 1], (0.2, 0.4))):
                for th in ths:
                    pk = peaks(score, th)
                    for part, mask in (("seen", seen), ("unseen", ~seen)):
                        tp_p, n_p, tp_e = detection(ev[mask[ev]], pk[mask[pk]] if len(pk) else pk)
                        key2 = (key[0], name, th, part)
                        res[key2] = res.get(key2, np.zeros(4)) + np.array([tp_p, n_p, tp_e, mask[ev].sum()])
        # kill test: short true possessions with the ball seen, found or missed by the submission's networks
        pred = temporal.decode_steps(frames, *possession_probs(key, frames), SUB_COST)
        m = explore.miss_table(truth, pred, 0, len(truth), seen)
        rows = []
        for r in m[(m.seconds < 5) & (m.ball_seen > 0.75)].itertuples():
            c = 1 if r.label == "H" else 2
            rows.append({"source": key[0], "reason": r.reason, "first": p[max(r.start - 5, 0):r.start + 25, c].max(),
                         "n_touch": len(peaks(p[r.start:r.stop, c], 0.5)),
                         "other": p[max(r.start - 5, 0):r.start + 25, 3 - c].max()})
        return res, rows

    out = bench.map_games(evaluate, keys, processes=4)
    total = {}
    for res, _ in out:
        for k, v in res.items():
            total[k] = total.get(k, 0) + v
    count = {"wc": len(wc), "sc": len(scg)}
    print("\n{}: touch detection on held-out games, both teams, within +-3 frames (ball seen / not at the touch):".format(os.path.basename(CACHE)))
    for (src, name, th, part), (tp_p, n_p, tp_e, n_e) in sorted(total.items()):
        n = count[src]
        print("  {} {:<8} th {:.1f} {:<6}: touches {:5.0f}/game, peaks {:5.0f}/game, precision {:.2f}, recall {:.2f}".format(
            src, name, th, part, n_e / n, n_p / n, tp_p / max(n_p, 1), tp_e / max(n_e, 1)))
    k = pd.DataFrame([r for _, rows in out for r in rows])
    k["group"] = np.where(k.reason == "found", "found", np.where(k.reason.isin(["wrong team", "merged"]), "missed (team/merged)", "missed (other)"))
    print("\nkill test, short (< 5 s) true possessions with the ball seen > 75%, per game and by outcome:")
    print(k.groupby(["source", "group"]).agg(per_game=("first", "size"), first_touch_p50=("first", lambda s: (s >= 0.5).mean()),
                                             first_touch_median=("first", "median"), two_touches=("n_touch", lambda s: (s >= 2).mean()),
                                             other_team_first=("other", lambda s: (s >= 0.5).mean())).assign(
        per_game=lambda t: t.per_game / [count[s] for s in t.index.get_level_values(0)]).round(2).to_string())
