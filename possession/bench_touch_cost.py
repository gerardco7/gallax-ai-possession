"""Option B: a change of possession costs less where the ball is touched by that team.

    python possession/bench_touch_cost.py

Decode only, on the submitted ensemble's out-of-fold probabilities (5 seeds per World Cup fold,
second-touch labels; cached under external/pff_wc2022/oof/). A touch is a peak of the smoothed ball's
acceleration (touch_signal.py) whose nearest player is within R m and belongs to that team. Entering
H at a step costs lam * exp(-gamma) when H touched within one step (0.2 s) of it, lam otherwise; the
same for A, and entering D always costs lam. gamma = 0 is the submitted decoder. lam, gamma, the
acceleration threshold and R are chosen on the other three folds; each game is compared with the
submitted decoder (lam 5) game by game, and the failure breakdown of both is printed.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import explore  # noqa: E402
import external_pff as ep  # noqa: E402
import periods  # noqa: E402
import temporal  # noqa: E402
import touch_signal as ts  # noqa: E402
import vote  # noqa: E402
from bench_pff import folds  # noqa: E402
from bench_temporal import cached_features  # noqa: E402

SEEDS = ["tcn_second_touch"] + ["tcn_s{}_second_touch".format(s) for s in range(1, 5)]
OOF = os.path.join(ep.PFF_DIR, "oof", "second_touch_seeds.npz")
LAMBDAS = (5.0, 7.0, 10.0)
GAMMAS = (0.0, 0.5, 1.0, 2.0, 4.0)
THRESH = (20.0, 30.0)
RADII = (1.5, 3.0)


def oof_probabilities():
    if os.path.isfile(OOF):
        z = np.load(OOF)
        return {int(k[2:]): (z[k], z["per" + k[2:]]) for k in z.files if k.startswith("p_")}
    nets = [temporal.load_models(os.path.join(temporal.MODEL_DIR, "second_touch"), prefix=p) for p in SEEDS]
    out = {}
    for k, test, _ in folds(ep.games()):
        for g in test:
            frames, _ = ep.load(g)
            res = [temporal.step_probabilities(n[k], frames, cached_features(g)) for n in nets]
            out[g] = (np.mean([r[0] for r in res], axis=0), res[0][2])
    os.makedirs(os.path.dirname(OOF), exist_ok=True)
    np.savez_compressed(OOF, **{"p_{}".format(g): p for g, (p, _) in out.items()},
                        **{"per{}".format(g): per for g, (_, per) in out.items()})
    return out


def viterbi_directed(score, cost):
    """Viterbi over additive scores where entering state j at step t costs cost[t, j]; staying wins ties."""
    n = len(score)
    back = np.zeros((n, 3), dtype=np.int8)
    total = score[0].copy()
    off = 1.0 - np.eye(3)
    for t in range(1, n):
        m = total[None, :] - cost[t][:, None] * off  # m[j, i]: from i into j
        best = m.argmax(1)
        stay = m[np.arange(3), np.arange(3)] >= m.max(1)
        back[t] = np.where(stay, np.arange(3), best)
        total = np.where(stay, total, m.max(1)) + score[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = np.argmax(total)
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def touch_steps(frames, thr, radius):
    """(steps, 2) booleans: H / A touched the ball within one step of each 5 fps step."""
    ball = ts.smooth(frames["ball"])
    accel, _ = temporal._ball_change(ball)
    p = ts.peaks(np.nan_to_num(accel), thr)
    d = np.stack([np.nanmin(np.linalg.norm(frames[t][p] - ball[p][:, None], axis=2), axis=1)
                  for t in ("home", "away")], axis=1) if len(p) else np.zeros((0, 2))
    starts = periods.blocks(frames["period"], temporal.STEP)[0]
    out = np.zeros((len(starts), 2), bool)
    with np.errstate(invalid="ignore"):
        for team in (0, 1):
            mine = (d[:, team] <= radius) & (d[:, team] <= d[:, 1 - team])
            idx = np.searchsorted(starts, p[mine], side="right") - 1
            for shift in (-1, 0, 1):
                out[np.clip(idx + shift, 0, len(starts) - 1), team] = True
    return out


def decode(frames, step_p, step_period, lam, gamma, touched):
    steps = np.full(len(step_p), "X", dtype="<U1")
    base = lam / temporal.STEP
    for a, b in periods.slices(step_period):
        cost = np.full((b - a, 3), base)
        cost[:, :2] *= np.where(touched[a:b], np.exp(-gamma), 1.0)
        steps[a:b] = vote.STATES[viterbi_directed(step_p[a:b], cost)]
    out = periods.expand(steps, frames["period"], temporal.STEP)
    out[frames["period"] == 0] = "X"
    return out


def nested(df, keys):
    out = []
    for k in sorted(df.fold.unique()):
        best = df[df.fold != k].groupby(keys)["spellF1"].mean().idxmax()
        sel = df[df.fold == k]
        for key, v in zip(keys, best):
            sel = sel[sel[key] == v]
        out.append(sel.assign(choice=str(best)))
    return pd.concat(out).set_index("game")


def breakdown(rows):
    m = pd.concat([r[0] for r in rows])
    e = pd.concat([r[1] for r in rows])
    n = m.game.nunique()
    miss = m.reason.value_counts(normalize=True)
    short = m[m.duration.isin(["0-2 s", "2-5 s"])]
    seen = m[m.ball_seen > 0.75]  # the provided match has the ball ~90% of the time, the World Cup ~41%
    return ("found {:.1%} (under 5 s {:.1%}); misses: wrong team {:.0%}, merged {:.0%}, called dead {:.0%}; "
            "invented {:.1f}/game ({:.1f} other team); ball seen >75%: found {:.1%} (under 5 s {:.1%})").format(
        (m.reason == "found").mean(), (short.reason == "found").mean(),
        miss.get("wrong team", 0) / (1 - miss["found"]), miss.get("merged", 0) / (1 - miss["found"]),
        miss.get("called dead", 0) / (1 - miss["found"]), len(e) / n, (e.under == "other team").sum() / n,
        (seen.reason == "found").mean(), (seen[seen.seconds < 5].reason == "found").mean())


if __name__ == "__main__":
    probs = oof_probabilities()
    fold_of = {g: k for k, test, _ in folds(ep.games()) for g in test}
    g0 = next(iter(probs))
    frames, _ = ep.load(g0)
    p, per = probs[g0]
    assert np.array_equal(decode(frames, p, per, temporal.LAMBDA, 0.0, np.zeros((len(p), 2), bool)),
                          temporal.decode_steps(frames, p, per, temporal.LAMBDA)), "gamma 0 must be the submitted decoder"

    def one(g):
        frames, truth = ep.load(g, "second_touch")
        p, per = probs[g]
        rows = []
        for thr in THRESH:
            for radius in RADII:
                touched = touch_steps(frames, thr, radius)
                for lam in LAMBDAS:
                    for gamma in GAMMAS:
                        if gamma == 0 and (thr, radius) != (THRESH[0], RADII[0]):
                            continue
                        r = bench.score(decode(frames, p, per, lam, gamma, touched), truth)
                        rows.append({"game": g, "fold": fold_of[g], "thr": thr, "radius": radius, "lambda": lam,
                                     "gamma": gamma, "spellF1": r["spell_f1_mean"], "macroF1": r["macro_f1"],
                                     "poss": r["n_pred_spells"], "touch_share": touched.any(1).mean()})
        return rows

    df = pd.DataFrame([r for rs in bench.map_games(one, list(probs)) for r in rs])
    keys = ["thr", "radius", "lambda", "gamma"]
    pd.set_option("display.width", 200)
    print("mean over 64 games, best cells:")
    print(df.groupby(keys)[["spellF1", "macroF1", "poss"]].mean().sort_values("spellF1", ascending=False).head(12).round(4).to_string())
    print("\nby gamma (best thr/radius/lambda for each):")
    print(df.groupby(keys)["spellF1"].mean().groupby("gamma").max().round(4).to_string())
    ref = df[(df.gamma == 0) & (df["lambda"] == temporal.LAMBDA)].set_index("game")
    pick = nested(df, keys)
    d = pick.spellF1 - ref.spellF1
    h = 1.96 * d.std() / np.sqrt(len(d))
    print("\nchosen on the other folds: {}".format(sorted(pick.choice.unique())))
    print("spellF1 {:.4f} vs submitted {:.4f}: {:+.4f} (95% CI {:+.4f} .. {:+.4f}) better in {}/{}; macroF1 {:.4f} vs {:.4f}; "
          "possessions/game {:.0f} vs {:.0f}".format(pick.spellF1.mean(), ref.spellF1.mean(), d.mean(), d.mean() - h,
                                                     d.mean() + h, int((d > 0).sum()), len(d), pick.macroF1.mean(),
                                                     ref.macroF1.mean(), pick.poss.mean(), ref.poss.mean()))
    best = df.groupby(keys)["spellF1"].mean().idxmax()
    print("share of steps with a touch flagged (best cell): {:.1%}".format(
        df[(df.thr == best[0]) & (df.radius == best[1])].touch_share.mean()))

    def tables(g, setting):
        frames, truth = ep.load(g, "second_touch")
        p, per = probs[g]
        thr, radius, lam, gamma = setting
        pred = decode(frames, p, per, lam, gamma, touch_steps(frames, thr, radius))
        seen = ~np.isnan(frames["ball"][:, 0])
        return (explore.miss_table(truth, pred, 0, len(truth), seen).assign(game=g),
                explore.extra_table(truth, pred, 0, len(truth)).assign(game=g))

    for name, setting in (("submitted", (THRESH[0], RADII[0], temporal.LAMBDA, 0.0)), ("best cell " + str(best), best)):
        print("\n{}: {}".format(name, breakdown(bench.map_games(lambda g: tables(g, setting), list(probs)))))
