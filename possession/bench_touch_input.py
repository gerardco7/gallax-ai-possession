"""The touch detector's probabilities as two more inputs of the possession network, World Cup + SkillCorner.

    python possession/bench_touch_input.py --labels second_touch --short-weight 2 --hp channels=128 --sc-weight 1

The submission's recipe (bench_temporal_sc.py, same flags, folds and masking), with one more feature per
team: the touch detector's probability that a player of that team touches the ball, its maximum over each
0.2 s step. The probabilities are touch_detector.py's out-of-fold ones (--detector, default
wc+sc_res10): each game's from the detector that did not see it, as in stacking. Called "touch" in
temporal.TEAM_FEATURES, so temporal._drop_ball removes it with the ball and the home/away swap moves it
with its team. Compared game by game with the same seed of bench_temporal_sc.py on the same held-out
games, cost chosen on the other folds of each dataset.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_temporal as bt  # noqa: E402  (reads --labels, --short-weight, --hp, --seed)
import bench  # noqa: E402
import bench_temporal_sc as bts  # noqa: E402
import explore  # noqa: E402
import external_pff as ep  # noqa: E402
import external_skillcorner as sc  # noqa: E402
import periods  # noqa: E402
import provenance  # noqa: E402
import temporal  # noqa: E402
from bench_pff import folds  # noqa: E402
from bench_short_weight import nested, paired  # noqa: E402
from bench_touch_cost import breakdown  # noqa: E402

DETECTOR = sys.argv[sys.argv.index("--detector") + 1] if "--detector" in sys.argv else "wc+sc_res10"
TOUCH_DIR = os.path.join(ep.PFF_DIR, "touch", DETECTOR)
BASE_TAG = bts.TAG
BASE_SCHEMA = temporal.feature_schema()  # the cached 24 features are read under the schema they were written with
NB, NT = len(temporal.BALL_FEATURES), len(temporal.TEAM_FEATURES)
temporal.TEAM_FEATURES.append("touch")
TAG = BASE_TAG + "_touch-" + DETECTOR.replace("+", "")
OUT = os.path.join(ep.PFF_DIR, "sc_runs", TAG)
LAMBDAS = bts.LAMBDAS


def augmented(source, game):
    if source == "wc":
        frames, labels = ep.load(game, "second_touch")
        path, inp = os.path.join(bt.FEATURE_DIR, "{}.npz".format(game)), os.path.join(ep.OUT_DIR, "{}.npz".format(game))
    else:
        frames, labels = sc.load(game)
        path, inp = os.path.join(sc.SC_DIR, "features", "{}.npz".format(game)), os.path.join(sc.OUT_DIR, "{}.npz".format(game))
    feats, centres = provenance.cached_features(path, [inp], BASE_SCHEMA, None)
    p = np.load(os.path.join(TOUCH_DIR, "{}_{}.npz".format(source, game)))["p"].astype(np.float32)
    starts, stops, _ = periods.blocks(frames["period"], temporal.STEP)
    touch = np.stack([p[a:b, 1:].max(0) for a, b in zip(starts, stops)])
    x = np.concatenate([feats[:, :NB + NT], touch[:, :1], feats[:, NB + NT:], touch[:, 1:]], axis=1)
    assert x.shape[1] == temporal.n_features()
    return frames, labels, (x, centres)


def sequences(source, games):
    out = []
    for g in games:
        frames, labels, (x, centres) = augmented(source, g)
        out += temporal.halves(x, bts.masked(labels) if source == "sc" else labels, centres, frames["period"])
    return out


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    wc, scg = ep.games(), sc.games()
    base = pd.read_csv(os.path.join(ep.PFF_DIR, "sc_runs", BASE_TAG, "results.csv"))
    base = base[base.model == "sc"].assign(model="base")
    base_oof = dict(np.load(os.path.join(ep.PFF_DIR, "sc_runs", BASE_TAG, "oof.npz")))
    rows, probs = [], {}
    for k, test, _ in folds(wc):
        rest = [g for g in wc if g not in test]
        val, fit = rest[::bt.VAL_EVERY], [g for i, g in enumerate(rest) if i % bt.VAL_EVERY]
        sc_test = scg[k::4]
        sc_rest = [g for g in scg if g not in sc_test]
        sc_val, sc_fit = sc_rest[::8], [g for i, g in enumerate(sc_rest) if i % 8]
        path = os.path.join(OUT, "{}_fold{}.pt".format(TAG, k))
        if os.path.isfile(path):
            model = temporal.load_models(OUT, prefix=TAG)[k]
        else:
            model = temporal.train(sequences("wc", fit) + sequences("sc", sc_fit) * bts.SC_WEIGHT,
                                   sequences("wc", val) + sequences("sc", sc_val), log=lambda s: print(s, flush=True),
                                   provenance={"training": ["pff:" + str(g) for g in fit] + ["skillcorner:" + str(g) for g in sc_fit],
                                               "validation": ["pff:" + str(g) for g in val] + ["skillcorner:" + str(g) for g in sc_val],
                                               "inputs": provenance.inputs([os.path.join(ep.OUT_DIR, str(g) + ".npz") for g in rest]
                                                                           + [os.path.join(sc.OUT_DIR, str(g) + ".npz") for g in sc_rest])})
            temporal.save_model(model, path)
        print("fold {} ready".format(k), flush=True)
        for dataset, source, games in (("worldcup", "wc", test), ("skillcorner", "sc", sc_test)):
            for g in games:
                frames, truth, feats = augmented(source, g)
                p, _, per = temporal.step_probabilities(model, frames, feats)
                probs[dataset, g, "touch"] = (p, per)
                probs[dataset, g, "base"] = (base_oof["{}_{}_sc".format(dataset, g)], per)
                for lam in LAMBDAS:
                    r = bench.score(temporal.decode_steps(frames, p, per, lam), truth)
                    rows.append({"dataset": dataset, "game": g, "fold": k, "model": "touch", "lambda": lam,
                                 "spellF1": r["spell_f1_mean"], "macroF1": r["macro_f1"], "poss": r["n_pred_spells"]})
    df = pd.concat([pd.DataFrame(rows), base], ignore_index=True)
    df.to_csv(os.path.join(OUT, "results.csv"), index=False)
    np.savez_compressed(os.path.join(OUT, "oof.npz"), **{"{}_{}_touch".format(d, g): p for (d, g, n), (p, _) in probs.items() if n == "touch"})
    print("\n" + TAG)
    for dataset, d in df.groupby("dataset"):
        picked = {m: nested(x, ["lambda"]) for m, x in d.groupby("model")}
        for m, x in picked.items():
            print("  {} {}: spellF1 {:.4f} macroF1 {:.4f} poss/game {:.0f} costs {}".format(
                dataset, m, x.spellF1.mean(), x.macroF1.mean(), x.poss.mean(), sorted(x["lambda"].unique())))
        paired("{} touch minus base (spellF1)".format(dataset), picked["touch"].spellF1, picked["base"].spellF1)
        paired("{} touch minus base (macroF1)".format(dataset), picked["touch"].macroF1, picked["base"].macroF1)
        for m in ("base", "touch"):
            lam = picked[m]["lambda"].mode()[0]

            def tables(g, m=m, lam=lam, dataset=dataset):
                frames, truth = (ep.load(g, "second_touch") if dataset == "worldcup" else sc.load(g))
                p, per = probs[dataset, g, m]
                pred = temporal.decode_steps(frames, p, per, lam)
                seen = ~np.isnan(frames["ball"][:, 0])
                return (explore.miss_table(truth, pred, 0, len(truth), seen).assign(game=g),
                        explore.extra_table(truth, pred, 0, len(truth)).assign(game=g))

            print("  {} {} at cost {:g}: {}".format(dataset, m, lam, breakdown([tables(g) for g in d.game.unique()])))
