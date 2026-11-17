"""The temporal network trained on the World Cup and SkillCorner together, scored out of fold on both.

    python possession/bench_temporal_sc.py --labels second_touch --short-weight 2 --hp channels=128 --sc-weight 1
    python possession/bench_temporal_sc.py --labels second_touch --short-weight 2 --hp channels=128 --sc-weight 3 --seed 1

Takes bench_temporal.py's flags (they configure training when it is imported). Fold k holds out the
World Cup fold k of bench_temporal.py (16 games) and SkillCorner games [k::4] (5 games); of the rest, it
stops on every 8th World Cup game and 2 SkillCorner games and fits on 42 + 13. --sc-weight N repeats
each SkillCorner half N times in the training list, so its crops are drawn N times as often. SkillCorner
starts a restart's possession at its reception where the World Cup starts it at the restart, so the
3 s either side of every possession that follows a dead ball are left out of its loss.

Scored at every cost in LAMBDAS on the held-out games of both datasets, against the same seed's
128-channel k = 2 fold networks (World Cup only) on the same games; the cost is chosen on the other
folds of each dataset. Models and out-of-fold probabilities under external/pff_wc2022/sc_runs/<TAG>/.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_temporal as bt  # noqa: E402  (reads --labels, --short-weight, --hp, --seed)
import bench  # noqa: E402
import explore  # noqa: E402
import external_pff as ep  # noqa: E402
import external_skillcorner as sc  # noqa: E402
import provenance  # noqa: E402
import temporal  # noqa: E402
from bench_pff import folds  # noqa: E402
from bench_short_weight import nested, paired  # noqa: E402
from bench_touch_cost import breakdown  # noqa: E402

SC_WEIGHT = int(sys.argv[sys.argv.index("--sc-weight") + 1]) if "--sc-weight" in sys.argv else 1
MASK_S = 3.0
LAMBDAS = (5.0, 7.0, 10.0, 15.0)
TAG = bt.TAG + "_sc{}".format(SC_WEIGHT)
OUT = os.path.join(ep.PFF_DIR, "sc_runs", TAG)
BASE_DIR = os.path.join(temporal.MODEL_DIR, "second_touch_sw2_channels128")
BASE_PREFIX = bt.TAG  # the same seed's World Cup-only networks


def sc_features(game):
    path = os.path.join(sc.SC_DIR, "features", "{}.npz".format(game))
    return provenance.cached_features(path, [os.path.join(sc.OUT_DIR, "{}.npz".format(game))], temporal.feature_schema(),
                                      lambda: temporal.features(sc.load(game)[0]))


def masked(labels):
    """'X' (not graded) within MASK_S of every start of a possession that follows a dead ball."""
    out = labels.copy()
    starts = np.flatnonzero(np.isin(labels[1:], ["H", "A"]) & (labels[:-1] == "D")) + 1
    w = int(MASK_S * temporal.FPS)
    for s in starts:
        out[max(s - w, 0):s + w] = "X"
    return out


def sc_sequences(games):
    out = []
    for g in games:
        frames, labels = sc.load(g)
        feats, centres = sc_features(g)
        out += temporal.halves(feats, masked(labels), centres, frames["period"])
    return out


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    wc, scg = ep.games(), sc.games()
    for g in scg:
        sc_features(g)
    base = temporal.load_models(BASE_DIR, prefix=BASE_PREFIX)
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
            model = temporal.train(bt.sequences(fit) + sc_sequences(sc_fit) * SC_WEIGHT, bt.sequences(val) + sc_sequences(sc_val),
                                   log=lambda s: print(s, flush=True),
                                   provenance={"training": ["pff:" + str(g) for g in fit] + ["skillcorner:" + str(g) for g in sc_fit],
                                               "validation": ["pff:" + str(g) for g in val] + ["skillcorner:" + str(g) for g in sc_val],
                                               "inputs": provenance.inputs([os.path.join(ep.OUT_DIR, str(g) + ".npz") for g in rest]
                                                                           + [os.path.join(sc.OUT_DIR, str(g) + ".npz") for g in sc_rest])})
            temporal.save_model(model, path)
        print("fold {} ready".format(k), flush=True)
        for dataset, games, load, feats in (("worldcup", test, lambda g: ep.load(g, "second_touch"), bt.cached_features),
                                            ("skillcorner", sc_test, sc.load, sc_features)):
            for g in games:
                frames, truth = load(g)
                f = feats(g)
                for name, net in (("sc", model), ("base", base[k])):
                    p, _, per = temporal.step_probabilities(net, frames, f)
                    probs[dataset, g, name] = (p, per)
                    for lam in LAMBDAS:
                        r = bench.score(temporal.decode_steps(frames, p, per, lam), truth)
                        rows.append({"dataset": dataset, "game": g, "fold": k, "model": name, "lambda": lam,
                                     "spellF1": r["spell_f1_mean"], "macroF1": r["macro_f1"], "poss": r["n_pred_spells"]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "results.csv"), index=False)
    np.savez_compressed(os.path.join(OUT, "oof.npz"), **{"{}_{}_{}".format(d, g, n): p for (d, g, n), (p, _) in probs.items()})
    print("\n" + TAG)
    for dataset, d in df.groupby("dataset"):
        picked = {m: nested(x, ["lambda"]) for m, x in d.groupby("model")}
        for m, x in picked.items():
            print("  {} {}: spellF1 {:.4f} macroF1 {:.4f} poss/game {:.0f} costs {}".format(
                dataset, m, x.spellF1.mean(), x.macroF1.mean(), x.poss.mean(), sorted(x["lambda"].unique())))
        paired("{} sc minus base (spellF1)".format(dataset), picked["sc"].spellF1, picked["base"].spellF1)
        paired("{} sc minus base (macroF1)".format(dataset), picked["sc"].macroF1, picked["base"].macroF1)
        for m in ("base", "sc"):
            lam = picked[m]["lambda"].mode()[0]

            def tables(g, m=m, lam=lam, dataset=dataset):
                frames, truth = (ep.load(g, "second_touch") if dataset == "worldcup" else sc.load(g))
                p, per = probs[dataset, g, m]
                pred = temporal.decode_steps(frames, p, per, lam)
                seen = ~np.isnan(frames["ball"][:, 0])
                return (explore.miss_table(truth, pred, 0, len(truth), seen).assign(game=g),
                        explore.extra_table(truth, pred, 0, len(truth)).assign(game=g))

            print("  {} {} at cost {:g}: {}".format(dataset, m, lam, breakdown([tables(g) for g in d.game.unique()])))
