"""Option A: short possessions weigh more in the loss (bench_temporal.py --short-weight k), against seed 0.

    python possession/bench_short_weight.py 1 2 2_min2   # k values, or k_min<s> for --short-min runs

Reads the versioned results of bench-temporal-tcn_second_touch (k = 0, the submitted seed 0), its
seeds 1-4, and bench-temporal-tcn_second_touch_sw<k> for each k given. Per game, each setting is
scored at the cost the other three folds prefer; k itself is also chosen on the other folds. Then the
failure breakdown of each k at its most chosen cost, from that run's fold networks.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import explore  # noqa: E402
import external_pff as ep  # noqa: E402
import temporal  # noqa: E402
from bench_pff import folds  # noqa: E402
from bench_temporal import cached_features  # noqa: E402
from bench_touch_cost import breakdown  # noqa: E402

RUNS = os.path.join(ep.PFF_DIR, "runs")


def run_dir(tag):
    dirs = [d for d in glob.glob(os.path.join(RUNS, "bench-temporal-" + tag, "*")) if os.path.isfile(os.path.join(d, "results.csv"))]
    if len(dirs) != 1:
        raise SystemExit("expected one run for {}, found {}".format(tag, len(dirs)))
    return dirs[0]


def nested(df, keys):
    out = []
    for k in sorted(df.fold.unique()):
        best = df[df.fold != k].groupby(keys)["spellF1"].mean().idxmax()
        best = best if isinstance(best, tuple) else (best,)
        sel = df[df.fold == k]
        for key, v in zip(keys, best):
            sel = sel[sel[key] == v]
        out.append(sel)
    return pd.concat(out).set_index("game")


def paired(name, mine, ref):
    d = (mine - ref).dropna()
    h = 1.96 * d.std() / np.sqrt(len(d))
    print("  {:<34} {:+.4f} (95% CI {:+.4f} .. {:+.4f}) better in {}/{}".format(
        name, d.mean(), d.mean() - h, d.mean() + h, int((d > 0).sum()), len(d)))


if __name__ == "__main__":
    ks = sys.argv[1:]
    fold_of = {g: k for k, test, _ in folds(ep.games()) for g in test}
    tags = {"0": "tcn_second_touch"} | {k: "tcn_second_touch_sw" + k for k in ks}
    df = pd.concat([pd.read_csv(os.path.join(run_dir(t), "results.csv")).assign(k=k) for k, t in tags.items()])
    df = df[df["lambda"] > 0].assign(fold=lambda x: x.game.map(fold_of))
    seeds = [nested(pd.read_csv(os.path.join(run_dir(t), "results.csv")).assign(fold=lambda x: x.game.map(fold_of)),
                    ["lambda"]).spellF1 for t in ["tcn_second_touch"] + ["tcn_s{}_second_touch".format(s) for s in range(1, 5)]]
    typical = pd.concat(seeds, axis=1).mean(1)
    pd.set_option("display.width", 200)
    print(df.groupby(["k", "lambda"])[["spellF1", "macroF1", "poss"]].mean().unstack("k").round(4).to_string())
    picked = {k: nested(d, ["lambda"]) for k, d in df.groupby("k")}
    print("\ncost chosen on the other folds:")
    for k, d in picked.items():
        print("  k {}: spellF1 {:.4f} macroF1 {:.4f} poss/game {:.0f} (costs {})".format(
            k, d.spellF1.mean(), d.macroF1.mean(), d.poss.mean(), sorted(d["lambda"].unique())))
    print("\ngame by game:")
    for k in ks:
        paired("k {} minus seed 0".format(k), picked[k].spellF1, picked["0"].spellF1)
        paired("k {} minus a typical seed".format(k), picked[k].spellF1, typical)
    both = nested(df, ["k", "lambda"])
    paired("k and cost chosen on other folds", both.spellF1, picked["0"].spellF1)
    print("  (k chosen per fold: {})".format(both.groupby("fold").k.first().tolist()))

    for k, tag in tags.items():
        lam = picked[k]["lambda"].mode()[0]
        models = temporal.load_models(os.path.join(run_dir(tag), "models"), prefix=tag)

        def tables(g, models=models, lam=lam):
            frames, truth = ep.load(g, "second_touch")
            p, _, per = temporal.step_probabilities(models[fold_of[g]], frames, cached_features(g))
            pred = temporal.decode_steps(frames, p, per, lam)
            seen = ~np.isnan(frames["ball"][:, 0])
            return (explore.miss_table(truth, pred, 0, len(truth), seen).assign(game=g),
                    explore.extra_table(truth, pred, 0, len(truth)).assign(game=g))

        print("\nk {} at cost {:g}: {}".format(k, lam, breakdown([tables(g) for g in ep.games()])))
