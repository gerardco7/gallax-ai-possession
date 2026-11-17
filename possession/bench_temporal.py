"""The temporal network on the World Cup bench, in the same folds as bench_pff.py.

    python possession/bench_temporal.py              # the TCN
    python possession/bench_temporal.py --arch lstm  # the bidirectional LSTM, plus its average with the TCN
    python possession/bench_temporal.py --arch transformer   # the conv + transformer hybrid, likewise
    python possession/bench_temporal.py --arch transformer_pure  # attention alone, likewise
    python possession/bench_temporal.py --seed 3                 # the TCN with another training seed
    python possession/bench_temporal.py --labels second_touch    # trained and scored on second-touch labels
    python possession/bench_temporal.py --labels second_touch --short-weight 1   # short possessions weigh more
    python possession/bench_temporal.py --labels second_touch --short-weight 2 --short-min 2   # only 2-5 s ones
    python possession/bench_temporal.py --labels second_touch --short-weight 2 --hp channels=128   # other hp

Per fold: train on the other 48 games (42 to fit, 6 to decide when to stop), score the 16 test
games for every Viterbi switching cost in LAMBDAS, and compare game by game with the methods of
bench_pff.py (external/pff_wc2022/bench_results.csv). Features are cached per game under
external/pff_wc2022/features/; each fold's network is saved under external/pff_wc2022/models/.
"""

import multiprocessing
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import external_pff as ep  # noqa: E402
import temporal  # noqa: E402
import provenance  # noqa: E402
from bench_pff import folds  # noqa: E402

LAMBDAS = (0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0)
FEATURE_DIR = os.path.join(ep.PFF_DIR, "features")
MODEL_DIR = os.path.join(ep.PFF_DIR, "models")
ARCH = sys.argv[sys.argv.index("--arch") + 1] if "--arch" in sys.argv else "tcn"
SEED = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else temporal.SEED
temporal.SEED = SEED
LABELS = sys.argv[sys.argv.index("--labels") + 1] if "--labels" in sys.argv else "events"  # see external_pff.load
HP_OVERRIDES = {a.split("=")[0]: type(temporal.DEFAULTS[a.split("=")[0]])(a.split("=")[1])
                for a in sys.argv[sys.argv.index("--hp") + 1].split(",")} if "--hp" in sys.argv else {}
temporal.DEFAULTS.update(HP_OVERRIDES)  # recorded in the run config and each checkpoint's hp
SHORT_WEIGHT = float(sys.argv[sys.argv.index("--short-weight") + 1]) if "--short-weight" in sys.argv else 0.0
SHORT_STEPS = 5 * temporal.FPS // temporal.STEP  # possessions under 5 s
SHORT_MIN_S = float(sys.argv[sys.argv.index("--short-min") + 1]) if "--short-min" in sys.argv else 0.0
MIN_STEPS = SHORT_MIN_S * temporal.FPS / temporal.STEP  # runs shorter than this keep the plain weights
SANDWICH = float(sys.argv[sys.argv.index("--sandwich") + 1]) if "--sandwich" in sys.argv else 1.0
SEEN_WEIGHT = float(sys.argv[sys.argv.index("--seen-weight") + 1]) if "--seen-weight" in sys.argv else 1.0
TAG = (ARCH + ("" if SEED == 0 else "_s{}".format(SEED))  # seed models: tcn_s<seed>_fold<k>.pt
       + ("" if LABELS == "events" else "_" + LABELS)
       + ("" if SHORT_WEIGHT == 0 else "_sw{:g}".format(SHORT_WEIGHT))
       + ("" if SHORT_MIN_S == 0 else "_min{:g}".format(SHORT_MIN_S))
       + ("" if SANDWICH == 1 else "_sand{:g}".format(SANDWICH))
       + ("" if SEEN_WEIGHT == 1 else "_seen{:g}".format(SEEN_WEIGHT))
       + "".join("_{}{}".format(k, v) for k, v in sorted(HP_OVERRIDES.items())))
OUT = os.path.join(ep.PFF_DIR, "bench_temporal.csv" if TAG == "tcn" else "bench_temporal_{}.csv".format(TAG))
VAL_EVERY = 8


def cached_features(game):
    path = os.path.join(FEATURE_DIR, "{}.npz".format(game))
    return provenance.cached_features(path, [os.path.join(ep.OUT_DIR, "{}.npz".format(game))],
                                      temporal.feature_schema(), lambda: temporal.features(ep.load(game)[0]))


def short_weights(y, trans_weight, _base=temporal._weights):
    """temporal._weights, and every step of an H or A run under 5 s weighs at least
    1 + SHORT_WEIGHT * (min(5 s / its length, 5) - 1). Swapped in here rather than edited in
    temporal.py, whose source hash the submitted checkpoints are bound to."""
    w = _base(y, trans_weight)
    labels = y.cpu().numpy()
    extra = np.ones(labels.shape, np.float32)
    for row, lab in enumerate(labels):
        edges = np.r_[0, np.flatnonzero(lab[1:] != lab[:-1]) + 1, len(lab)]
        runs = [lab[a] for a in edges[:-1]]
        teams = [i for i, r in enumerate(runs) if r in (0, 1)]
        for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            if lab[a] in (0, 1) and MIN_STEPS <= b - a < SHORT_STEPS:
                extra[row, a:b] = 1 + SHORT_WEIGHT * (min(SHORT_STEPS / (b - a), 5.0) - 1)
                j = teams.index(i)  # a run between two runs of the other team (dead balls aside) weighs SANDWICH times more
                if SANDWICH != 1 and 0 < j < len(teams) - 1 and runs[teams[j - 1]] == runs[teams[j + 1]] != lab[a]:
                    extra[row, a:b] *= SANDWICH
    return torch.maximum(w, torch.tensor(extra, device=w.device))


if SHORT_WEIGHT:
    temporal._weights = short_weights

_LAST_X = []


def seen_batches(data, rng, hp, gen, _base=temporal._batches):
    """temporal._batches, keeping the last crop's features for seen_weights."""
    for x, y in _base(data, rng, hp, gen):
        _LAST_X[:] = [x]
        yield x, y


def seen_weights(y, trans_weight, _base=temporal._weights):
    """The weights above, times SEEN_WEIGHT on every step where the tracker has the ball (feature 0,
    after noise and ball drop): the provided match has the ball ~90% of the time, the World Cup ~41%."""
    return _base(y, trans_weight) * torch.where(_LAST_X[0][..., 0] > 0.5, SEEN_WEIGHT, 1.0)


if SEEN_WEIGHT != 1:
    temporal._batches = seen_batches
    temporal._weights = seen_weights


def sequences(games):
    out = []
    for g in games:
        frames, labels = ep.load(g, LABELS)
        feats, centres = cached_features(g)
        out += temporal.halves(feats, labels, centres, frames["period"])
    return out


if __name__ == "__main__":
    os.makedirs(FEATURE_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    t0 = time.time()
    with multiprocessing.Pool(8) as pool:
        pool.map(cached_features, ep.games())
    print("features ready in {:.0f} s".format(time.time() - t0), flush=True)
    split = [{"fold": k, "test": test, "validation": [g for g in ep.games() if g not in test][::VAL_EVERY],
              "fit": [g for i, g in enumerate([g for g in ep.games() if g not in test]) if i % VAL_EVERY]}
             for k, test, _ in folds(ep.games())]
    run = provenance.Run(ep.PFF_DIR, "bench-temporal-" + TAG,
                         {"seed": SEED, "labels": LABELS, "short_weight": SHORT_WEIGHT,
                          **({"short_min_s": SHORT_MIN_S} if SHORT_MIN_S else {}),
                          **({"sandwich": SANDWICH} if SANDWICH != 1 else {}),
                          **({"seen_weight": SEEN_WEIGHT} if SEEN_WEIGHT != 1 else {}),"hp": dict(temporal.DEFAULTS, arch=ARCH, crop=temporal.DEFAULTS["tf_window"] if ARCH.startswith("transformer") else temporal.DEFAULTS["crop"]), "costs": list(LAMBDAS)}, split,
                         [os.path.join(ep.OUT_DIR, str(g) + ".npz") for g in ep.games()]
                         + ([os.path.join(MODEL_DIR, n) for n in sorted(os.listdir(MODEL_DIR)) if n.startswith("tcn_fold") and n.endswith(".pt")] if ARCH != "tcn" else []),
                         {"status": "reused development CV, not an untouched test"})
    if run.manifest["status"] == "complete":
        print("Completed run already verified at", run.path)
        raise SystemExit(0)
    MODEL_DIR = str(run.path / "models")
    os.makedirs(MODEL_DIR, exist_ok=True)
    rows = []
    tcn_models = temporal.load_models(temporal.MODEL_DIR, hp=dict(temporal.DEFAULTS), allow_legacy=True) if ARCH != "tcn" else None
    for k, test, _ in folds(ep.games()):
        rest = [g for g in ep.games() if g not in test]
        val, fit = rest[::VAL_EVERY], [g for i, g in enumerate(rest) if i % VAL_EVERY]
        t1 = time.time()
        hp = {"arch": ARCH}
        if ARCH.startswith("transformer"):  # attention over the training crop must fit in memory: crops of one window
            hp["crop"] = temporal.DEFAULTS["tf_window"]
        model = temporal.train(sequences(fit), sequences(val), log=lambda s: print(s, flush=True), hp=hp,
                               provenance={"training": ["pff:" + str(g) for g in fit],
                                           "validation": ["pff:" + str(g) for g in val],
                                           "inputs": provenance.inputs([os.path.join(ep.OUT_DIR, str(g) + ".npz") for g in rest])})
        temporal.save_model(model, os.path.join(MODEL_DIR, "{}_fold{}.pt".format(TAG, k)))
        print("fold {}: trained on {} games (+{} for stopping) in {:.0f} s".format(k, len(fit), len(val), time.time() - t1), flush=True)
        for g in test:
            frames, truth = ep.load(g, LABELS)
            step_p, _, step_period = temporal.step_probabilities(model, frames, cached_features(g))
            variants = {ARCH: step_p}
            if tcn_models is not None:  # the average of this network and the same fold's TCN
                tcn_p = temporal.step_probabilities(tcn_models[k], frames, cached_features(g))[0]
                variants["tcn+" + ARCH] = (step_p + tcn_p) / 2
            for name, p in variants.items():
                for lam in LAMBDAS:
                    r = bench.score(temporal.decode_steps(frames, p, step_period, lam), truth)
                    rows.append({"game": g, "method": name, "lambda": lam, "spellF1": r["spell_f1_mean"],
                                 "macroF1": r["macro_f1"], "poss": r["n_pred_spells"], "true_poss": r["n_gt_spells"]})
        run.results(pd.DataFrame(rows))
    df = pd.DataFrame(rows)
    run.results(df, complete=True)
    print(df.groupby(["method", "lambda"])[["spellF1", "macroF1", "poss", "true_poss"]].mean().round(4).to_string())
    if "--historical-references" not in sys.argv:
        print("Versioned results saved at", run.path)
        raise SystemExit(0)
    print("Historical references below are not provenance-matched controlled comparisons")
    if not all(os.path.isfile(os.path.join(ep.PFF_DIR, name)) for name in ['bench_results.csv']):
        print("Historical flat comparison CSVs unavailable; versioned results saved at", run.path)
        raise SystemExit(0)
    old = pd.read_csv(os.path.join(ep.PFF_DIR, "bench_results.csv"))
    old["key"] = old["method"] + np.where(old["lambda"].astype(str) == "-", "", "@" + old["lambda"].astype(str))
    ref = old.pivot_table(index="game", columns="key", values="spellF1")
    if ARCH != "tcn":
        tcn = pd.read_csv(os.path.join(ep.PFF_DIR, "bench_temporal.csv"))
        ref["tcn"] = tcn[tcn["lambda"] == temporal.LAMBDA].set_index("game")["spellF1"]
    for name in df["method"].unique():
        best = df[df["method"] == name].groupby("lambda")["spellF1"].mean().idxmax()
        mine = df[(df["method"] == name) & (df["lambda"] == best)].set_index("game")["spellF1"]
        print("\n{} at lambda {} against the earlier methods, game by game:".format(name, best))
        for key in (["tcn"] if ARCH != "tcn" else []) + ["vote_soft@10.0", "control", "baseline_smooth50"]:
            d = (mine - ref[key]).dropna()
            print("  minus {:<18} {:+.4f}  (95% CI {:+.4f} .. {:+.4f})  better in {}/{}".format(
                key, d.mean(), d.mean() - 1.96 * d.std() / np.sqrt(len(d)), d.mean() + 1.96 * d.std() / np.sqrt(len(d)),
                int((d > 0).sum()), len(d)))
    print("\ntotal {:.0f} s; wrote {}".format(time.time() - t0, os.path.relpath(OUT)))
