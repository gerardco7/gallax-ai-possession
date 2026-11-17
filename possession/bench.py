"""The test bench: score every method on the external games with the official metric.

    python possession/bench.py                        # every method in methods.METHODS
    python possession/bench.py baseline_smooth50      # just these
    python possession/bench.py --submit baseline_smooth50   # run one on the provided match

Scoring uses evaluation/possession_metrics.evaluate -- the function that produces the official
board -- so a number here means what it means on results day, on different games. Choosing between
methods happens here and only here: the rules forbid tuning on the provided match or its ground
truth, and the public five minutes are too few possessions to rank anything anyway.

--submit writes evaluation/results/ours_<method>.csv and runs the official validation on it. It prints
counts only, never data.
"""

import argparse
import multiprocessing
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, HERE)
import possession_metrics as pm  # noqa: E402

import external_metrica  # noqa: E402
import frames as data  # noqa: E402
from methods import MATCH_ONLY, METHODS, NEEDS_BROADCAST, TRAINABLE  # noqa: E402
import temporal  # noqa: E402
from submission import write_submission  # noqa: E402
from games import GAME, GAMES  # noqa: E402

COLUMNS = ("spell_f1_mean", "macro_f1", "n_pred_spells", "n_gt_spells", "mean_spell_duration_s")


PROCESSES = 8
_WORK = None


def _call(item):
    return _WORK(item)


def map_games(fn, items, processes=PROCESSES):
    """fn over items in parallel. Forked workers inherit fn and everything it closes over (trained
    models included), so nothing is pickled but the items and the results."""
    global _WORK
    _WORK = fn
    with multiprocessing.get_context("fork").Pool(min(processes, len(items))) as pool:
        return pool.map(_call, items)


def score(labels, truth):
    return pm.evaluate(pm.label_codes(truth), pm.label_codes(labels))


def bench(names, games=external_metrica.GAMES, degraded=False):
    loaded = {game: data.load_metrica(game) for game in games}
    if degraded:
        loaded = {g: (data.degrade(f, t, seed=g), t) for g, (f, t) in loaded.items()}
        print("REALISTIC bench: " + data.DEGRADE_DESCRIPTION)
    print("{:<22}".format("method") + "".join(
        "  g{} spellF1 macroF1  poss(gt)  dur".format(g) for g in games) + "   MEAN spellF1")
    for name in names:
        row, spell = "{:<22}".format(name), []
        for game, (frames, truth) in loaded.items():
            if name in TRAINABLE:  # trained on the other games only, always on their clean data
                method = TRAINABLE[name]([data.load_metrica(g) for g in games if g != game])
            else:
                method = METHODS[name]
            r = score(method(frames), truth)
            spell.append(r["spell_f1_mean"])
            row += "  {:>11.4f} {:>7.4f} {:>4}({:>3}) {:>4.1f}".format(
                r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"],
                r["mean_spell_duration_s"])
        print(row + "   {:.4f}".format(np.mean(spell)))


# the submission recipe with the touch detector's probabilities as inputs (bench_touch_input.py), decoded at
# the submission's cost 7 (the World Cup is flat between 7 and 10, the 3 Metrica games fall with the cost):
# name -> directory under <models> (its detector/ holds the four fold detectors) and network tag
TOUCH_SUBMISSIONS = {
    # detector trained on the World Cup and SkillCorner (touch_detector.py --sources wc+sc --res10)
    "tcn_second_touch_sw2_ch128_sc1_touch_seeds": ("second_touch_sw2_channels128_sc1_touch", "second_touch_sw2_channels128_sc1_touch-wcsc_res10"),
    # detector trained on the World Cup only (touch_detector.py --sources wc --res10)
    "tcn_second_touch_sw2_ch128_sc1_touch_wc_seeds": ("second_touch_sw2_channels128_sc1_touch_wc", "second_touch_sw2_channels128_sc1_touch-wc_res10"),
}


def submit(name, allow_legacy_models=False, model_dir=None):
    match = data.load_match()
    def load_networks(prefix="tcn"):
        hp = dict(temporal.DEFAULTS, arch=prefix) if allow_legacy_models else None
        return temporal.load_models(directory=model_dir or temporal.MODEL_DIR, prefix=prefix, hp=hp, allow_legacy=allow_legacy_models)
    networks = {  # networks trained by bench_temporal.py on the World Cup games, and their switching cost
        "tcn": (lambda: load_networks(), temporal.EVENT_LAMBDA),  # trained on the event labels
        "lstm": (lambda: load_networks("lstm"), 2.0),
        "tcn_lstm": (lambda: load_networks() + load_networks("lstm"), 2.0),
        "tcn_lstm_tf": (lambda: load_networks() + load_networks("lstm")
                        + load_networks("transformer"), 2.0),
        # the TCN trained on second-touch labels (bench_temporal.py --labels second_touch)
        "tcn_second_touch": (lambda: load_networks("tcn_second_touch"), temporal.LAMBDA),
        # the TCN trained on second-touch labels with seeds 0-4 per fold, averaged; in <models>/second_touch
        "tcn_second_touch_seeds": (lambda: [m for p in ["tcn_second_touch"] + ["tcn_s{}_second_touch".format(s) for s in range(1, 5)]
                                            for m in temporal.load_models(os.path.join(model_dir or temporal.MODEL_DIR, "second_touch"),
                                                                          prefix=p)], temporal.LAMBDA),
        # the same with short possessions weighted (bench_temporal.py --short-weight 2), cost 7 chosen on the
        # World Cup folds (bench_short_weight_seeds.py); in <models>/second_touch_sw2
        "tcn_second_touch_sw2_seeds": (lambda: [m for p in ["tcn_second_touch_sw2"] + ["tcn_s{}_second_touch_sw2".format(s) for s in range(1, 5)]
                                                for m in temporal.load_models(os.path.join(model_dir or temporal.MODEL_DIR, "second_touch_sw2"),
                                                                              prefix=p)], 7.0),
        # the same at 128 channels (--hp channels=128), cost 10 chosen on the World Cup folds; in
        # <models>/second_touch_sw2_channels128
        "tcn_second_touch_sw2_ch128_seeds": (lambda: [m for p in ["tcn_second_touch_sw2_channels128"] + ["tcn_s{}_second_touch_sw2_channels128".format(s) for s in range(1, 5)]
                                                      for m in temporal.load_models(os.path.join(model_dir or temporal.MODEL_DIR, "second_touch_sw2_channels128"),
                                                                                    prefix=p)], 10.0),
        # the same trained on the World Cup and SkillCorner together (bench_temporal_sc.py --sc-weight 1), cost 7
        # chosen on the World Cup folds; in <models>/second_touch_sw2_channels128_sc1
        "tcn_second_touch_sw2_ch128_sc1_seeds": (lambda: [m for p in ["tcn_second_touch_sw2_channels128_sc1"] + ["tcn_s{}_second_touch_sw2_channels128_sc1".format(s) for s in range(1, 5)]
                                                          for m in temporal.load_models(os.path.join(model_dir or temporal.MODEL_DIR, "second_touch_sw2_channels128_sc1"),
                                                                                        prefix=p)], 7.0),
        # the TCN with seeds 0-4 per fold (bench_temporal.py --seed), averaged; seed models in <models>/seeds
        "tcn_seeds": (lambda: load_networks() + [m for s in range(1, 5) for m in temporal.load_models(
            os.path.join(model_dir or temporal.MODEL_DIR, "seeds"), prefix="tcn_s{}".format(s),
            hp=dict(temporal.DEFAULTS, arch="tcn") if allow_legacy_models else None,
            allow_legacy=allow_legacy_models)], 3.0),
    }
    base = name[:-5] if name.endswith("_sync") else name
    if name in NEEDS_BROADCAST or ((base in networks or base in TOUCH_SUBMISSIONS) and name.endswith("_sync")):
        match = data.add_broadcast(match)
    if base in TOUCH_SUBMISSIONS:
        import touch_inputs  # here, not at the top: touch_detector imports this module
        touch_inputs.enable()
        folder, tag = TOUCH_SUBMISSIONS[base]
        directory = os.path.join(model_dir or temporal.MODEL_DIR, folder)
        models = [m for p in ["tcn_" + tag] + ["tcn_s{}_{}".format(s, tag) for s in range(1, 5)]
                  for m in temporal.load_models(directory, prefix=p)]
        labels = touch_inputs.method(models, os.path.join(directory, "detector"), 7.0,
                                     broadcast_ball=name.endswith("_sync"))(match)
    elif base in networks:
        load, lam = networks[base]
        labels = temporal.ensemble_method(load(), broadcast_ball=name.endswith("_sync"), lam=lam)(match)
    elif name in MATCH_ONLY:
        labels = MATCH_ONLY[name](match)
    elif name in TRAINABLE:
        labels = TRAINABLE[name]([data.load_metrica(g) for g in external_metrica.GAMES])(match)
    else:
        labels = METHODS[name](match)
    # Prefixed so a method named like an organisers' baseline never overwrites their committed file.
    path = os.path.join(ROOT, "evaluation", "results", "ours_{}.csv".format(name))
    write_submission(path, labels, expected_rows=GAMES[GAME]["n_video_frames"])
    print("wrote {}  (valid)".format(os.path.relpath(path, ROOT)))
    print("  " + "  ".join("{} {}".format(c, int(np.sum(labels == c))) for c in "HADX"))
    return path


def selftest():
    frames, truth = data.load_metrica(1)
    assert score(truth, truth)["spell_f1_mean"] == 1.0
    assert set(np.unique(METHODS["baseline_smooth50"](frames))) <= set("HADX")
    print("selftest ok")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("methods", nargs="*", help="method names; default every method")
    parser.add_argument("--submit", metavar="METHOD", help="run one method on the provided match")
    parser.add_argument("--model-dir", help="versioned run checkpoint directory")
    parser.add_argument("--allow-legacy-models", action="store_true", help="explicitly use historical bare weights with the frozen baseline configuration")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--realistic", action="store_true", help="score on the degraded external games")
    args = parser.parse_args()
    if args.selftest:
        selftest()
    elif args.submit:
        submit(args.submit, allow_legacy_models=args.allow_legacy_models, model_dir=args.model_dir)
    else:
        bench(args.methods or [n for n in list(METHODS) + list(TRAINABLE) if n not in NEEDS_BROADCAST],
              degraded=args.realistic)
