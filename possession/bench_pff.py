"""Test bench on the 64 games of PFF FC's 2022 World Cup data.

    python possession/bench_pff.py            # 4 folds; results to external/pff_wc2022/bench_results.csv

The games are used as they are: their tracking is automatic and uncorrected (frames["real"]), so
nothing is degraded. Four folds of 16 test games; each fold's models are trained on 16 of the other
48 games (every third, so the choice is fixed), which bounds memory and time -- the learning curve
measured on the Metrica games flattens long before that many. Every game is tested exactly once, by
models that never saw it. The three voters are trained once per fold and both votes are built from
them, for every LAMBDA in the grid.

Licence of the data unconfirmed: see external_pff.py.
"""

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import external_pff as ep  # noqa: E402
import gaps  # noqa: E402
import hmm  # noqa: E402
import methods  # noqa: E402
import touches  # noqa: E402
import vote  # noqa: E402

FOLDS = 4
TRAIN_EVERY = 3
LAMBDAS = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
OUT = os.path.join(ep.PFF_DIR, "bench_results.csv")


def folds(all_games):
    for k in range(FOLDS):
        test = [g for i, g in enumerate(all_games) if i % FOLDS == k]
        rest = [g for g in all_games if g not in test]
        yield k, test, rest[::TRAIN_EVERY]


def score_row(game, name, labels, truth, lam="-"):
    r = bench.score(labels, truth)
    return {"game": game, "method": name, "lambda": lam, "spellF1": r["spell_f1_mean"], "macroF1": r["macro_f1"],
            "poss": r["n_pred_spells"], "true_poss": r["n_gt_spells"]}


STEP = 5  # votes decoded at 5 fps (vote.decode)


if __name__ == "__main__":
    rows = []
    for k, test, train_ids in folds(ep.games()):
        t0 = time.time()
        train = [ep.load(g) for g in train_ids]
        stoppages = gaps.train(train)
        hm, tm = hmm.PossessionHMM().fit(train, stoppages), touches.TouchModel().fit(train, stoppages)
        del train
        ctrl = methods.control_both(stoppages)
        baseline = methods.baseline(50)
        print("fold {}: trained on {} games in {:.0f} s".format(k, len(train_ids), time.time() - t0), flush=True)

        def one_game(g):
            frames, truth = ep.load(g)
            both = {"control": ctrl(frames), "hmm": hm.both(frames), "touches": tm.both(frames)}
            labels = {v: both[v][0] for v in vote.VOTERS}
            probs = [both[v][1] for v in vote.VOTERS]
            out = [score_row(g, "baseline_smooth50", baseline(frames), truth)]
            out += [score_row(g, v, labels[v], truth) for v in vote.VOTERS]
            for lam in LAMBDAS:
                out.append(score_row(g, "vote", vote.combine([labels[v] for v in vote.VOTERS], frames["period"], lam, STEP), truth, lam))
                out.append(score_row(g, "vote_soft", vote.combine_soft(probs, frames["period"], lam, STEP), truth, lam))
            return out

        for game_rows in bench.map_games(one_game, test):
            rows += game_rows
        print("fold {}: tested {} games, {:.0f} s in all".format(k, len(test), time.time() - t0), flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)
    df = pd.DataFrame(rows)
    summary = df.groupby(["method", "lambda"]).agg(spellF1=("spellF1", "mean"), spellF1_sd=("spellF1", "std"),
                                                   macroF1=("macroF1", "mean"), poss=("poss", "mean"),
                                                   true_poss=("true_poss", "mean"), games=("game", "count"))
    summary["spellF1_se"] = summary["spellF1_sd"] / np.sqrt(summary["games"])
    print(summary.round(4).sort_values("spellF1", ascending=False).to_string())
    print("\nwrote", os.path.relpath(OUT))
