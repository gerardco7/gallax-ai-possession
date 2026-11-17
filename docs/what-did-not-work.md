# What did not work

*Gallax AI, More than a Hack 2027, Phase 1.* This repository contains only the code that produces the
submission and retrains it. Everything below was built, measured and **dropped**. The code for these
experiments lived in our research repository; the numbers and the reasons are kept here, because the
technical document asks for dropped assets as well as kept ones. The full, dated record of every
experiment is [`experiments.md`](experiments.md).

Unless a line says otherwise, every number is **spellF1 on the 64 PFF World Cup 2022 games**. Each
game is scored by networks that never saw it (4 folds of 16), and every comparison is a paired
per-game difference with a 95% interval. None of these decisions was made on the provided game or on
its public 5 minutes.

## 1. Rule-based methods (the first three weeks)

| Method | spellF1 | Why it was dropped |
|---|---|---|
| Organisers' baseline (nearest player, majority vote) | 0.336 | reference |
| Gap classifier (stoppage vs lost play) | below the control rule | only decides the ball-missing gaps |
| Control rule (player within 2 m for 0.8 s) | 0.397 | cannot see short possessions: it needs 0.8 s of control |
| HMM over H / A / D | 0.381 | flips 21% more on the match than on the benches; overconfident (says 95–99%, right 75%) |
| Rule-based touch detector | 0.307 | learned the corrected ball's sharp kinks on Metrica's sample data, which real tracking does not have; worse than the control rule in 64 of 64 games |
| Vote of the three, with probabilities | 0.405 | the best rule-based system, 0.3 below the network |

Smaller rule-based ideas, each measured and dropped:

- **Clear-control shortcut** for short possessions (a lone player near the ball takes it after a few
  hundredths of a second): +0.002 at best, worse leave-one-game-out. Short possessions happen in crowds.
- **"Carry" the last team whenever the ball is missing**: never calls a dead ball (macroF1 0.79 → 0.44).
- **A ball-speed condition** in the control rule: tracking noise makes speed over a few frames unreliable.
- **The control rule stacked into the HMM**: within noise; both read the same signals.
- **A fixed 0.7 s delay on the HMM's changes**: +0.015, removed on purpose. It shifts the output
  instead of fixing the model.
- **Calibrating the voters** (Platt scaling): worse than uncalibrated, with an unstable switching cost.
- **Ghost-player cleaning**: the external games have *more* "duplicates" than the match, so a filter
  would delete real players.
- **Rule-based methods re-tuned on the World Cup**: +0.005 to +0.008. Kept out so the fallback has no
  PFF dependency.

## 2. Other networks and ensembles

| Variant | Result against the TCN | Why |
|---|---|---|
| Bidirectional LSTM | −0.054 with the tuned recipe | the TCN is best in all 64 games |
| Conv + attention hybrid transformer | −0.039 | learns slower, validation loss stays higher |
| Pure transformer | −0.120 | too little data for attention alone |
| TCN + LSTM + transformer averaged | −0.013 (10 of 64 better) | they share 73–77% of the TCN's errors |
| Weighted averages | the folds always pick the TCN alone | the apparent gain was the switching cost |
| TCN + three rule-based voters | −0.17 to −0.18, worse in 64 of 64 | weak voters outvote the long context where it is right |
| A change-of-team output head | −0.028 (7 of 64 better) | the network gives short possessions no clear change to find |
| A network at 25 fps instead of 5 | no gain | see "One change at a time" in the log |
| Several sampling phases at inference | no gain | nothing is lost on the 0.2 s grid |

## 3. Features and decoding

- **Hand-written touch features** (ball acceleration and turn near each team's nearest player): no
  gain. The *learned* touch detector that replaced them is in the submission (+0.003 on the World Cup,
  +0.012 on SkillCorner, +0.016 on Metrica).
- **Adaptive switching cost**: no gain over a fixed cost.
- **Log-posterior Viterbi** (log emissions instead of probabilities): promising on DFL with clean
  tracking, but on the World Cup it loses at every fixed cost, and even the out-of-fold choice (cost 8)
  is −0.001 (95% CI −0.002 .. +0.000). It adds ~28 predicted segments per game to match 5 more.
- **A cheaper change towards the team touching the ball** in the decoder: the number of possessions
  barely moves (207 against 206).
- **The touch detector overriding the decoder** (a run of same-team touches pulls the probabilities
  towards that team): level (−0.0001). Where the detector sees such a run, the decoder already agrees
  98% of the time.
- **A touch-gated decoder**: +0.002, and all of it comes from the H/A cost alone. Not worth a new knob.
- **"Does the team that comes to the ball win it?"** (`steal`): no usable signal with the ball seen.
- **Features from the last touches** (time since each team's last touch): dropped for the same reason
  as the override.

## 4. Training data and labels

- **Labels that count every touch as a possession**: train the network to chase possessions the
  challenge does not count. Replacing them with the *second-touch* rule (§4 of the technical document)
  was the single largest gain, 0.639 → 0.717.
- **Requiring 2 s of control** in the labels: contradicts the challenge's own definition, which counts
  very short possessions. Not adopted.
- **Training on IDSSE (DFL) only**: 0.43. **Adding IDSSE to the World Cup**: −0.007. Its per-frame
  possession has 7.4 possessions per live minute against ~4 in our labels, a different convention.
- **Training on Metrica's 3 sample games only** (the plan B if PFF's data could not be used): 0.441.
- **A touch detector trained on the World Cup and SkillCorner together**: detects SkillCorner's
  touches much better (80% at 84% precision against 56% at 54%) but, as an input, it is 0.003 worse
  on the World Cup and 0.010 worse on Metrica. The submission uses the World Cup-only detector.

## 5. Provided assets we stopped using

- **Smart Tagging**: its phases have fixed windows and 32 of 59 set pieces have no team. Exploratory
  analysis only.
- **"TV not live" as a dead-ball signal**: it separates well on the match (the TV is not live in 73%
  of the frames we call dead, against 14% of the live ones), but there is no external broadcast data
  to fit or tune it on, so using it would mean tuning on the provided game.
- **Vetoing broadcast windows whose ball disagrees with the tactical one**: the disagreement came from
  a ball in the air projecting differently from two cameras. Replaced by a flag; coverage would have
  fallen from 62% to 54%.
- **Cutting broadcast shots by track churn**: the broadcast tracker keeps its tracks across cuts.

## 6. What we would try with more time

- Retune the hyperparameters for the second-touch labels (the search was run on the old labels).
- Find external broadcast data, so that replays and camera cuts can be fitted as a dead-ball signal.
- Short possessions of the other team (median 2.6 s) are still the main miss. Even with no switching
  cost, more than half are never the network's most likely team, so the gain has to come from the
  inputs, not from the decoder.
