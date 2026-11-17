# Ball possession from tracking: technical document

*More than a Hack 2027, Phase 1. Team: **Gallax AI**. 17 November 2026. Code:
<https://github.com/gerardco7/gallax-ai-possession>. Numbers are from `docs/experiments.md`, the full
record; everything we dropped is summarised in `docs/what-did-not-work.md`.*

## Summary

For every frame of FC Barcelona – Athletic we output H, A, D or X with a temporal convolutional
network (TCN) that reads the tactical tracking 5 times a second, decoded with a Viterbi pass that
makes a change of possession cost something. The network is trained on **external data only**:
the 64 games of PFF FC's World Cup 2022 and 20 games of SkillCorner's open broadcast tracking, with
possession labels derived from each provider's events by the challenge's own definition (a touch
without control does not change possession). A small touch detector, trained on the World Cup, adds
per team the probability that one of its players is touching the ball. Twenty networks (5 seeds ×
4 folds) are averaged. On the 64 World Cup games and the 20 SkillCorner games, each scored by networks
that never saw it, the system reaches **spellF1 0.732 / macroF1 0.908** (World Cup) and
**0.772 / 0.895** (SkillCorner). Nothing is trained, tuned or chosen on
the provided game or its ground truth.

## 1. Approach

**Input.** The provided tactical tracking (ATD, 25 fps): 22 players and the ball in pitch metres.
Where the tactical ball is missing, it is filled from the provided broadcast tracking. The TV
recording is aligned to the tactical video automatically, by matching where the players stand
(`possession/broadcast_sync.py`), and the broadcast ball is used only where the broadcast is live
and agrees with the tactical one.

**Features** (`possession/temporal.py`), averaged over 0.2 s steps and computed per half:
- ball: seen or not, position along and across the pitch in the attacking direction, speed, distance
  to the nearest line, out of play;
- per team: distance of its three nearest players to the ball, how many are within 2 m and 5 m,
  centroid, spread, pace, and the touch detector's probability that one of its players touches the
  ball (its maximum over the step).

**Touch detector** (`possession/touch_detector.py`). A small TCN (64 channels, ±2.5 s) at 10 fps over
contact features (the ball's speed, acceleration and turn; per team the two nearest players, how fast
the nearest closes in, its speed relative to the ball, players within 1 and 2 m), supervised by PFF's
on-the-ball events (start, and end when held 0.4 s). Four fold detectors; each game's probabilities in
training come from the one that did not see it, and the match gets their average. With the ball seen
it finds 75% of the touches at 81% precision on held-out World Cup games (the best hand-written rule:
53% at 32%). Trained on SkillCorner's player possessions as well it detects SkillCorner's touches
far better (80% at 84% against 56% at 54%), but as an input it is worse on the World Cup and on
Metrica, so the submission uses the World Cup one.

The attacking direction is inferred from the data, and team identity is removed by training on
home/away-swapped copies.

**Model.** A non-causal TCN: 128 channels, 6 dilated residual blocks seeing ±13 s. Its outputs are
P(H), P(A) and P(D) per step. Training uses:
- random 10-minute crops;
- the ball removed in blocks (so the network learns to work without it), and feature noise;
- a loss weighted ×4 within 1 s of a label change;
- possessions under 5 s weighted more, each step of one at least 1 + 2 × (min(5 s / its length, 5) − 1),
  so a 1 s possession weighs 9 and a 4 s one 1.5 (`bench_temporal.py --short-weight 2`);
- early stopping on held-out games.

Hyperparameters come from a random search on World Cup games (`possession/tune_temporal.py`,
configuration 30), with the width then doubled from 64 to 128 channels.

**Decoding.** The probabilities of the 20 networks are averaged, then a Viterbi pass picks the H/A/D
sequence with switching cost 7 (chosen on held-out World Cup folds) per half.

**Why this and not something else.** Every alternative was measured on the same 64 games, game by
game with 95% intervals:
- rule-based methods (control rules, an HMM, a touch detector and votes between them) scored
  0.40 or less;
- an LSTM, a conv + transformer hybrid and a pure transformer were all below the TCN, and averaging
  them with it did not help (they share 73-77% of its errors);
- hand-written touch features, a 25 fps network and an adaptive switching cost did not help either.

What moved the score most was the **training target**. Labels that count every touch train the network to
chase possessions the challenge does not count. Switching to the definition's "second touch" rule,
with seed averaging on top, took the World Cup score from 0.639 to 0.717 (see §4 and
`docs/experiments.md`). Weighting the short possessions in the loss and doubling the width took it
to 0.729: the network's probabilities get sharper, so the decoder can afford a higher switching cost
and invents fewer possessions (35 per game against 41).

The last step is the **training data's regime**. The World Cup's tracking has the ball in 64% of live
play; the provided match has it in 93% (with the broadcast filling the tactical ball), and so does
SkillCorner's open data (89%). Adding SkillCorner's 20 games to the training (13 per fold network,
the 3 s around every restart left out of the loss, because SkillCorner starts a restart's possession
at its reception) leaves the World Cup level (0.730 against 0.729) and gains +0.055 on SkillCorner's
held-out games (20 of 20), where it also finds more of the short possessions. Part of that gain is the
network learning SkillCorner's own labelling; the rest is the regime the match is in.

The touch detector's probabilities as inputs (§1), at the same switching cost: World Cup +0.003
(95% CI −0.001 .. +0.006, 34 of 64), SkillCorner +0.012 (+0.006 .. +0.018, 15 of 20), Metrica +0.016
(each of its 3 games, whose labels come from neither provider). Hand-written touch features had not
helped; a learned detector gives the network the touches it was missing in short possessions.

## 2. Assets used, and dropped

| Asset | Used for | Status |
|---|---|---|
| Tactical tracking (ATD) | the model's input | **used** |
| Broadcast tracking + homography | aligning the TV to the tactical video; filling the missing ball | **used** |
| Video (tactical and broadcast) | watched to understand the data; never an input, never compared with our output | not an input |
| Smart Tagging | exploratory analysis only | **dropped**: phases have fixed windows, 32 of 59 set pieces have no team |
| "TV not live" as a dead-ball signal | rule-based methods | **dropped** from the final system: no external broadcast data to train or tune it on |
| Public 5-minute ground truth | checking the pipeline and the file; one logged look per fixed version | never trained, tuned or chosen on |

## 3. Training data

| Dataset | What | Labels: source, who, how much | Licence |
|---|---|---|---|
| **PFF FC World Cup 2022** | 64 games, broadcast tracking at 29.97 fps resampled to 25 (ball only where seen, players visible or estimated); training (48 games per fold network) and held-out testing (16 per fold) | Derived by our code (`possession/external_pff.py`) from PFF's event data, which PFF's analysts annotated: possession from on-the-ball events with the second-touch rule; dead ball from out-of-play and end events and before restarts. 64 games, ~244 possessions per game | Released by PFF FC free of charge to anyone through a request form (2024). See the note below |
| Metrica Sports sample data | 3 games, used only as a secondary test bench and for the plan-B network; **not in the submission's training** | Derived from Metrica's event data by the same rules | Metrica Sports' public sample data (github.com/metrica-sports/sample-data) |
| **SkillCorner open data** | 20 Australian A-League 2024/25 games of broadcast tracking (10 fps, resampled to 25), the ball seen in 89% of live play like the provided match; training (13 games per fold network) and held-out testing (5 per fold) | Derived by our code (`possession/external_skillcorner.py`) from SkillCorner's player possessions (dynamic events) with the same second-touch rule; dead ball from game interruptions and the ball leaving the pitch | MIT (SkillCorner and PySport) |
| IDSSE / DFL (Bassek et al., 2025) | 7 Bundesliga games, used as an extra test bench (the World Cup network scores 0.626 there without training on it); training on it did not help | DFL's own per-frame BallPossession / BallStatus | CC BY 4.0 |

No label of any kind comes from the provided game.

**Note on PFF's licence.** PFF FC published the dataset for free download through a request form, and
asked users to share their findings and tag the company. Neither the announcement nor the
specification PDFs that ship with the data state licence terms, and the files are marked "PFFA
Acquisition LLC proprietary information". We asked PFF (fchelp@pff.com) to confirm that the data may
be used in this competition and have had no answer by the deadline. Anyone can obtain the data the
same way we did, so the organisers can inspect it, as RULES.md requires. The released weights are
trained on it, so they carry the same status. If PFF's terms turn out to exclude this use, our
licence-clean fallback is the network trained on Metrica or IDSSE data only, which scores about 0.43
on the World Cup (§6).

## 4. The possession rules

Taken from the challenge's definition, applied to the external labels and therefore learned:

- **A change of possession needs control.** In live play, the other team takes possession only
  when its next on-the-ball event is also its own, or its player holds the ball ≥ 1 s. This drops
  ~93 single-event "possessions" per World Cup game: rebounds, one-touch passes, clearances and
  challenges. The change is placed at the first touch of the confirmed run.
- **Loose and contested balls stay with the last team in control.** This follows from the rule
  above.
- **Restarts.** After a stoppage, the first on-the-ball event gives possession (throw-ins, goal
  kicks, free kicks, corners, kick-offs).
- **Dead ball** from the ball going out or the end of a period, and before every restart from the end
  of the previous on-the-ball event, which covers fouls and goals.
- **Aerial duels** won without the winner's team controlling the ball afterwards do not change
  possession, since a single touch is not a run.

With the rule, the World Cup's possession shape (4.1 possessions per live minute, median 8.7 s)
matches the aggregates of the public window (4.2, median 7.6 s). The rule was taken from the text,
not tuned on the window.

## 5. Results

| System (World Cup, 64 games, second-touch labels, held-out folds) | spellF1 | macroF1 |
|---|---|---|
| Organisers' baseline (`baseline_smooth50`) | 0.34 | — |
| Best rule-based vote | ~0.40 | — |
| TCN, one seed | 0.702-0.711 | ~0.899 |
| TCN, 5 seeds × 4 folds (previous submission) | 0.717 | 0.902 |
| + short possessions weighted, 128 channels, one seed | 0.720 | 0.904 |
| + short possessions weighted, 128 channels, 5 seeds × 4 folds | 0.729 | 0.906 |
| + SkillCorner in the training, 5 seeds × 4 folds | 0.730 | 0.907 |
| **+ touch detector inputs, 5 seeds × 4 folds (submission)** | **0.732** | **0.908** |

| System (SkillCorner, 20 games, held-out folds) | spellF1 | macroF1 |
|---|---|---|
| TCN, 5 seeds × 4 folds, World Cup only (the first submission's recipe, 64 channels) | 0.689 | 0.866 |
| + short possessions weighted, 128 channels, World Cup only | 0.711 | 0.868 |
| + SkillCorner in the training | 0.766 | 0.893 |
| **+ touch detector inputs (submission)** | **0.772** | **0.895** |

| System (Metrica sample data, 3 games, second-touch labels) | spellF1 | macroF1 |
|---|---|---|
| First submission's recipe | 0.661 | 0.869 |
| + short possessions weighted, 128 channels | 0.663 | 0.870 |
| + SkillCorner in the training | 0.682 | 0.873 |
| **+ touch detector inputs (submission)** | **0.697** | **0.878** |

Short possessions weighted and 128 channels against the first recipe: +0.013 on the World Cup (95% CI
+0.010 .. +0.016, 53 of 64 games) and +0.013 on SkillCorner (+0.007 .. +0.019, 18 of 20) before it
was trained on. SkillCorner in the training: World Cup +0.0005 (95% CI −0.003 .. +0.004), SkillCorner
+0.055 (+0.046 .. +0.065, 20 of 20). Touch detector inputs, both at cost 7: World Cup +0.003 (−0.001 ..
+0.006, 34 of 64), SkillCorner +0.012 (+0.006 .. +0.018, 15 of 20), Metrica +0.016 (3 of 3).

On the public 5 minutes, looked at once with each version fixed beforehand: 0.681 / 0.857 for this
one, 0.696 / 0.862, 0.681 / 0.867 and 0.696 / 0.866 for the three before it, all 10 possessions where
there are 17. On 17 possessions one
possession moves spellF1 by about 6 points, so the window can neither confirm nor contradict the
World Cup gain.

## 6. Limitations

- **Too few possessions.** 209 per World Cup game against 244, 10 against 17 in the public window
  and 196 over the whole match (figures below from the World Cup-only network; the submission's are
  within a point: it finds 69% and invents 40 per game). On the World Cup the system finds
  68% of the true possessions (IoU >= 0.5). Of the 79 misses per game, 35% are the other team kept,
  36% merged into a longer possession of the same team, 24% called dead. Merged and other-team misses
  are mostly one error: a short possession of the other team (median 2.6 s) that the network does
  not see. With the ball seen over 75% of a possession, as on most of the match, 82% are found, but
  only 41% of the short ones.
- **Missing ball.** With the ball unseen most of a possession, the network tends to call it dead. On
  the provided match the broadcast fills most gaps (90% of the public window has a ball).
- **Domain shift.** The model is trained on broadcast tracking from another provider and another
  competition. Its confidence on the match (median 0.97 against 0.97 on the World Cup) and the TV
  check (dead in 73% of the frames where the TV is not live, against 14% of the live ones) suggest it
  transfers, but they cannot prove it.
- **Labelling conventions.** SkillCorner's possessions start at a reception where PFF's start at the
  touch, and its dead ball follows its own interruptions; restarts are masked out of its loss, but
  other differences remain, and the ground truth of the match follows neither provider.
- **Dependence on PFF's data.** Without it, the best licence-clean model (trained on IDSSE or
  Metrica only) scores about 0.43.
- **With more time:** retune the hyperparameters for the second-touch labels; use the broadcast's
  replays and cuts as a dead-ball signal, if external broadcast data can be found to fit it on.

## 7. Reproduction

Environment: Python 3.12.3 on Linux x86_64, CPU only. The versions are pinned in `requirements.txt`:
numpy 2.5.4, pandas 3.1.1, torch 2.15.0 and matplotlib 3.11.2. They are the versions recorded in the
checkpoints' run manifests.

From a fresh clone, with the data bundle in `data/` and the weights from the repository's `v1.0`
release in `models/` (`gallax-ai-models-v1.tar.gz`, SHA-256 `56ba6816…f942c`; the 20 networks and,
in `detector/`, the four touch detectors):

```bash
pip install -r requirements.txt
python reproduce.py --models models --expect c495d19ed0692c48c6b58c46c8c88e53ea57262f6115ef55a1b8deec438b5139
```

`reproduce.py` aligns the broadcast to the tactical video, which takes about 25 minutes on one CPU core
the first time. It then runs the four touch detectors, loads the 20 networks, decodes, writes
`Gallax_AI.csv` through the validated exporter and prints its SHA-256. We ran exactly this from a
fresh clone, with nothing derived from the data left over, and got the submitted file byte for byte.

`possession/temporal.py`, `periods.py` and `explore.py` are committed with CRLF line endings and
marked `-text` in `.gitattributes`.
Each checkpoint records the SHA-256 of the exact source it was trained with and refuses to load
against any other bytes.

To retrain (GPU, about 40 minutes per network seed), convert the
World Cup and SkillCorner data, train the touch detectors, then the five seeds:

```bash
python possession/external_pff.py
git clone --depth 1 https://github.com/SkillCorner/opendata external/skillcorner   # tracking via Git LFS
python possession/external_skillcorner.py
python possession/bench_temporal_sc.py --labels second_touch --short-weight 2 --hp channels=128 --sc-weight 1 --seed 0   # the base runs the comparison reads
python possession/touch_detector.py --sources wc --res10
python possession/bench_touch_input.py --labels second_touch --short-weight 2 --hp channels=128 --sc-weight 1 --detector wc_res10 --seed 0   # ... up to --seed 4
```

A GPU-retrained network scores the same on the benches but is not bit-identical: only the released
weights reproduce the submitted file.
