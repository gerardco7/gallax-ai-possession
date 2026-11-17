# Experiment log

What was tried, what it measured and what was kept. Every number here comes from the external games
(Metrica Sports sample games 1-3) unless it says otherwise; "realistic" is the degraded copy of
those games built by `frames.degrade`. Scores are spellF1, the competition's ranking metric.

## Pending

- **Temporal network (TCN):** first version measured (see the World Cup bench); next, a higher
  epoch cap, training on Metrica too, and checking its transfer to the provided match. A small non-causal temporal convolutional network over
  per-frame features at 5 fps with ~±50 s of context, H/A/D probabilities, Viterbi on top; trained
  on the World Cup games (folds by game) plus the Metrica games.
- **Transformer** variant of the same network, on the same features and folds (the LSTM is measured,
  see the World Cup bench). It needs more data than a TCN, so it is the riskiest on 64 games.

## Kept

| Step | Realistic spellF1 (bench at the time) | Notes |
|---|---|---|
| Organisers' baseline (nearest player, ±2 s majority vote) | 0.341 | reference |
| Gap classifier (stoppage vs lost play) | 0.360 | `gaps.py` |
| Control rule, R = 2.0 m, T = 0.8 s | 0.472 | `control.py`, tuned by `tune_control.py` |
| HMM over H / A / D | 0.472 | `hmm.py`, tuned by `tune_hmm.py` |
| Still-ball stoppage classifier (threshold 0.6) | 0.391 → 0.439 for the control rule on the bench with a visible ball in stoppages | `gaps.py` |

| Touch detector (third method, threshold 0.85) | 0.480 on the smoothed bench (control 0.442, HMM 0.446) | `touches.py` |
| Vote of the three, switching cost LAMBDA = 20 | 0.480 (leave-one-game-out 0.480; best voter 0.480) | `vote.py`, `tune_vote.py` |
| Vote with probabilities, LAMBDA_SOFT = 5 | 0.494 (leave-one-game-out 0.477) | `vote.combine_soft`, `tune_vote.py --soft` |

The two vote rows are measured without the HMM delay below (it was removed); with it they were
0.489 and 0.520.

### Smoothing the realistic bench's ball (0.4 s moving average)
The touch detector first scored 0.497 on the realistic bench but found 4-5x fewer touches per
minute on the provided match than on the external games. The cause: the external games' corrected
ball travels in straight lines between events, with a sharp kink at each one, and the bench's noise
was added on top without removing the kinks; the provided match's automatic tracker filters the
ball, so its velocity changes are far gentler (p99 of the change over 0.16 s: 5.9 m/s against 18 on
the bench). The bench now smooths its ball over 0.4 s -- a generic filter length, chosen before
looking -- which brings that p99 to 8.5-9 m/s (check only; nothing was tuned on the match's
figures). Re-tuned after the change: the control rule, HMM and delay keep their parameters (each
grid is flat within 0.006 of its best); the touch detector holds 0.480 at threshold 0.85 and now
finds ~2x fewer touches per minute on the match than on the bench (was 4-5x). Its transfer to the
match is the least certain of the three methods.

### Timing (signed latency of predicted transitions, external games)
The HMM places team-to-team changes 0.72 s (clean) / 0.84 s (realistic) early: the evidence leans
towards a team as its player approaches the ball, before he controls it, and Viterbi uses the whole
sequence. A fixed delay corrected it (+0.015 on both benches, 0.7 s) but was **removed** by
decision: a constant shift of the output is a correction of the symptom rather than of the model,
and we preferred not to rely on it. The control rule backdates a change to where control began and
its median latency is 0.00 s.

Restarts are on time (median 0.00 s, 76-98% within ±0.5 s). Stoppages are on time on the clean
bench (+0.04 s) but 1.1-1.4 s early on the realistic one, because the realistic copy makes the
tracker lose the ball up to 3 s before a stoppage and the gap classifier calls the stoppage from
the moment the ball disappears. That is our own simulation's assumption, so it is **not** corrected:
tuning a delay to it would fit the assumption, and how early the real tracker loses the ball is
not something we can measure without the ground truth. Listed as a limitation.

The realistic bench changed twice (smooth instead of per-frame ball noise; a visible ball in 70% of
stoppages), so scores from different rows are not comparable with each other.

## World Cup bench (PFF FC, 64 games, automatic tracking as delivered)

`bench_pff.py`: four folds of 16 test games, models trained on 16 other games per fold, nothing
degraded (the raw broadcast tracking already loses the ball in play and the player near it).
Hyperparameters as chosen on the Metrica bench. Mean spellF1 over 64 games (standard error ~0.007),
and the paired difference with the control rule game by game:

| Method | spellF1 | vs control rule (95% CI) | better than control in |
|---|---|---|---|
| Vote with probabilities, LAMBDA 10 | **0.405** | +0.009 (+0.006 .. +0.012) | 48 / 64 games |
| Vote with labels, LAMBDA 20 | 0.398 | +0.002 (−0.002 .. +0.005) | 38 / 64 |
| Control rule | 0.397 | — | — |
| HMM | 0.381 | −0.016 (−0.021 .. −0.011) | 12 / 64 |
| Organisers' baseline | 0.336 | −0.061 | 2 / 64 |
| Touch detector | 0.307 | −0.089 (−0.098 .. −0.080) | 0 / 64 |

**Hyperparameter search** (`tune_temporal.py`; the 16 games of fold 3 kept as a reserve; the other 48
tune in two halves of 24 that take turns; the defaults plus 40 random configurations, Viterbi cost
1/2/5/10 each). What mattered, by the mean of each configuration's best score: the loss weight of
steps within 1 s of a change (1: 0.557, 2: 0.566, 4: 0.579), clearly the strongest; 32 channels too
few (64 and 128 equal); 8 layers better than 6 or 9 on average; removing the ball in 20% of the
crops slightly better than none; a constant learning rate better than cosine; noise, kernel,
dropout and crop length had little effect. On the reserve, trained on all 48 tuning games:
configuration 30 (64 channels, 6 layers, dropout 0.2, lr 4.4e-4, batch 32, ball removed in 20% of
crops, noise 0.05, weight 4 near changes) 0.647, +0.036 over the defaults (95% CI +0.029 .. +0.043,
16/16 games); configuration 15 0.642 (+0.031, 16/16); configuration 16 0.629 (+0.018, 13/16). On the
provided match, for information only, configuration 30 finds 312 possessions against 278 for the
defaults with the same separation by the TV. **Configuration 30 adopted** as the defaults.

**Retrained with configuration 30 on the four folds** (64 games, same folds): TCN spellF1 **0.636**,
macroF1 0.897, at switching cost 5 (still the best). +0.023 over the TCN with the previous defaults
(95% CI +0.019 .. +0.027, better in 56/64) and +0.014 over the previous TCN + LSTM (51/64). The LSTM
retrained with the same recipe got *worse* (0.582 at cost 2, −0.054 vs the TCN, 0/64), and averaging
now costs: TCN + new LSTM 0.630 (−0.005, 26/64); TCN + the previous-default LSTM 0.632 at cost 1
(−0.004, 28/64). The TCN's own gain covers what the LSTM used to add. **Kept: the TCN alone, cost 5.**
Three networks averaged per fold (TCN c30 + new LSTM + hybrid transformer): 0.623 at cost 2,
−0.013 vs the TCN alone (95% CI −0.016 .. −0.010, better in 10/64); with the pure transformer instead
0.612 (−0.024, 1/64). Dropped.
**Why averaging no longer helps** (step errors at 5 fps on the 64 test games, each game scored by its
own fold's networks): TCN wrong on 9.4% of steps, LSTM 10.8%, hybrid transformer 10.3%. 77% of the
TCN's errors are shared with the LSTM (73% with the transformer); where the TCN is wrong the LSTM is
right on 2.2% of steps, but where the LSTM is wrong the TCN is right on 3.6%. Half of the TCN's errors
fall within 1 s of a true change (15% of the steps) and three quarters with the ball unseen (58% of
the steps). The TCN has the best spellF1 of the three in every one of the 64 games, so even picking
the best network per game would add nothing.

**Weighted averages, weight and cost chosen on the other three folds** (TCN weight 0.5..1.0, cost
1..5): with the LSTM +0.003 over the TCN at cost 5 (95% CI +0.001 .. +0.005, 37/64); with the
transformer the other folds always choose weight 1.0, the TCN alone, at cost 3: +0.004 (+0.002 ..
+0.006, 45/64). The gain is the switching cost, not the partner: the TCN alone at cost 3 is 0.639
against 0.636 at cost 5 (cost 3 was missing from the earlier grid). No partner added.
**Seed ensembles** (4 more TCNs per fold, seeds 1-4, same recipe; seed 0 is the saved network):
single networks at cost 3 score 0.639, 0.632, 0.625, 0.634, 0.627 by seed, mean 0.631. The saved one
is the luckiest draw. Averages of the first n seeds at cost 3: 2 → 0.640, 3 → 0.639, 4 → 0.641,
5 → 0.640. Against seed 0 alone that is +0.001 to +0.002 and not significant (5 seeds +0.0007, 95%
CI −0.002 .. +0.003, 35/64). Against a typical single seed it is about +0.009. The ensemble does not
beat the network we have, but it removes the luck of the draw: a retrained single network would
land anywhere in 0.625-0.639.
Pure transformer (attention only, same recipe, ~21 min per fold): 0.516 at cost 2, −0.120 vs the
TCN (0/64); with the TCN 0.609 (−0.026, 1/64). Dropped.

Transfer to the provided match (fold ensemble, aggregate only): 254 possessions; median step
confidence 0.972 against 0.968 on World Cup games; TV not live in 72.8% of frames labelled dead
against 13.7% of those labelled alive; shares H 0.34, A 0.23, D 0.43.

**Bidirectional LSTM** (`bench_temporal.py --arch lstm`; same features, folds and training; 2 layers,
64 units per direction, 167,875 parameters, ~200 s to train per fold): spellF1 0.599 at switching
cost 2, 0.014 below the TCN (95% CI −0.019 .. −0.009; better in 21/64 games). **Averaging the TCN and
the LSTM** of each fold: spellF1 **0.621**, macroF1 **0.898**, +0.009 over the TCN (95% CI +0.006 ..
+0.012; better in 51/64) -- unlike adding the weak rule-based voters, two strong networks that read
time differently do help, a little.

**Transformer (conv + attention hybrid)** (`bench_temporal.py --arch transformer`; two convolutions,
then 4 encoder layers, 4 heads, width 64, attention within 5-minute windows that slide with 50%
overlap at inference; 176,771 parameters; previous default training recipe, crops of one window):
spellF1 0.573 at switching cost 2, 0.039 below the TCN (95% CI −0.044 .. −0.035; better in 1/64). It
learns more slowly (~15 min per fold alone on the GPU against ~1 min for the TCN) and its validation
loss stayed above the TCN's. Averaged with the TCN: 0.611, no gain over the TCN alone (−0.002,
31/64) -- unlike the LSTM, it adds nothing the TCN lacks.

**Plan B: the TCN trained on the Metrica games only** (`bench_metrica_tcn.py`; 3 games, each with 8
realistic copies; three networks, each holding one game out to stop, averaged), scored on the 64
World Cup games, which it never learns from: spellF1 0.441 at switching cost 5. Better than the
vote with probabilities by +0.036 (95% CI +0.028 .. +0.043, 55/64 games) and than the control rule
by +0.044 (56/64), but 0.172 below the network trained on the World Cup (0/64). Without the PFF
licence the fallback is this network rather than the rule-based methods.

**Rule-based methods re-tuned on the World Cup** (`tune_pff.py`: 32 games to tune, 32 held out).
Chosen: still-ball threshold 0.7, R 2.0 m, T 0.6 s, ALPHA 0.2, BETA 1.0, LAMBDA 20 for both votes
with all three voters. On the 32 held-out games against the Metrica-tuned values: control rule
+0.005 (0.404), vote with labels +0.004 (0.404), vote with probabilities +0.008 (0.411), HMM
unchanged. **Not applied to the code**: the code keeps the Metrica-tuned values, so that the
rule-based methods remain a fallback with no PFF dependency should the PFF licence not be confirmed.

**Temporal network (TCN)** (`temporal.py`, `bench_temporal.py`), same folds, trained on the other
48 games (42 fit + 6 to stop), Viterbi switching cost 2: **spellF1 0.526, macroF1 0.869**, better
than every earlier method in all 64 games -- +0.121 (95% CI +0.113 .. +0.129) over the vote with
probabilities, +0.130 over the control rule, +0.190 over the baseline. Each fold trains in ~20 s
on the GPU, and training was still improving when it hit the 60-epoch cap. It finds ~220
possessions per game where there are 358, more than any other method.

Second version: the training halves held on the GPU and cropped there, four random crops per half
per epoch, validation every second epoch, Viterbi at the network's 5 fps; the epoch cap raised to
300 with early stopping (patience 20). The folds stopped by themselves at epochs 86-128, training
in 61-82 s each (the whole bench, evaluation included, still ~6.5 minutes). **spellF1 0.612,
macroF1 0.894** (switching cost 2 or 5), +0.207 (95% CI +0.198 .. +0.216) over the vote with
probabilities, better in all 64 games; ~263 possessions per game where there are 358.

Third version: the "players tracked" feature removed. On the World Cup games it was always 11
(estimated off-camera players are included) while the provided match tracks ~10 per team, so the
network would have read a value it had never seen (28 standard deviations away). spellF1 0.615,
unchanged; every remaining feature is within 0.42 standard deviations of the World Cup's on the
match. Transfer checks on the match (no ground truth): confidence as on unseen World Cup games
(median 0.983 against 0.991), 239 possessions, and the TV is not live in 73.5% of the frames it calls
dead against 13.6% of those it calls alive -- the cleanest separation of any method.

**The touch detector does not survive real tracking**: worse than the baseline, and worse than the
control rule in every one of the 64 games, although it was the best method on the Metrica bench
(0.480). What it learned there were the corrected ball's sharp kinks at events, which neither real
tracking nor the bench's smoothing reproduces well enough. Every method finds about 180-230
possessions per game where there are 358.

 (validation only, version fixed before each look, nothing changed after)

| Version | spellF1 | macroF1 | possessions (truth 17) |
|---|---|---|---|
| Organisers' baseline | 0.326 | 0.621 | 26 |
| Baseline + broadcast | 0.360 | 0.586 | 23 |
| Gap classifier + carry + broadcast | 0.359 | 0.612 | 22 |
| Control rule + broadcast | 0.447 | 0.672 | 17 |
| Control rule + still balls + broadcast | 0.467 | 0.808 | 19 |
| HMM + still balls + 0.7 s delay + broadcast | 0.487 | 0.728 | 20 |
| *after smoothing the bench ball (models retrained, parameters unchanged):* | | | |
| Control rule + broadcast | 0.467 | 0.809 | 19 |
| HMM + broadcast | 0.389 | 0.757 | 20 |
| Touch detector + broadcast | 0.537 | 0.755 | 18 |
| Vote of the three + broadcast | 0.556 | 0.825 | 19 |

| Vote with probabilities + broadcast | 0.463 | 0.768 | 21 |
| *after removing the HMM delay:* | | | |
| HMM + broadcast | 0.357 | 0.746 | 20 |
| Vote of the three + broadcast | 0.544 | 0.821 | 19 |
| Vote with probabilities + broadcast | 0.453 | 0.770 | 21 |

The official bootstrap (`run_evaluation.py --bootstrap 1000`) puts the 5-95% interval of spellF1 at
0.39-0.70 for the vote with labels and 0.33-0.60 for the vote with probabilities, and the scorer
itself says gaps below ~0.33 are within noise on these 17 possessions. The two votes are
indistinguishable here; the choice between them is made on the external games.

| *temporal network (TCN), trained on the World Cup games only, 4 fold networks averaged, broadcast ball, switching cost 5:* | | | |
| TCN + broadcast | **0.696** | **0.889** | 10 |

| LSTM + broadcast (4 fold networks, switching cost 2) | 0.648 | 0.899 | 12 |
| TCN + LSTM + broadcast (8 networks averaged, switching cost 2) | 0.619 | 0.883 | 14 |

On the World Cup games the order is the reverse (TCN + LSTM 0.621 > TCN 0.613 > LSTM 0.599, over
64 games); here the three are within two possessions of each other on 17, inside the window's noise.

The TCN finds 10 possessions in the window where there are 17: its possessions are long (25.9 s on
average against 14.2 s). On the whole match it gives 239.

| *TCN retrained with configuration 30 (decided on the World Cup bench before this look):* | | | |
| TCN + broadcast | 0.619 | 0.840 | 14 |

Down from 0.696 here while up +0.023 over 64 World Cup games: 14 possessions instead of 10 on 17,
inside this window's noise. Kept, since the choice is made on the external games. (The LSTM and
TCN + LSTM rows above are the previous-default networks.)

| *same look, networks retrained with configuration 30:* | | | |
| LSTM + broadcast (cost 2) | 0.607 | 0.906 | 14 |
| TCN + LSTM + broadcast (cost 2) | 0.619 | 0.890 | 14 |
| TCN + LSTM + hybrid transformer + broadcast (cost 2) | 0.619 | 0.891 | 14 |

| TCN, 5 seeds × 4 folds averaged + broadcast (cost 3) | 0.607 | 0.863 | 14 |
| TCN trained on second-touch labels + broadcast (cost 5, chosen on the World Cup) | 0.696 | 0.859 | 10 |
| TCN on second-touch labels, 5 seeds × 4 folds averaged + broadcast (cost 5) | 0.696 | 0.866 | 10 |

All four networks find the same 14 possessions' worth of spells here; the averages only move
macroF1. Nothing chosen from this.

The HMM moved from 0.487 to 0.389 with the same parameters, only retrained on the smoothed bench:
on seventeen possessions that is under two possessions, and a reminder of how noisy this window is.

Seventeen possessions: one possession moves spellF1 by about six points. These numbers track
progress; they chose nothing.

## Where the TCN fails (explore.miss_table / extra_table)

World Cup, 64 games, TCN fold networks at cost 3. The TCN finds 58% of the 358 true possessions per
game (same team, IoU >= 0.5): 17% of those under 2 s, 49% at 2-5 s, 75% at 5-10 s, 88-92% above
10 s. **82% of the misses last under 5 s.** Why they are missed: other team kept (a missed change)
39%; merged into a longer possession of the same team 34%; called dead 21%; boundaries shifted 6%.
Possessions that begin after a change are found 52% of the time; those after a dead ball 74%. With the
ball seen under 25% of the possession only 36% are found, and 20% are called dead; with it seen over
half the time, 70-78%. Matched possessions start 1.3 s and end 1.1 s off on average. The TCN also
invents 72 possessions per game: 38 are a split of one team's possession, 24 a change that did not
happen, 11 under a dead ball. Dead spells under 2 s are found 5% of the time, 2-5 s 54%.

Public five minutes (seed ensemble, aggregates only; `notebooks/errors_public.ipynb` has the detail
for a person to read): 9 of 17 found. Of the 8 misses, 4 are the other team kept (all under 5 s), 3
merged and 1 shifted. All 8 begin after a change, and all have the ball seen over 75% of the time
(broadcast ball). None are called dead. 5 invented possessions: 2 changes that did not happen, 3
splits.

So the failure is the same in both: short possessions after a change. The ball is not the problem
there. The network sees the ball and still does not switch team for a few seconds.

### The same breakdown on second-touch labels (submitted 5-seed ensemble)

`python possession/errors_wc.py --labels second_touch`: World Cup, 64 games, each decoded with the
average of its fold's five seed networks at cost 5, judged against the second-touch labels. Found
68% of the 244 true possessions per game (58% of 358 before; the labels no longer count the
single-touch ones). By length: 14% under 2 s, 43% at 2-5 s, 71% at 5-10 s, 86-91% above 10 s.
**64% of the 79 misses per game last under 5 s.** Why: other team kept 35%, merged into the same
team's longer possession 33%, called dead 26%, boundaries shifted 6%. After a change 64% found,
after a dead ball 73%; a missed change is mostly after a change (17% of those against 3%).

The ball decides "called dead": with it seen under 25% of the possession (28% of all possessions on
these broadcast data) only 36% are found and 25% are called dead; seen over half the time, 82-84%
are found and 1% called dead. **56% of all misses are in possessions where the ball is seen under a
quarter of the time.** The public window has the ball 90% of the time, so the World Cup score
understates what to expect there and the misses that matter on the match are wrong team and merged.

Matched possessions start 1.6 s and end 1.3 s off on average, with no bias (medians -0.04 and 0.00 s).
Invented possessions fell from 72 to 41 per game: 21 split one team's possession, 13 are a change
that did not happen, 5 sit under a dead ball. Dead spells under 2 s are found 4% of the time, 2-5 s 49%,
5-10 s 74%.

So: the second-touch labels removed most invented possessions and the single-touch chases. What
remains is the same short possession after a change (seen ball, the network does not switch, or
switches and merges it), plus possessions with no ball, which the provided match mostly does not have.

### Width and change weight on top of k = 2 (`bench_temporal.py --hp`)

Seed 0, k = 2, one setting changed at a time (`bench_short_weight.py 2 2_channels96 ...`); cost chosen
on the other folds, 64 World Cup games:

| setting | spellF1 | macroF1 | poss/game | minus k = 2 seed 0 |
| --- | ---: | ---: | ---: | --- |
| k = 2 (64 channels, change weight 4) | 0.7140 | 0.9008 | 198 | |
| 96 channels | 0.7153 | 0.9013 | 199 | +0.0013 |
| **128 channels** | **0.7205** | **0.9040** | 204 | **+0.0066** |
| 192 channels | 0.7202 | 0.9033 | 210 | +0.0062 |
| 256 channels | 0.7132 | 0.9009 | 203 | -0.0008 |
| change weight 2 | 0.7147 | 0.9016 | 207 | +0.0007 |
| change weight 8 | 0.7089 | 0.8983 | 201 | -0.0051 |

Against seed 0 without k, 128 channels is +0.0123 (95% CI +0.0077 .. +0.0168), better in 50/64; with
width chosen on the other three folds, three folds pick 128 and one 192. Width tops out at 128-192 and
falls at 256 (more to fit, same 64 games). 128 channels on one seed finds more possessions (67.4% vs
66.2%, under 5 s 34.9% vs 32.6%) and invents a few more (39.5 vs 36.4 per game).

**The 5-seed ensemble at 128 channels, k = 2** (`bench_short_weight_seeds.py 2_channels128`, out of
fold, cost chosen on the other folds — 10 in every fold) against the submitted ensemble:

| ensemble | spellF1 | macroF1 | poss/game | found (under 5 s) | invented/game |
| --- | ---: | ---: | ---: | --- | ---: |
| submitted (64 ch, k = 0, cost 5) | 0.7162 | 0.9016 | 204 | 67.7% (35.2%) | 40.8 |
| k = 2, 64 ch, cost 7 | 0.7216 | 0.9031 | | | 38 |
| **k = 2, 128 ch, cost 10** | **0.7291** | **0.9059** | 201 | 67.8% (35.6%) | **35.3** |

spellF1 +0.0129 (95% CI +0.0097 .. +0.0161), better in 53/64; macroF1 +0.0043 (+0.0035 .. +0.0051),
better in 59/64. Against the 64-channel k = 2 ensemble, +0.0075. Width and k add up: the ensemble
finds as many possessions as the submitted one and invents 5.5 fewer per game, with merges taking a
larger share of the remaining misses (36% vs 33%). The best candidate for the submission so far; not
yet in `temporal.py`, not retrained on the LF checkout.

**Public five minutes, one logged look** (version fixed on the World Cup before looking: k = 2, 128
channels, five seeds, cost 10, broadcast ball; `bench.py --submit tcn_second_touch_sw2_ch128_seeds_sync`):
spellF1 0.681, macroF1 0.867, 10 possessions where there are 17. The submitted ensemble scores 0.696 /
0.866 and the 64-channel k = 2 ensemble 0.696 / 0.857. The same 10 possessions, one of them now
matched at fewer IoU thresholds (the drop is a fraction of one possession, about 6 points each); frame
by frame it is the best of the three. On 17 possessions this neither confirms nor contradicts the
World Cup gain. Over the whole match it predicts 196 possessions (submitted 218, 64-channel k = 2
211; the match should hold about 250). Not chosen from.

**Adopted as the submission** (9 November): `reproduce.py` now builds `tcn_second_touch_sw2_ch128_seeds_sync`
on the CPU, deterministically (two runs, the same bytes): `Gallax_AI.csv`, sha256
`464eea03893ddcf58eefb42fb2a2fc9f667db9d5785aa232986ab4dbfea35950`, 196 possessions over the match.
Chosen on the World Cup gain alone (+0.0129, 53/64); the public look above played no part.

On top of 128 channels (seed 0), neither more depth nor more dropout helps: 7 layers (receptive
field doubled to ~51 s) 0.7149, dropout 0.3 0.7127, against 0.7205; every fold keeps plain 128.
k = 3 at 128 channels ties k = 2 (0.7205 both; the folds split 2-2).

**Counts per World Cup game** (out of fold): 244 true possessions. The submitted ensemble predicts 206
(165 found, 41 invented) and misses 79 (27.5 other team, 26 merged, 20 called dead, 4 shifted); the
128-channel k = 2 ensemble predicts 201 (166 found, 35 invented) and misses 79 (28, 28, 19, 4).

### Merged and wrong-team misses: the network does not see them (`bench_rescue.py`)

On the 128-channel k = 2 ensemble at cost 10, the network's probability of the right team inside each
true possession, by how it was decoded:

| reason | mean p | max p | steps where it is the top class | mean length |
| --- | ---: | ---: | ---: | ---: |
| found | 0.84 | 0.97 | 91% | 18.6 s |
| merged | 0.78 | 0.90 | 91% | 6.8 s |
| wrong team | 0.33 | 0.47 | 24% | 3.6 s |
| called dead | 0.25 | 0.44 | 17% | 7.8 s |

In 46% of the wrong-team misses the right team is never the top class at any step. A merged
possession is seen; what is not seen is what separated it from its neighbour: of the 17.6 hidden gaps
per game between two true possessions of one team inside one predicted spell, 15.2 are a possession
of the other team (median 2.6 s) and 2.4 a dead ball (median 4 s). So the two misses are mostly one
error: a short possession of the other team that the network does not see, costing three possessions
(it, and the two it separated). A decoder cannot recover what the probabilities do not hold: a
separate cost for H<->A changes and for changes into or out of D, both chosen on the other folds
(every fold: 10 and 7), gives +0.0004 (95% CI -0.0004 .. +0.0012), better in 27/64. Dropped.

**The loss instead** (`bench_temporal.py --sandwich S`, 128 channels, seed 0): a short run between two
runs of the other team (dead balls aside) weighs S times its k = 2 weight. S = 1.5: 0.7216 (+0.0011
over plain 128 channels, three folds of four pick it); S = 2: 0.7173. Both find more short possessions
(under 5 s found 36.8% and 38.6% against 34.9%) and invent more (43.9 and 46.1 per game against 39.5,
mostly the other team): the network now sees some of the short possessions, and sees them where there
are none as often. Inside seed noise; not adopted.

**Most short misses have no ball.** Short (< 5 s) World Cup possessions per game by the share of their
frames with the ball seen: under 25%, 44 (30% found); 25-75%, 20 (44%); over 75%, 14 (41%). The
provided match has the ball about 90% of the time, so the World Cup score is dominated by a regime
the match mostly lacks, and with the ball seen 56-59% of short possessions are still missed.
Short possessions after a change are found 28% of the time (49 per game), after a dead ball 48% (29).

**Ball coverage per World Cup game**: 29-54% of live frames (median 41%); no game is near the
provided match's ~90%, so a "high-coverage games" view does not exist. Per possession instead
(128-channel k = 2 ensemble, cost 10), the ones with the ball seen over 75% of their frames, 71 per
game, are found 82% of the time with 4.7 invented per game; the misses among them are 7.4 merged and
4.6 wrong team, and the short ones (14 per game) are found 41%. `bench_touch_cost.breakdown` now
prints this ball-seen share alongside the totals, as the closest view of the match's regime.

**Two decode-only checks, both dropped** (out-of-fold probabilities of the same ensemble):
- Cleaning the decoded labels (a team flip under 1 s back to the team around it, a dead ball under
  3 s inside one team's possession removed, a live run under 4 s between dead balls made dead):
  flips under 1 s never survive cost 10 (no change); the dead rule −0.0008 (CI −0.0014 .. −0.0003);
  the live rule −0.0081, all three −0.0089 (7/64). The decoder already does what these rules do.
- A cheaper change where the ball is seen (cost lam × f on steps with the ball in over half their
  frames; lam 5-20, f 0.3-1.5, chosen on the other folds): every fold picks f = 1, the cost as it is.
  The short possessions the network misses are not near-misses the cost hides.

**Ball-seen steps weigh more in the loss** (`bench_temporal.py --seen-weight W`: every step where the
tracker has the ball weighs W times its k = 2 weight; 128 channels, seed 0, cost 10 on every fold).
W = 2: 0.7206 against 0.7205 (+0.0000), macroF1 0.9029 against 0.9040; possessions with the ball
seen over 75% found 82.1% against 81.3%, the short ones among them 43.3% against 41.1%; invented 38.9
per game against 39.5. W = 3: 0.7168, the same ball-seen gain and worse overall. W = 2 moves recall
towards the match's regime at no cost on the World Cup total, inside one seed's noise; checked on a
second seed before anything is decided.
Seed 1: 0.7212 against 0.7198 (+0.0014, CI −0.0022 .. +0.0050, 38/64); ball seen over 75% found
82.1% against 81.1%, the short ones 42.6% against 39.5%; invented 40.3 per game against 38.0. Two seeds
agree on the direction where the ball is seen (+0.8 and +1.0 points, short ones +2.2 and +3.1) and on
a tie overall; seeds 2-4 trained for an ensemble comparison.
**Five seeds, ensemble against ensemble** (cost chosen on the other folds; 7 or 10 for W = 2, 10 for the
submission): World Cup 0.7262 against 0.7291, −0.0029 (95% CI −0.0057 .. −0.0000), better in 27/64;
macroF1 −0.0008. With the ball seen over 75% it still finds a little more (82.0% against 81.7%, short
ones 42.3% against 41.0%) and invents more (36.7 per game against 35.3). On SkillCorner, the match's
regime: 0.7009 against 0.7018, −0.0009 (CI −0.0050 .. +0.0032), 9/20; macroF1 +0.0005. The single-seed
gain where the ball is seen does not survive averaging, and it is paid for in invented possessions.
**Dropped.**

**Ball visibility, every dataset against the challenge match** (share of frames inside the halves with
a ball position; live and dead by each dataset's labels, and for the match by our own submitted labels,
since its ground truth is not ours to use):

| | games | all | live | dead |
|---|---:|---:|---:|---:|
| World Cup (PFF) | 64 | 41.8% | 64.5% (48-75%) | 9.0% |
| SkillCorner | 20 | 59.3% | 89.4% (80-94%) | 23.9% |
| IDSSE (DFL, optical) | 7 | 100% | 100% | 100% |
| IDSSE degraded (`frames.degrade`) | 7 | 92.1% | 87.9% | 97.4% |
| Metrica | 3 | 62.5% | 90.9% | 2.8% |
| challenge match, tactical ball | 1 | 59.3% | 88.8% | 20.5% |
| challenge match, + broadcast ball (our input) | 1 | 63.0% | 92.7% | 23.8% |

The "~90%" quoted above for the match is its live play. In live play the World Cup has the ball 64.5%
of the time and the match 92.7%; SkillCorner (89.4% live, 59.3% overall, 23.9% dead) is the match's
regime almost exactly, Metrica too in live play but on 3 games, and the degraded IDSSE copy has the
ball where it should not (97% of dead time). SkillCorner is the case for training on more than the
World Cup.

### Does a team that comes to the ball win it? (`possession/steal.py`) — dropped

The idea: the short possessions of the other team that the network misses might be visible as the
moment that team reaches the ball. Every time a player of the team *not* in possession (by the
128-channel k = 2 ensemble's out-of-fold decoding at cost 10) comes within 2 m of a seen ball, at a
local minimum of that team's nearest distance (±0.4 s), is a candidate; it is positive when a true
possession of that team starts within 0.6 s. Features of the moment and the 3 s after it (distances,
players within 2 and 5 m, ball speed and turn, how long the player stays near, who is closer after
1-3 s, how far the ball advances, position) plus the ensemble's own probabilities; a gradient-boosted
classifier trained out of fold on the networks' folds.

414 candidates per game, 37 positive (9%). Out-of-fold AUC 0.920, against 0.894 for the network's own
probability of that team in the 1-3 s after; at score 0.5, 22 per game with precision 0.65. But 90% of
the positives it scores over 0.5 are possessions the decoder already finds. The ones it was for, short
possessions decoded as the other team (22 per game): only 28% have a candidate at all, because the
ball is mostly not seen, and 1% score over 0.5. With the ball seen over 75% (4.2 per game) 70% have a
candidate, and its best score has median 0.09 (90th percentile 0.41). At the moment the team reaches
the ball, these possessions look like the challenges that do not win it; the tracking at hand does
not tell them apart. In the match's regime this miss is also small: 4.2 of the 71 possessions per
game with the ball seen. Dropped.

### Are the World Cup labels in sync with the tracking? Yes

If PFF's event times were off the tracking by half a second or more, a 2 s possession would be half
misplaced and the network could not learn short ones. Measured on every on-the-ball (OTB) event of the
64 games (2119 per game, 110 of them starting a second-touch possession under 5 s), on the grid the
labels are built on:
- the event's own ball snapshot against the tracking ball: median 0.36 m apart at the event time, and
  no other offset within ±2 s is closer;
- the event team's nearest player to the tracking ball (ball seen across ±2 s): median 0.80 m at the
  event time, and the median profile over offsets is lowest at +0.08 s (events starting a short
  possession: 1.00 m, lowest at 0.00 s);
- the team is within 2 m of the ball within ±0.2 s of the event for 94-95% of events, short
  possession starts included.

The labels are aligned to the tracking to within a frame or two, short possessions as well as long
ones. What does not line up is the ball: across ±2 s of an event that starts a short possession, it is
seen throughout only 4% of the time (26% for all events). Not the cause; nothing to fix.

### A change-of-team output head (`bench_change_head.py`) — dropped

Configuration 30 with one more 1x1 output: the logit of "a change between H and A within 0.4 s".
Trained on the k = 2 possession loss plus ALPHA times a binary cross-entropy on it (positives weighted
POS_WEIGHT); early stopping still on the possession loss. temporal.py untouched: the training loop is
temporal.train's with the extra term. Decoded two ways, chosen on the other folds: `aux` (the head
only shapes training) and `steer` (a change costs lam * exp(-gamma * P(change))). Seed 0, against the
k = 2 network of seed 0 (0.7140):

| ALPHA, POS_WEIGHT | P(change) near a change / elsewhere | aux | steer |
|---|---|---|---|
| 1, 5 | 0.47 / 0.06 | 0.6998 (−0.0142, CI −0.0185 .. −0.0099) | 0.7014 (−0.0125) |
| 0.2, 1 | 0.16 / 0.02 | 0.7042 (−0.0097, CI −0.0141 .. −0.0054) | 0.7037 (−0.0103) |
| 0, 1 (control) | — | identical to the reference at every shared cost | |

The control proves the loop is faithful (bit-identical scores at costs 5, 10 and 20; both trainings
are deterministic, weights equal to the last bit on a rerun), so the loss itself does the harm. The
head learns its task, and the possession output gets worse (macroF1 0.895-0.897 against 0.901): the
extra objective pulls the shared 64 channels towards local change detection, which the possession
output did not lack. Steering the cost with it recovers at most +0.0017 of the loss. Same lesson as
B and the contact features: telling the network or the decoder *where* changes happen does not help;
what helped (k = 2) changed which errors the possession loss cares about.

**Focal weighting of the possession loss** (`--alpha 0 --focal G`: each step's cross-entropy times
(1 - p_true)^G on top of the k = 2 weights) — dropped. G = 1: 0.6862 (-0.0277, better in 7/64);
G = 2: 0.6800 (-0.0340, better in 2/64); both pick the lowest cost in the grid (5) and predict fewer
possessions (188, 178 vs 198). Focal weighting takes weight off the easy steps, which here are the
middle of long possessions; the network then stops being sure inside them and the decoder pays for
it in merges. k = 2 already puts the weight where it is needed, on the short runs, explicitly.

### Several sampling phases at inference (`bench_phases.py`)

The network reads 0.2 s blocks, and where they start is arbitrary. The k = 2 ensemble out of fold, run
5 times with the blocks of each half starting 0-4 frames later, the probabilities averaged at 25 fps
and decoded at 25 fps, against phase 0 alone decoded the same way (0.7217, the same as decode_steps'
0.7216). Cost chosen on the other folds (7 for both): spellF1 0.7218 against 0.7217, +0.0001 (95% CI
−0.0008 .. +0.0010), 31/64; macroF1 +0.0001. Nothing: at ±13 s of context and a cost that smooths
over seconds, the 0.2 s grid is not where anything is lost. Dropped.

### Transfer to labels we did not make: IDSSE (`bench_transfer_idsse.py`)

The submitted and the k = 2 ensembles on the 7 IDSSE games, all 20 networks each (no IDSSE game was
in any fold), at the cost chosen on the World Cup (5 and 7), against DFL's per-frame possession;
nothing chosen here. Clean tracking: spellF1 0.5356 against 0.5347 (+0.0009, better in 4/7), macroF1
0.9074 against 0.9070. Ball removed like broadcast tracking (`frames.degrade`, seed 3001): 0.4716
against 0.4707 (+0.0010, 3/7), macroF1 0.8417 against 0.8394 (6/7). Neutral: k = 2 costs nothing
on independent labels and gains nothing visible. DFL counts 437 possessions per game against our
~230, because its possession follows every touch (the event-label network, which also counts every
touch, scored 0.626 here), so the definition gap swamps a gain the size of k = 2's. IDSSE checks that
a change does no harm; it cannot referee one this small.

### Option B: a cheaper change towards the team that touches the ball (`bench_touch_cost.py`)

Decode only, on the submitted ensemble's out-of-fold probabilities, judged on second-touch labels.
A touch is a peak of the smoothed ball's acceleration whose nearest player is within R m; entering
that team's possession within 0.2 s of it costs lam * exp(-gamma) instead of lam (a directed Viterbi;
gamma 0 reproduces the submitted decoder exactly, checked). Grid: lam 5, 7, 10; gamma 0-4; threshold
20, 30 m/s^2; R 1.5, 3 m. 14% of steps carry a touch.

The score falls steadily with gamma: best cell per gamma 0.7170 (0), 0.7163 (0.5), 0.7155 (1),
0.7145 (2), 0.7137 (4). Chosen on the other folds, every fold picks gamma 0 (lam 5 or 7):
−0.0010 against the submitted decoder (95% CI −0.0023 .. +0.0004), better in 6/64. Possessions per
game barely move (207 against 206 at gamma 0.5). Dropped.

Same reading as the adaptive cost on event labels, now with a signal from outside the network:
on the short possessions it misses, the network gives the other team too little probability for any
reasonable change cost to switch, and where a cheaper cost does switch it switches wrongly as often
as rightly. The fix, if any, is in the network (option A), not the decoder.

### Option A: short possessions weigh more in the loss (`bench_temporal.py --short-weight k`)

Every step of an H or A run under 5 s weighs at least 1 + k * (min(5 s / its length, 5) - 1) in the
loss (5k + 1 for a 1 s possession, 1 at 5 s), on top of the x4 near label changes. Second-touch labels,
configuration 30, seed 0, the 4 folds; the weight is swapped in from bench_temporal.py so that
temporal.py, whose hash the submitted checkpoints carry, is untouched. Compared with
`python possession/bench_short_weight.py 1 2`, cost chosen on the other folds:

| k | spellF1 | macroF1 | poss/game | cost | minus seed 0 (95% CI), better in |
|---|---|---|---|---|---|
| 0 (seed 0 as submitted) | 0.7082 | 0.8990 | 207 | 5 | |
| 1 | 0.7103 | 0.8996 | 211 | 5 | +0.0021 (−0.0000 .. +0.0041), 37/64 |
| 2 | **0.7140** | **0.9008** | 198 | 10 | **+0.0057 (+0.0030 .. +0.0085), 41/64** |

k = 2 is also +0.0066 (+0.0030 .. +0.0101) against a typical seed and above all five single seeds
(0.702-0.711); choosing k on the other folds picks 2 in every fold.

Then k = 3 and 4 (`bench_short_weight.py 1 2 3 4`): 0.7129 at cost 10, +0.0047 (+0.0012 .. +0.0081),
40/64; and 0.7099 at cost 10, +0.0017 (−0.0022 .. +0.0056), 34/64. Across k the score rises and
falls (0.708, 0.710, 0.714, 0.713, 0.710): k = 2 is the top. Choosing k and the cost together on the
other folds gives only +0.0013 (−0.0017 .. +0.0044): fold 0's other folds prefer k = 3, and that
fold's k = 3 network came out poor (0.690 against 0.707 for k = 2 and 0.697 for k = 0). With one
seed per fold, k = 2 and k = 3 cannot be told apart, and the gain of either over k = 0 needs the
five seeds to be trusted.

**Five seeds of k = 2** (`python possession/bench_short_weight_seeds.py 2`), the same 5 seeds × 4
folds ensemble as the submission, against the submitted ensemble, both with the cost chosen on the
other folds:

| ensemble | spellF1 | macroF1 | poss/game | cost |
|---|---|---|---|---|
| submitted (k = 0) | 0.7162 | 0.9016 | 204 | 5 or 7 |
| k = 2 | **0.7216** | **0.9031** | 204 | 7 in every fold |

+0.0054 spellF1 (95% CI +0.0029 .. +0.0080), better in 46/64; +0.0015 macroF1 (+0.0010 .. +0.0021),
51/64. k = 2 is ahead at every cost from 3 to 20. The gain survives the ensemble rather than
overlapping with it. At its cost it finds 67.8% (under 5 s 35.6%) against 67.7% (35.2%), and
invents 38 possessions per game against 41.

**Public five minutes, one logged look** (version fixed on the World Cup before looking: k = 2, five
seeds, cost 7, broadcast ball; `bench.py --submit tcn_second_touch_sw2_seeds_sync`): spellF1 0.696,
macroF1 0.857, 10 possessions where there are 17 — the same spellF1 and possession count as the
submitted ensemble (0.696 / 0.866). On 17 possessions one possession is about 6 points, so this
neither confirms nor contradicts the World Cup gain. Over the whole match it predicts 211
possessions against the submitted 218. Not chosen from.

**Only the 2-5 s possessions weighted** (`--short-weight 2 --short-min 2`, seed 0; runs under 2 s
keep the plain weights): 0.7120 at cost 5, +0.0038 (+0.0014 .. +0.0061) against seed 0, 40/64,
below k = 2 on all short runs (0.7140). It does what the band was meant to do (under 5 s found 36.1%
against 33.8%) but invents more (46 per game against 44, and 36 for full k = 2). So most of k = 2's
gain comes from weighting the possessions under 2 s heavily, which sharpens the probabilities enough
for a higher cost; aiming at the 2-5 s band trades it for recall. Not kept.

The gain does not come the way it was meant to. At its chosen cost, k = 2 finds slightly *fewer*
possessions (66.2% against 66.9%, under 5 s 32.6% against 33.8%) and invents far fewer (36 per game
against 44, other-team ones 12 against 14). The weighted network's probabilities are sharper, so the
decoder can afford a higher cost (10 against 5) and drops spurious changes. The best cost moving up
with k, and k = 2 beating k = 1, say neither k nor the cost is at its best yet.

### One change at a time against the misses (`bench_variants.py`)

**Adaptive switching cost** (decode only, the saved TCN): a change costs lam * exp(-beta * e), where e
is how much the network's own probabilities move across the step (0.6 s each side). Grid lam 2-12,
beta 0-8. Chosen on the other folds, all four folds pick beta = 0, the fixed cost 3 as it is. The
best cell with beta > 0 is 0.638 (lam 5, beta 1) against 0.639. Making changes cheaper does not
recover short possessions: the network gives them no clear change in probability. Dropped.

**Touch features** (`temporal.use_touch_features`: ball acceleration and change of direction, and per
team the acceleration weighted by how close its nearest player is, 28 features instead of 24;
retrained on the 4 folds, seed 0): 0.635 at cost 3, −0.005 against the saved TCN (95% CI −0.009 ..
−0.000, 27/64). Against the five seeds of the TCN as it is (0.625-0.639, mean 0.631) it is inside the
seed spread. The failure profile is unchanged: 57% found, 17% of those under 2 s, misses 39% wrong
team and 33% merged. Not kept.

**The network at 25 fps** (`temporal.STEP = 1`: features and Viterbi on every frame; 8 layers, so it
sees ±10 s, and crops of 5 min; otherwise configuration 30, retrained on the 4 folds, seed 0, about
7 min per fold against 3): 0.624 at cost 3, −0.016 against the saved TCN (95% CI −0.021 .. −0.010,
12/64). That is below all five seeds of the TCN as it is (0.625-0.639). Short possessions are not
found more often (under 2 s 15% against 17%), and the misses split as before. Configuration 30 was
tuned at 5 fps, so a retuned 25 fps network could do better, but nothing here suggests the short
possessions are lost to the step. Not kept.

**Is ball acceleration a touch signal at all?** Peaks of the acceleration (local maxima over a
threshold, at most one per 0.2 s) against the touches in PFF's events: the start and end of every
on-the-ball event, 2,058 distinct per game, 33 per minute with the ball seen. The comparison is
against the same peaks moved 2 s later, which is what chance gives at this density.
- *Raw positions, as the touch features used them:* 120-190 peaks per seen minute. Within 0.2 s of
  a touch, precision 0.25-0.30 against 0.24 by chance. It is **noise**: differentiating raw
  positions twice turns tracking jitter into accelerations as large as a kick's. That alone
  explains why the touch features changed nothing.
- *Positions smoothed first (Savitzky-Golay, 7 frames, order 2), threshold 20-30 m/s^2:* 54-91
  peaks per seen minute. Precision 0.38-0.45 against 0.24 by chance within 0.2 s, and 0.67-0.73
  against 0.46 within 0.4 s. Recall of touches with the ball seen 0.82 / 0.62, against 0.42 / 0.27
  by chance. This is a real signal, though still about twice as many peaks as touches, and part of
  the gap is the events' own timing.
- At a peak that matches a touch, the nearest player belongs to the touching team 82-88% of the time.

So the touch features deserve one more try with the ball smoothed before differentiating, and with
each 0.2 s step keeping the maximum rather than the mean.

### Single touches in the World Cup labels (the "second touch" question)

The World Cup labels give possession to any team with an on-the-ball event, even one. The
challenge's definition says a simple touch without control, or a deflection, does not change
possession. Among the runs of consecutive on-the-ball events by one team in live play (376 per
game, `label_rules.py`), 32% are a single event, 16% two and 52% three or more.

Relabelled with the definition's own rule (in live play the other team takes possession only when
its next on-the-ball event is also its own; holding the ball 1 or 2 s changes nothing), and compared
with possessions per minute of live ball:

| labels | possessions / game | per live minute | < 2 s | median | TCN spellF1 (trained on current) |
|---|---|---|---|---|---|
| events (current) | 358 | 6.0 | 24% | 4.8 s | 0.639 |
| second touch | 243 | 4.1 | 8% | 9.0 s | 0.667 |
| public ground truth (aggregate, 17 possessions) | | 4.2 | | | |
| IDSSE / DFL | | 8.3 | | | |

Only 2% of frames change. The TCN, trained on the current labels, scores +0.028 higher against the
second-touch labels (95% CI +0.022 .. +0.034, 56/64). A third of the "possessions" it was trained and
judged on are single touches the challenge does not count, and those are exactly the short misses of
the error analysis. The rule comes from the definition's text. The public density only confirms it
(4.2 per live minute, ±1 at 17 possessions), and nothing was tuned on it.

**Does the rule throw away real possessions?** After a stoppage, the first on-the-ball event gives
possession as before, so throw-ins, goal kicks, free kicks and corners are untouched. In live play
the rule removes 93 single-event possessions per game. Of those, 62% go straight back to the
previous team and 38% end with the ball out. By PFF's action type they are rebounds 27%, one-touch
passes 24%, clearances 23%, challenges 17%, touches 4% and shots 2%. The player touched the ball
once in 88% of them (2+ touches in 2%), and held it a median of 0.0 s (1 s or more in 2%). None
involves the goalkeeper: a keeper who collects the ball holds it, or plays it to a teammate, and
either confirms the possession (the 1 s hold variant covers holding and gives the same numbers).
They are the touches without control, deflections and clearances that the definition excludes.

**Checked against the shape of the public ground truth** (aggregates only; the rule is not tuned on
it):

| labels | per live minute | direct changes per live min | < 2 s | < 5 s | median |
|---|---|---|---|---|---|
| World Cup, events (current) | 6.0 | 4.4 | 24% | 52% | 4.6 s |
| World Cup, second touch or 1 s | 4.1 | 2.5 | 9% | 32% | 8.7 s |
| IDSSE (DFL) | 8.3 | 6.4 | 30% | 61% | 3.7 s |
| public ground truth, 17 possessions | 4.2 (95% 2.2-6.2) | 3.0 | 12% | 35% | 7.6 s |

On all five measures the public window sits next to the second-touch labels and away from the
current ones and DFL's. With 17 possessions from one match, each measure alone is weak, and the
measures are not independent; together they are consistent with the definition's reading. The window
still holds possessions under 2 s (2 of 17), so the ground truth is not simply "no short
possessions".

**Retrained on the second-touch labels** (`external_pff.load(game, "second_touch")`,
`bench_temporal.py --labels second_touch`, same folds, seed and recipe), judged on the
second-touch labels with the cost chosen on the other folds: **0.708** at cost 5, against 0.703 for
the TCN trained on event labels, whose folds choose cost 10. That is +0.005 (95% CI +0.001 ..
+0.009, 42/64). Most of the gain is in judging and decoding against what the challenge counts, which
a higher switching cost already does for the old network. Retraining adds a small, real step and
finds 207 possessions per game where the labels hold 244. **Kept**: the second-touch labels are the
training and judging target from here, and the TCN trained on them is the new candidate.

**Five seeds on the second-touch labels**, retrained with the period-safe code (`bench_temporal.py
--labels second_touch --seed 0..4`, versioned runs; `bench_seeds.py`), cost chosen on the other
folds (5 nearly everywhere): single seeds 0.708, 0.702, 0.706, 0.711, 0.709, a much narrower spread
than on the event labels (0.625-0.639). Averaging the first 2, 3, 4 and 5 seeds gives 0.713, 0.715,
0.717 and **0.717** (macroF1 0.902, 206 possessions per game). Five seeds against seed 0: +0.009 (95%
CI +0.006 .. +0.012, 49/64); against a typical single seed +0.010 (+0.008 .. +0.012, 54/64). **Kept**:
`bench.submit("tcn_second_touch_seeds_sync")`, 218 possessions on the match.

## SkillCorner open data (20 A-League 2024/25 games, MIT)

`external_skillcorner.py` converts SkillCorner's open broadcast tracking (10 fps, resampled to 25 by
linear interpolation; the ball only where detected, players detected or extrapolated) and builds
second-touch labels from SkillCorner's own player possessions with the World Cup's rule (HOLD_S, the
next possession of the same team; a game interruption after a possession is dead ball from when the
ball leaves the pitch). Shape against the World Cup's second-touch labels: 199 possessions per game,
3.8 per live minute (4.1), median 10.7 s (8.7 s), under 2 s 6% (9%), under 5 s 25% (32%), dead 46%;
the ball seen 59% of the time (41%), 16 players per frame (22 on PFF, which estimates the ones out of
view). SkillCorner's possessions are a little coarser, but this is the closest independent bench to
the challenge's labels and to broadcast tracking that we have.

**The World Cup ensembles on it** (`bench_skillcorner.py`; all 20 networks of each, nothing trained or
chosen here, each at its World Cup cost):

| ensemble | spellF1 | macroF1 | possessions / game |
|---|---:|---:|---:|
| previous submission (k = 0, 64 channels, cost 5) | 0.6885 | 0.8657 | 228 |
| submission (k = 2, 128 channels, cost 10) | 0.7018 | 0.8678 | 216 |

+0.0132 (95% CI +0.0073 .. +0.0192), better in 18/20; macroF1 +0.0020, 18/20. The World Cup gain
(+0.0129) carries over at the same size to another league, provider and labelling. Both ensembles
predict more possessions than SkillCorner's labels hold (216 against 199; 68 invented per game), the
other side of SkillCorner's coarser possessions.

**What the SkillCorner games are, and what the submission misses there** (`skillcorner_errors.py`,
all 20 networks at cost 10). 98.5 minutes per game; the ball seen 59% of the time, but almost all the
unseen time is dead (replays, close-ups): 180 of the 199 possessions per game have the ball over 75% of
their frames, which is the provided match's regime and not the World Cup's (71 of 244). 16 players
per frame, 57% of them detected, the rest extrapolated. spellF1 0.600-0.809 per game.

Found 74.6% of the true possessions (World Cup 68%). Misses per game: 20.1 merged, 17.4 other team,
9.4 called dead, 3.6 shifted. Invented: 67.8 per game, 33.1 of them over dead ball, 19.4 inside a
possession of the same team, 15.3 of the other.

| | per game | found | missed per game |
|---|---:|---:|---:|
| under 2 s | 12.7 | 7% | 11.8 |
| 2-5 s | 38.1 | 53% | 18.0 |
| 5-10 s | 45.7 | 82% | 8.3 |
| 10-20 s | 49.9 | 90% | 5.2 |
| over 20 s | 52.8 | 86% | 7.4 (most called dead, with the ball rarely seen) |

By SkillCorner's phase of play of the possession's start: "chaotic" 49 per game, found 51%, 24 missed
(half of all misses), and "disruption" 6 per game, 33%; build-up, create, finish, transition and quick
breaks 83-91%. By how it starts: interceptions after a restart (throw-in, free kick, goal kick, corner
interceptions) 52-58%, recoveries and open-play interceptions 76-77%, receptions of a restart 79-82%
(corners 50%). In the attacking third 63% (merged 20%), elsewhere 75-79%. So the misses are short
possessions in scrambles, where the ball changes team without either holding it.

The 33 possessions per game invented over dead ball: 70% end within 3 s of a possession that follows
a stoppage, 36% of the same team. SkillCorner starts a restart's possession when a player receives it
(the taker of a throw-in or a free kick has no player possession), where the World Cup's labels start
it at the restart itself; the rest carry the previous team on past the ball going out. Found
possessions start a median 0.4 s before SkillCorner's (the same convention: reception against first
touch) and end 0.1 s after it. Part of what SkillCorner counts as invented is its convention, not ours.

**Training on the World Cup and SkillCorner together** (`bench_temporal_sc.py`, 128 channels, k = 2,
seed 0; fold k holds out World Cup fold k and SkillCorner games [k::4], and fits on 42 + 13 games; the
3 s around every possession start after a dead ball masked in SkillCorner's loss), against the same
seed's World Cup-only fold networks on the same held-out games, cost chosen on the other folds of each
dataset:

| SkillCorner weight | World Cup spellF1 | vs World Cup only | SkillCorner spellF1 | vs World Cup only |
|---|---:|---|---:|---|
| none (the submission's recipe) | 0.7197 | | 0.6975 | |
| 1 | **0.7261** | **+0.0064 (CI +0.0020 .. +0.0108), 44/64** | 0.7561 | +0.0587 (+0.0478 .. +0.0695), 20/20 |
| 3 | 0.7096 | −0.0100 (−0.0146 .. −0.0054), 15/64 | 0.7570 | +0.0596 (+0.0493 .. +0.0699), 20/20 |

On SkillCorner most of the gain is the network learning SkillCorner's own way of labelling: it invents
34 possessions per game instead of 61 and predicts 183 against 199 true (205 before), and its misses
move to "called dead" (32% against 17%). That part says little about the match, whose ground truth
follows the challenge's definition and not SkillCorner's. The World Cup column is the one that
matters, since its labels did not change: with SkillCorner at its natural weight the network is better
on the World Cup too (+0.0064, 44/64, macroF1 +0.0006), while weighting SkillCorner 3 times pulls the
network towards SkillCorner's conventions and costs the World Cup −0.010. Weight 1 goes on to five
seeds for an ensemble comparison with the submission.

**Five seeds.** Per seed, World Cup against the same seed's World Cup-only networks: +0.0064, −0.0007,
+0.0064, +0.0010, −0.0057 (seeds 0-4; mean +0.0015); SkillCorner +0.056 to +0.067, 19-20/20 each.
Ensemble against ensemble (the submission's 20 networks on the World Cup; on SkillCorner the same
seeds' World Cup-only fold networks on the same held-out games), cost chosen on the other folds:

| | World Cup spellF1 | macroF1 | possessions | SkillCorner spellF1 | macroF1 |
|---|---:|---:|---:|---:|---:|
| World Cup only (the submission) | 0.7291 | 0.9059 | 201 | 0.7105 | 0.8678 |
| World Cup + SkillCorner | 0.7296 | 0.9065 | 209 | 0.7659 | 0.8926 |

World Cup +0.0005 (95% CI −0.0027 .. +0.0038), better in 33/64, macroF1 +0.0006 (39/64): a tie. It
finds more where the ball is seen (83.0% against 81.7%, short ones 44.7% against 41.0%) and invents more
(40.3 per game against 35.3), at cost 7 instead of 10. SkillCorner +0.055 (20/20), mostly its own
labelling convention. The seed-0 gain was the luckier end of a spread that averages to nothing on the
World Cup. **Not adopted on the World Cup rule**; kept as the base for the touch detector, which is
trained on the same two sources.

**Public five minutes, one logged look** (version fixed before looking: World Cup + SkillCorner, five
seeds × four folds, cost 7 as the World Cup folds chose, broadcast ball;
`bench.py --submit tcn_second_touch_sw2_ch128_sc1_seeds_sync`): spellF1 0.696, macroF1 0.862, 10
possessions where there are 17, the same counts of matched, missed and invented as the submission
(0.681 / 0.867) and the one before it (0.696 / 0.866). Within a fraction of one possession of both.
Over the whole match it predicts 203 possessions (the submission 196).

**Adopted as the submission** (9 November, on the team's decision: level on the World Cup, +0.055 on
SkillCorner, the match's regime): `reproduce.py` builds `tcn_second_touch_sw2_ch128_sc1_seeds_sync` on
the CPU, deterministically (two runs, the same bytes): `Gallax_AI.csv`, sha256
`8e8552c05955c631eb73baf826a541a06423cf7db21423a334636648d372380e`.

## A learned touch detector (`touch_detector.py`, `bench_touch_input.py`)

**The detector.** Per 25 fps frame, does a player of H or of A touch the ball: a small TCN (64 channels,
±2.5 s) over contact features (the ball's speed, acceleration and turn; per team the two nearest
players, how fast the nearest closes in, its speed relative to the ball, players within 1 and 2 m),
supervised by PFF's on-the-ball events (start, and end when held 0.4 s) and, with `--sources wc+sc`,
by SkillCorner's player possessions; everything at SkillCorner's 10 fps (`--res10`). Out of fold on
both datasets, peaks matched to true touches within ±3 frames, ball seen at the touch:

| detector | SkillCorner precision / recall | World Cup precision / recall |
|---|---|---|
| hand rule (ball acceleration, nearest team within 1.5 m), threshold 0.4 | 0.59 / 0.51 | 0.40 / 0.47 |
| learned, World Cup only, threshold 0.3 | 0.54 / 0.56 | 0.81 / 0.75 |
| learned, World Cup + SkillCorner, threshold 0.3 | 0.84 / 0.80 | 0.80 / 0.75 |

Kill test on the submission's out-of-fold misses (short possessions, ball seen over 75%): the first
touch of the right team is seen (≥ 0.5) in 34-39% of the missed ones against 55-58% of the found ones,
two touches in 16-18% against 45-50%. Some signal where the steal classifier had none.

**As two more inputs of the possession network** (one per team: the detector's maximum over each 0.2 s
step, stacked out of fold; the submission's recipe otherwise; seed 0, against the same seed of
`bench_temporal_sc.py` on the same held-out games):

| detector used | World Cup | SkillCorner | SkillCorner, short found |
|---|---|---|---|
| none (seed 0 of the submission) | 0.7261 | 0.7561 | 37.0% |
| World Cup + SkillCorner | 0.7237 (−0.0024, CI −0.0059 .. +0.0012, 29/64) | 0.7760 (+0.0199, CI +0.0105 .. +0.0292, 15/20) | 43.2% |
| World Cup only | 0.7247 (−0.0014, CI −0.0054 .. +0.0025, 28/64) | 0.7689 (+0.0128, CI +0.0039 .. +0.0216, 15/20) | 41.4% |

Where the ball is seen the touches help: on SkillCorner +0.013 to +0.020, and the short possessions
found go from 37% to 41-43% with fewer invented. The World Cup-only detector never saw a SkillCorner
label, so its +0.013 there is not SkillCorner's convention leaking through the touch targets. On the
World Cup, where the ball is missing most of the time and the detector finds 13-20% of the unseen
touches, it is level; on its ball-seen possessions the World Cup-only detector finds more short ones
(49.7% against 43.2%) and invents more. Seeds 1-4 are next.

## Every version on the 3 Metrica games (`bench_metrica_versions.py`)

Metrica's sample games are in no fold of any network, so each version averages all its networks at
the cost its World Cup folds chose. Labels: second touch built from Metrica's events
(`external_metrica.possession_labels(second_touch=True)`: 4.3-5.4 possessions per live minute, median
5.5-9 s; game 3's EPTS events lose fewer single touches to the rule) and, for reference, the event
labels. The ball is seen in 91% of live play, the match's regime. Three games: a direction, not a
verdict.

| version (second-touch labels) | game 1 | game 2 | game 3 | mean | macroF1 | possessions (true 317) |
|---|---:|---:|---:|---:|---:|---:|
| 1 previous submission (64 ch, cost 5) | 0.672 | 0.685 | 0.625 | 0.661 | 0.869 | 227 |
| 2 k = 2, 128 ch (cost 10) | 0.665 | 0.697 | 0.628 | 0.663 | 0.870 | 221 |
| 3 + SkillCorner, the submission (cost 7) | 0.682 | 0.711 | 0.652 | **0.682** | 0.873 | 220 |
| 3, seed 0 only (cost 10) | 0.664 | 0.711 | 0.643 | 0.673 | 0.872 | 216 |
| 4 + touch inputs, seed 0 (cost 10) | 0.669 | 0.706 | 0.656 | 0.677 | 0.876 | 214 |

Each submitted step is a gain here too, and the largest is SkillCorner's: +0.018 over 2, better in
all three games, on data whose labels are neither PFF's nor SkillCorner's. The touch inputs against
the same seed without them: +0.004 (2 of 3 games), macroF1 +0.004. On the event labels the order is
the same (0.602, 0.605, 0.613; seed 0 0.605 against 0.609 with touches). Every version predicts about
220 possessions where Metrica's second-touch labels hold 317: the short ones are missing here as
everywhere.

## Touch inputs, five seeds: the decision (`bench_touch_ensemble.py`, `bench_metrica_versions.py`)

**Ensemble against ensemble, out of fold**, five seeds with the wc+sc detector against the submission's
five seeds, the cost chosen on the other folds of each dataset: SkillCorner +0.0134 (95% CI +0.0073 ..
+0.0195, 17 of 20), World Cup +0.0008 (−0.0031 .. +0.0048, 30 of 64), macroF1 +0.0008 (41 of 64).
Per seed on SkillCorner +0.020, +0.017, +0.009, +0.007, +0.013; on the World Cup −0.002 .. +0.004.
The touch folds chose cost 10, the submission's 7.

**First version (cost 10) failed outside the training sources.** On the 3 Metrica games 0.675 against
the submission's 0.682 (worse in 2 of 3); public five minutes, one logged look, 0.652 / 0.851 against
0.696 / 0.862, 10 possessions where there are 17, one fewer of them matched (9 against 10).

**Why: the cost, not the network.** Both ensembles at every cost (spellF1):

| | 5 | 7 | 10 | 15 |
|---|---|---|---|---|
| World Cup, submission | 0.7250 | 0.7296 | 0.7290 | 0.7204 |
| World Cup, touch | 0.7272 | 0.7296 | 0.7305 | 0.7211 |
| SkillCorner, submission | 0.7521 | 0.7597 | 0.7659 | 0.7646 |
| SkillCorner, touch | 0.7668 | 0.7739 | 0.7802 | 0.7796 |
| Metrica, submission | 0.6845 | 0.6815 | 0.6706 | 0.6552 |
| Metrica, touch | 0.6875 | 0.6869 | 0.6752 | 0.6664 |

At the same cost the touch inputs are never worse anywhere. The World Cup is flat between 7 and 10
(0.001), so its folds picked 10 by a hair, and Metrica, the external data nearest the match's ball
coverage, falls with the cost. Changing only the inputs and keeping the submission's decoding (cost 7):
World Cup 0.0000 (−0.0035 .. +0.0035, 32 of 64), SkillCorner +0.0142 (+0.0080 .. +0.0203, 18 of 20),
Metrica +0.005 (0.6869 against 0.6815, 3 of 3: +0.006, +0.006, +0.004; event labels 0.616 against
0.613), macroF1 0.9072 / 0.8952 / 0.876 against 0.9065 / 0.8925 / 0.873. With the wc-only detector
(seed 0) SkillCorner still gains +0.013, and Metrica's labels are neither provider's, so the gain is
not the detector learning SkillCorner's labelling.

**Version fixed at cost 7, then one logged look at the public five minutes**
(`bench.py --submit tcn_second_touch_sw2_ch128_sc1_touch_seeds_sync`): 0.652 / 0.851, the same as at
cost 10 (the window decodes identically), 10 possessions where there are 17, 9 matched. One
possession under the submission; on 17 possessions that is noise and plays no part. Over the whole
match it predicts 195 possessions (the submission 203).

**Adopted as the submission** (9 November, on the external evidence: level on the World Cup, +0.014
on SkillCorner, +0.005 on Metrica's three games). `reproduce.py` builds it on the CPU,
deterministically (two runs, the same bytes): `Gallax_AI.csv`, sha256
`5c41f0257f1e7ae62089eb4805daf8f4a3e31aa601e935cf1b24c699bccd327b`. Models in
`<models>/second_touch_sw2_channels128_sc1_touch/` (20 networks) and its `detector/` (4 fold
detectors).

**The World Cup-only detector, five seeds** (the leakage check, `bench_touch_ensemble.py wc_res10`),
against the submission before the touch inputs, both at cost 7: World Cup +0.0027 (95% CI −0.0010 ..
+0.0064, 34 of 64), SkillCorner +0.0118 (+0.0055 .. +0.0180, 15 of 20), Metrica 0.697 against 0.682
(+0.017, +0.011, +0.018: 3 of 3), macroF1 0.9077 / 0.8947 / 0.878. Per seed on the World Cup
−0.001, +0.007, +0.001, +0.006, +0.009. So the SkillCorner gain is not the detector learning
SkillCorner's labelling, and this detector is the better input. Directly against the wc+sc one, both
at cost 7: World Cup +0.0027 (+0.0001 .. +0.0053, 41 of 64), SkillCorner −0.0024 (−0.0077 .. +0.0030,
8 of 20), Metrica +0.010 (3 of 3). It detects SkillCorner's touches worse (0.56 / 0.54 against
0.80 / 0.84) yet helps more on the World Cup and on Metrica: the wc+sc detector's extra skill is on
SkillCorner's own touches, which the possession network already sees there.

**Version fixed (World Cup detector, cost 7), then one logged look at the public five minutes**
(`bench.py --submit tcn_second_touch_sw2_ch128_sc1_touch_wc_seeds_sync`): 0.681 / 0.857, 10
possessions where there are 17, 10 of them matched. Over the whole match 196 possessions.

**Adopted as the submission in place of the wc+sc detector** (9 November, on the World Cup and
Metrica). `reproduce.py`, two runs, the same bytes: sha256
`c495d19ed0692c48c6b58c46c8c88e53ea57262f6115ef55a1b8deec438b5139`. Models in
`<models>/second_touch_sw2_channels128_sc1_touch_wc/` and its `detector/`.

## Short possessions after the touch inputs: where the misses are (9 November)

Everything below on the submission's out-of-fold probabilities (World Cup, SkillCorner) and the 3
Metrica games through its 20 networks, at cost 7. No network trained.

**The detector overriding the decoder** (`bench_override.py`): a run of ≥ 2 same-team touch peaks
pulls the step probabilities towards that team. Every setting is level: World Cup −0.0001 (nested,
6 of 64), SkillCorner 0.0000, Metrica 0.0000. Why: where the detector sees such a run the decoder
already says that team (98% of the run on the World Cup, 95% SkillCorner, 99% Metrica); the network
has absorbed the detector. And inside the short possessions we miss there is rarely a run to use
(3% of them on the World Cup, 21% SkillCorner, 12% Metrica; on Metrica, with the ball seen 98%, no
peak ≥ 0.5 of the right team in 56% of them). Features built from the same peaks (two last touches,
time since each team's last touch) were dropped for the same reason.

**How much the decoder could give**, short (< 5 s) possessions found against the switching cost:

| | cost 0.5 | 2 | 4 | 7 | 10 |
|---|---|---|---|---|---|
| World Cup short found | 47.5% | 45.4% | 42.8% | 38.7% | 36.2% |
| World Cup invented / game | 98 | 66 | 51 | 41 | 34 |
| SkillCorner short found | 54.2% | 53.6% | 51.7% | 49.6% | 46.3% |
| Metrica short found | 40.8% | 37.0% | 36.3% | 34.5% | 32.8% |

Even with almost no cost more than half the short possessions are never the network's most likely
team. The decoder is not the limit.

**A touch-gated decoder** (`bench_touch_gate.py`): switching to a team costs FAR − (FAR − NEAR) × that
team's touch probability, to and from dead ball 7. First with either team's touch: a cheaper switch
near a touch never helps (every pass is a touch). Per team: FAR 10, NEAR 4 chosen on the World Cup,
+0.0020 there (95% CI +0.0001 .. +0.0040, 44 of 64), SkillCorner −0.0006 (11 of 20), Metrica +0.0021
(3 of 3). Nearly all of it is FAR alone (10/7, no touches: World Cup 0.7345 against 0.7347): a dearer
H↔A switch than dead-ball switch, which invents fewer (37 against 41 per game) and finds no more short
possessions anywhere. Not adopted: small, and not what it was for.

**What the misses are**, short possessions by their best overlap with a predicted one of the right team:

| | never predicted | IoU < 0.3 | IoU 0.3-0.5 | found |
|---|---|---|---|---|
| World Cup | 37% | 14% | 12% | 38% |
| SkillCorner | 32% | 7% | 12% | 49% |
| Metrica | 42% | 17% | 8% | 33% |

The near misses are too long (88-95%), ending 0.2-1 s late, but there is no systematic latency to
correct: the signed delay of H↔A changes has median 0.00 s on the World Cup (0.08 SkillCorner, 0.16
Metrica). In the missed ones with the ball seen, a player of the right team comes within 1.5 m of the
ball in 81% (World Cup), 83% (SkillCorner), 97% (Metrica), and half last under 2 s: the evidence is in
the tracking, the network does not call it. A better touch detector is unlikely to change that: the
wc+sc detector finds far more of SkillCorner's touches (0.80 against 0.56) and gives nothing more on
SkillCorner's possessions.

**IDSSE relabelled with the second-touch rule** (other-team segments under 1 s between two of the same
team dropped from DFL's per-frame possession): 7.4 possessions per live minute, median 4.5 s, against
4.1 and 8.8 s on the World Cup (public window 4.2). DFL splits possession by another criterion
(duels, loose balls); its raw events carry wall-clock times not synced to the frame. Not used.

**Where that leaves short possessions**: the decoder and the touch signal are spent; what is left is
what the network represents. Its team features are aggregates (three nearest distances, counts), so
who receives the ball is lost; a per-player model is the next real step, and the costly one.

## IDSSE (DFL, 7 Bundesliga / 2. Bundesliga games, CC BY 4.0)

`external_idsse.py` downloads it from figshare (2.5 GB) and converts it. TRACAB optical tracking at
25 fps: the ball and 11 players per team in every frame, 91-98 minutes per game. Labels come straight
from the ball's frames (BallStatus 0 → D, else BallPossession 1/2 → H/A), not from events. Dead
share 0.41-0.52, like the World Cup (0.42-0.52). DFL's possession is finer-grained: 340-477
possession segments per 90 minutes against 270-340 on the World Cup, live segments with median 3.7 s
(4.5 s), 13% shorter than 1 s (8%), and 30 one-team flickers under 1 s per game (13).

**Experiment 1, the World Cup TCN on the 7 games without training on them** (4 fold networks
averaged): spellF1 0.626 / macroF1 0.922 at switching cost 2 on the clean tracking (0.560-0.661 per
game); 0.557 / 0.855 on a realistic copy (frames.degrade). It finds 295 possessions per game where
DFL has 437. The network carries to another league and tracking provider at about its World Cup
level; the gap to DFL's count is mostly DFL's short possessions.

**Experiment 2, trained on IDSSE only** (each game plus 4 realistic copies; 3 networks, each holding
one game out to stop, averaged), scored on the 64 World Cup games: spellF1 0.431 at cost 5, macroF1
0.798. That is −0.010 against the Metrica-only plan B (95% CI −0.017 .. −0.003, 27/64) and −0.209
against the World Cup network. Clean optical tracking, degraded by our own simulation, does not
teach the network what broadcast tracking looks like, whichever league it comes from. As a
licence-clean plan B it is worth the same as Metrica.

**Experiment 3, World Cup folds + IDSSE** (each IDSSE game plus 2 copies added to the 42 fitting
games): 0.632 at cost 3, −0.007 against the saved TCN (95% CI −0.011 .. −0.003, 23/64). Seed 0 is
the luckiest of five seeds, though (single seeds 0.625-0.639, mean 0.631), so against a typical seed
it is level. Adding IDSSE does not help. Not kept.


## Dropped

### Clear-control shortcut for short possessions
**Idea.** Possessions under 3 s are 36% of the external games' possessions and the control rule
finds 1-10% of them, since it needs 0.8 s of control. A second path: a player within r2 of the ball,
with the nearest opponent at least `gap` m further away, takes possession after only t2 s.
**Measured.** 48 combinations (r2 0.5-1.5 m, t2 0.04-0.32 s, gap 1-4 m). Best +0.002 over no
shortcut (0.4464 vs 0.4445), leave-one-game-out 0.4356 (worse); it adds about 8 of the ~140
possessions the rule misses.
**Why it fails.** Short possessions are touches, clearances and loose balls in a crowd: a player
alone on the ball almost never happens in them, and with ~1 m of ball noise "within 0.5 m" is hard
to confirm. Matching a 1 s possession at IoU 0.5 also needs both boundaries within ~0.3 s.
**Kept as a limitation:** possessions shorter than ~3 s are largely out of reach with this data.

### Requiring 2 s of control in the external labels
**Idea.** The event-derived labels count touches without control (recovery then immediate loss) as
possessions; the challenge's definition says a touch without control does not count.
**Measured.** The public five minutes contain a 0.8 s possession and another under 2 s, so the
official ground truth does count very short possessions; a 2 s minimum would contradict it. With
17 possessions the difference in the share of short ones (12% vs ~26%) is not significant.
**Decision.** Labels left as derived from events.

### "Carry": keep the last team whenever the ball is missing
**Measured.** Ties the baseline on the clean bench (0.417 vs 0.418), loses on the realistic one
with smooth noise (0.344 vs 0.368) and never calls a dead ball (macroF1 0.79 → 0.44).
**Kept inside the gap classifier**, which carries the last team only through gaps it calls play.

### Ball-speed condition in the control rule
**Measured.** A maximum ball speed of 8 or 15 m/s never improved on no condition: with tracking
noise a ball's speed over a few frames is unreliable.

### Stacking the control rule into the HMM
**Measured.** 0.480 vs 0.472 for either alone (leave-one-game-out 0.475 vs 0.472 / 0.460), within
noise. The evidence models learned to follow the control rule almost always, since both read the
same signals. Kept in the code (`hmm_ctrl`), not used.

### Ghost-player cleaning (duplicates, fragments, off-pitch, jumps)
**Measured** (`ghosts.py`). The player nearest the ball is a ghost on ~0.3% of frames of the
provided match. The external games, whose 22 players are all real, have *more* "duplicates" (two
teammates within 0.8 m for 1 s) than the match, so a duplicate filter would delete real players.
**Not implemented.**

### Uniform vote of the temporal network and the three rule-based methods
**Measured** (`vote4.py`, World Cup folds, every voter weighted the same). Vote with labels 0.435,
vote with probabilities 0.441, against 0.613 for the network alone: −0.17 to −0.18, worse than the
network in all 64 games. Three weaker voters that make the same short-context mistakes outvote the
network exactly where its long context is right. **Dropped.** Ways that could still use them --
their outputs as extra inputs to the network, a vote weighted towards the network, or rules only
where the network is unsure -- are untested.

### Calibrating the voters' probabilities (Platt scaling, cross-fitted)
**Measured.** The HMM is overconfident (says 95-99%, right 75%); calibration fixed that on paper but
the calibrated soft vote scored ~0.505 against 0.520 uncalibrated (with the delay), with an unstable
LAMBDA, and on the provided match still produced more possessions than any of its voters (395 vs
415). The extra switching on the match is not a calibration problem: the HMM flips 21% more there,
and twice as many of its flips are undone within 3 s, because a player within 1.5 m of the ball is
rarer (44% of frames against 49-57% on the bench) -- the bench keeps all 22 players, the match
tracks ~20.5, and the missing one is often the one in the crowd at the ball. **Dropped.**

### Vetoing broadcast windows whose ball disagrees
**Measured** (`broadcast_sync.py`). The 89 windows the veto removed matched on players as well as
the rest (0.62 m); the ball disagreed because a ball in the air projects to different ground points
from two cameras. Coverage fell from 62% to 54%. **Replaced by a flag** (`broadcast_ball_ok`).

### Cutting broadcast shots by track churn
**Measured.** The broadcast tracker keeps its tracks across camera cuts: only 10 frames inside shots
swap 4 or more tracks at once. No signal to cut on.

## Decoder and reproducibility experiment register (8 November 2026)

This section replaces the three short decoder entries that came with the period-safe halves and
log-Viterbi pull requests; the historical log above remains intact. The complete [comparison register](decoder-experiment-register.csv)
records **every arm**, including the DFL and World Cup grids: candidate/reference spellF1, paired
difference, 95% CI, better/equal/worse game counts, selection protocol and decision. Its
[metadata](decoder-experiment-register.json) bind the aggregate inputs by SHA-256.

All intervals in this register are descriptive paired whole-game percentile bootstraps, 10,000
resamples, seed 20261108 reset for each comparison. DFL degradation seeds are averaged within game
before comparing seven games. Original experiment receipts retain their original intervals; small
endpoint differences arise where those studies advanced a shared RNG between comparisons.

**Scope and current decision.** These studies reuse the same saved seed-0 configuration-30 TCN
weights, not freshly retrained networks. They do not compare against the five-seed ensemble
mentioned in `AGENTS.md`. They decide, for the **event labels**: period-safe features/decoding and
probability cost **3** (now `temporal.EVENT_LAMBDA`); logs stay experimental. The default
`temporal.LAMBDA` is 5 since the second-touch labels were adopted, whose possessions are longer; these
studies predate that. No challenge-window score was run for these studies.

### 1. Half boundaries and switching cost — kept, with limits

Independent decoding, velocities and sampling blocks per half fix state leakage across adjacent
halves and retain partial tail blocks. Test the fix at unchanged cost 5 before changing the cost.
These frozen-model DFL comparisons are **secondary evidence**, not a substitute for 64-game results.

| One change / cohort | spellF1 | Paired delta | 95% CI | Better games |
| --- | ---: | ---: | --- | --- |
| Half fix at cost 5, DFL clean | 0.607571 | +0.000105 | [-0.001297, +0.001595] | 3/7 (ties 1) |
| Half fix at cost 5, DFL degraded | 0.552565 | -0.000105 | [-0.001323, +0.001102] | 3/7 (ties 0) |
| Cost 3 vs 5, corrected halves, DFL clean | 0.619978 | +0.012407 | [+0.009415, +0.015112] | 7/7 (ties 0) |
| Cost 3 vs 5, corrected halves, DFL degraded | 0.560640 | +0.008074 | [+0.004259, +0.012439] | 7/7 (ties 0) |

**Interpretation.** Keep the half fix for correctness: its isolated mean change is effectively
neutral, and no primary 64-game speed/accuracy gain is claimed for that fix alone. Keep cost 3:
the historical primary World Cup table, **before** the half fix, gives **0.639192 vs 0.635583**, delta **+0.003609**, 95% CI **[+0.001532, +0.005647]**, **better in 45/64**.
Do not mix that historical feature protocol with the corrected-pipeline World Cup reference
0.639557 used below. [Frozen historical rows](historical-worldcup-probability-costs.csv) and
[source receipt](historical-worldcup-probability-costs-source.json).

The two complete DFL replays produced byte-identical result CSVs. [Detailed half-fix report](dfl-halves.md).

### 2. Log emissions on DFL — looked at, not adopted globally

Same cached probabilities, arithmetic averaging of the four fold networks before decoding; no
new network training/inference. The log floor is 1e-8 and costs are in the same configured units.
The complete DFL grid is 0/1/2/3/5/10/20/40/80. All its clean/degraded rows, paired CIs and decisions
are in the register. Fixed log cost 3 and the separately selected arm are:

| Candidate / cohort | spellF1 | Paired delta vs probability cost 3 | 95% CI | Better games |
| --- | ---: | ---: | --- | --- |
| Fixed log cost 3, DFL clean | 0.629027 | +0.009049 | [+0.005666, +0.011943] | 7/7 (ties 0) |
| Fixed log cost 3, DFL degraded | 0.565070 | +0.004430 | [-0.000180, +0.008609] | 6/7 (ties 0) |
| Log cost chosen on other six games, DFL clean | 0.628838 | +0.008861 | [+0.005628, +0.011768] | 7/7 (ties 0) |
| Log cost chosen on other six games, DFL degraded | 0.564555 | +0.003915 | [-0.001727, +0.008605] | 6/7 (ties 0) |

**Interpretation.** Promising on the secondary provider, particularly clean tracking, but the
degraded CI includes zero and the primary World Cup result below does not support adoption.
DFL selection groups every seed of the held-out game together. [Protocol and all results](dfl-log.md).

### 3. Log emissions on the World Cup — dropped

64 games, one excluded-fold network per game, with the same weights and inputs in both arms.
Do not ensemble the four networks on a World Cup test game: three may have trained on that game.
Each historical fold has 42 fitting games, six stopping games and 16 test games. Bare historical
weights lack embedded membership; their hashes and the training-script fold assignment are recorded.
These are reused development games, not a new blind test. No network fitting occurred.

Reference: corrected halves, probability cost 3, **spellF1 0.639557**, macroF1 0.897287.

| Fixed log cost | spellF1 | Paired delta | 95% CI | Better games |
| ---: | ---: | ---: | --- | --- |
| 1 | 0.606809 | -0.032748 | [-0.036010, -0.029452] | 0/64 (ties 0) |
| 2 | 0.621989 | -0.017568 | [-0.020317, -0.014718] | 5/64 (ties 0) |
| 3 | 0.629110 | -0.010447 | [-0.012614, -0.008239] | 5/64 (ties 0) |
| 5 | 0.635400 | -0.004157 | [-0.005238, -0.003081] | 13/64 (ties 0) |
| 8 | 0.638582 | -0.000975 | [-0.002271, +0.000351] | 30/64 (ties 1) |
| 10 | 0.636522 | -0.003035 | [-0.004911, -0.001142] | 23/64 (ties 0) |
| 15 | 0.629321 | -0.010236 | [-0.013246, -0.007126] | 12/64 (ties 0) |
| 20 | 0.620996 | -0.018561 | [-0.022559, -0.014467] | 8/64 (ties 0) |
| 40 | 0.583082 | -0.056475 | [-0.062112, -0.050753] | 0/64 (ties 0) |

**Interpretation.** Fixed log cost 3 loses on all four fold means and 59/64 games. Higher costs
recover much of the loss, but none of the tested fixed costs improves the primary mean. These grid
rows are descriptive, not permission to choose a winner from test labels.

### 4. Log-cost selection — preserve the old protocol and record the required one

The original follow-up chose costs **8/8/5/5** using each network's **six stopping games**, before
scoring its 16 test games. That kept test labels out of each cost choice, but **it is not the
other-three-fold selection now required by `AGENTS.md`**. Preserve it as historical evidence.

After importing the instructions, reaggregate the already recorded grid: for each target fold,
choose the cost on the other three folds (48 games), then score its 16 games. This chooses **8 in
all four folds**. This is a **retrospective development analysis**, not a new prediction run or
fresh holdout; the grid had already been viewed.

| Selection protocol | spellF1 | Paired delta vs current | 95% CI | Better games |
| --- | ---: | ---: | --- | --- |
| Historical six stopping games | 0.637242 | -0.002315 | [-0.003602, -0.001019] | 23/64 (ties 0) |
| Required other three folds, retrospective | 0.638582 | -0.000975 | [-0.002271, +0.000351] | 30/64 (ties 1) |

**Decision.** Do not adopt log emissions: even protocol-aligned selection does not improve the
mean, and its interval includes zero. Keep probability cost 3. [Nested selected rows](worldcup-log-three-fold-selection.csv)
and [original protocols, grids, hashes and failure interpretation](worldcup-log.md).

### 5. Failure analysis and reproducibility work

For fixed log cost 3, possession precision averaged over the official IoUs declines **0.727906 →
0.680504**, while recall rises **0.571192 → 0.585850**. About 27.7 extra predicted segments buy only
5.3 extra matched spells per game. More recovered possessions do not offset spurious/poorly
matched predictions. [Diagnostic and its algebraic derivation](worldcup-log.md#what-the-extra-segments-achieve).
This is an aggregate precision/recall diagnostic, not a fabricated per-miss causal classification.
The per-cause tables of `explore.miss_table` / `extra_table` came later (see "Where the TCN fails")
and were not part of these experiments.

**Kept, engineering rather than a new football-score claim:** frozen baseline/search configurations;
input-content/feature-schema cache invalidation; source/runtime/game-split run records;
self-describing new checkpoints; explicit legacy-weight opt-in; atomic results and sealed artifact
hash checks; reuse of verified complete runs. Nineteen regression tests passed, including a small
synthetic training/save/load check. That synthetic check is not a football training experiment.
No new seed ensemble or public-window evaluation was run. Existing historical results remain intact.

Rebuild this register from committed aggregate tables, without models or match assets:

```bash
uv run --no-project --quiet --with numpy --with pandas python possession/summarize_decoder_experiments.py
```
