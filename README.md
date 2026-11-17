# Gallax AI — ball possession from tracking

Our submission to **More than a Hack 2027, Phase 1** (Metrica Sports · FC Barcelona · MWCapital):
for every one of the 199,319 frames of FC Barcelona – Athletic, who has the ball (`H` home, `A` away,
`D` dead ball, `X` abstain).

A temporal convolutional network reads the provided tactical tracking 5 times a second. Where the
tactical ball is missing, it is filled from the provided broadcast tracking, which is aligned to the
tactical video automatically. A learned touch detector adds touch probabilities as inputs, and a Viterbi
pass makes each change of possession cost something. The network is trained **only on external
data** (PFF FC World Cup 2022 and SkillCorner open data). Nothing is trained, tuned or chosen on the
provided game or its ground truth.

| | spellF1 | macroF1 |
|---|---|---|
| PFF World Cup 2022, 64 games, held-out folds | 0.732 | 0.908 |
| SkillCorner open data, 20 games, held-out folds | 0.772 | 0.895 |
| Metrica sample data, 3 games (never trained on) | 0.697 | 0.878 |

- **Technical document:** [`docs/technical-document.md`](docs/technical-document.md)
- **What we tried and dropped:** [`docs/what-did-not-work.md`](docs/what-did-not-work.md)
- **Full experiment log:** [`docs/experiments.md`](docs/experiments.md)

## Reproduce the submission

The submitted file is `Gallax_AI.csv`, with SHA-256
`c495d19ed0692c48c6b58c46c8c88e53ea57262f6115ef55a1b8deec438b5139`.

**1. Environment.** Python 3.12.3 on Linux x86_64. Everything runs on the CPU: a GPU's floating point
would not give the same bytes.

```bash
git clone https://github.com/gerardco7/gallax-ai-possession.git
cd gallax-ai-possession
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # numpy 2.5.4, pandas 3.1.1, torch 2.15.0, matplotlib 3.11.2
```

Or, with [uv](https://docs.astral.sh/uv/), skip the venv and prefix the command in step 4 with
`uv run --no-project --python 3.12.3 --with-requirements requirements.txt`.

**2. Match data.** Unpack the competition's data bundle into `data/`, or point `MTAH_DATA_DIR` at
it. It is licensed for the competition only and is never committed here.

**3. Trained weights** (20 possession networks and 4 touch detectors, 30 MB), from this repository's
release:

```bash
mkdir -p models
curl -L -o models.tar.gz https://github.com/gerardco7/gallax-ai-possession/releases/download/v1.0/gallax-ai-models-v1.tar.gz
echo "56ba68167bd70adb36430a162cc3cc951df1403241911a61adee78126c9f942c  models.tar.gz" | sha256sum -c
tar -xzf models.tar.gz -C models
```

**4. Run.**

```bash
python reproduce.py --models models --expect c495d19ed0692c48c6b58c46c8c88e53ea57262f6115ef55a1b8deec438b5139
python evaluation/validate_submission.py Gallax_AI.csv
```

The first run aligns the broadcast to the tactical video (about 25 minutes on one CPU core) and saves the
result to `data/derived/`. After that, a run takes about a minute. `--expect` fails unless the output is
byte-identical to the submitted file.

> `possession/temporal.py` is committed with CRLF line endings on purpose, and `.gitattributes` keeps
> git from converting it. Each checkpoint records the SHA-256 of that exact file and refuses to load
> against any other bytes. Do not re-save it with LF endings.

## Retrain from scratch

This needs a GPU, about 40 minutes per seed, and the external data:

```bash
# PFF FC World Cup 2022: request it from PFF FC and unpack it into external/pff_wc2022/
python possession/external_pff.py
git clone --depth 1 https://github.com/SkillCorner/opendata external/skillcorner   # tracking via Git LFS
python possession/external_skillcorner.py
python possession/bench_temporal_sc.py --labels second_touch --short-weight 2 --hp channels=128 --sc-weight 1 --seed 0
python possession/touch_detector.py --sources wc --res10
for s in 0 1 2 3 4; do
  python possession/bench_touch_input.py --labels second_touch --short-weight 2 --hp channels=128 \
    --sc-weight 1 --detector wc_res10 --seed $s
done
```

Trained on a GPU, networks are deterministic up to floating point, so retrained weights score the
same on the benches but are not bit-identical to the released ones. The released weights are what
reproduce the submitted file.

## Layout

```
reproduce.py                 one command from data to Gallax_AI.csv
possession/
  temporal.py                the TCN: features, training, Viterbi decoding
  touch_detector.py          the learned touch detector
  touch_inputs.py            touch probabilities as network inputs, at inference
  broadcast_sync.py          aligns the TV broadcast to the tactical video
  frames.py, periods.py      common tracking format, half-safe slicing
  bench*.py                  training and benches on the external games
  external_pff.py            PFF World Cup 2022 -> frames + second-touch labels
  external_skillcorner.py    SkillCorner open data -> frames + labels
  submission.py              validated CSV export
  control.py, hmm.py, vote.py, gaps.py, touches.py, methods.py   rule-based methods (benches only)
evaluation/                  the organisers' scorer and validator, unchanged
starter/                     the organisers' loaders, unchanged
games.py                     the organisers' match constants, unchanged
```

## Licence

Our code is MIT (see `LICENSE`). The organisers' scorer, starter code and `games.py` keep their own MIT
licence. The match data is not covered by either. The status of the external training data, and so of
the released weights, is stated in §3 of the technical document.
