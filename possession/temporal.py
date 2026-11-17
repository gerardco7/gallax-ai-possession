"""A temporal convolutional network (TCN) for possession: long context, learned.

The other methods look at a second or two around each frame. This one reads ~±50 s: per-frame
features of the ball and of each team, at 5 frames per second, through eight residual layers of
dilated convolutions (dilation 1, 2, ..., 128), looking backwards and forwards -- the whole match is
available, nothing has to be decided live. It outputs P(H), P(A), P(D) per frame; Viterbi with a
switching cost (vote.combine_soft with a single voter) turns them into possessions.

Features are relative to each team's attacking direction, inferred from the data in every half
(the side a team stands on, on average, is the side it defends), so the same code reads the World
Cup games, the Metrica games and the provided match. Training swaps home and away half the time,
so the network cannot learn that "home" means anything. Seeds are fixed and the algorithms made
deterministic, so a trained model and its output are reproducible.
"""

import os
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import explore as ex  # noqa: E402
import frames as data  # noqa: E402
import vote  # noqa: E402
import periods  # noqa: E402
import provenance as records  # noqa: E402

FPS = 25
STEP = 5                      # 25 fps -> 5 fps
# Hyperparameters, overridable per call (tune_temporal.py searches them). layers L gives
# dilations 1, 2, ..., 2**(L-1): with kernel 3, 6 layers see +-63 steps = +-13 s at 5 fps, 8 layers +-51 s.
# The values are configuration 30 of tune_temporal.py's search (+0.036 spellF1 over the previous
# defaults on the 16 reserve games, better in all 16). Those previous defaults were channels 64,
# layers 8, dropout 0.1, lr 1e-3, batch 16, ball_drop 0, noise 0, trans_weight 1.
BASELINE_CONFIG = Path(HERE).parent / "configs" / "temporal-baseline-v1.json"
DEFAULTS = json.loads(BASELINE_CONFIG.read_text())

COSINE_EPOCHS = 150
MAX_EPOCHS = 300
PATIENCE = 20                 # epochs without improvement before stopping (validation every VAL_EVERY)
CROPS_PER_HALF = 4            # random crops drawn from every half in each epoch
VAL_EVERY = 2
SEED = 0
LAMBDA = 5.0                  # Viterbi switching cost for the second-touch labels, chosen out of fold on the World Cup
EVENT_LAMBDA = 3.0            # the same for a network trained on the event labels (docs/dfl-halves.md)
CLASSES = np.array(["H", "A", "D"])
BALL_FEATURES = ["seen", "ball_x", "ball_y", "ball_speed", "ball_line", "ball_out"]
# No "how many players are tracked": the World Cup games include estimated off-camera players, so
# it is always 11 there and the network would never learn what a missing player means.
TEAM_FEATURES = ["d1", "d2", "d3", "n2", "n5", "centroid_x", "spread_x", "spread_y", "pace"]
SPEED_LAG = 5                 # frames over which the ball's speed is measured (0.2 s)
TOUCH_BALL_FEATURES = ["ball_accel", "ball_turn"]
TOUCH_TEAM_FEATURES = ["touch"]


def n_features():
    return len(BALL_FEATURES) + 2 * len(TEAM_FEATURES)


def use_touch_features():
    """Add what a touch looks like: the ball's acceleration and change of direction, and per team
    how hard the ball changed near its nearest player. Off by default (bench_variants.py)."""
    if "touch" not in TEAM_FEATURES:
        BALL_FEATURES.extend(TOUCH_BALL_FEATURES)
        TEAM_FEATURES.extend(TOUCH_TEAM_FEATURES)


def feature_schema():
    return {"version": "period-safe-v1", "fps": FPS, "step": STEP,
            "ball": list(BALL_FEATURES), "team": list(TEAM_FEATURES),
            "sources": {name: records.file_hash(Path(HERE) / name)
                        for name in ("temporal.py", "periods.py", "explore.py", "frames.py")}}


def attack_signs(frames):
    """Per frame, +1 where the home team attacks towards x = length, -1 where it attacks x = 0."""
    length = frames["pitch"][0]
    sign = np.ones(len(frames["period"]))
    centroid = ex._nanmean(frames["home"][:, :, 0], 1)
    for half in (1, 2):
        m = frames["period"] == half
        if m.any():
            sign[m] = 1.0 if np.nanmean(centroid[m]) < length / 2 else -1.0
    return sign


def _team_block(xy, ball, sign, length, width):
    """Features of one team, in its own attacking direction (sign = +1 attacks x = length)."""
    d = np.linalg.norm(xy - ball[:, None, :], axis=2)
    d_sorted = np.sort(np.where(np.isnan(d), np.inf, d), axis=1)
    near = [np.clip(np.where(np.isinf(d_sorted[:, k]), 20, d_sorted[:, k]), 0, 20) / 10 for k in range(3)]
    with np.errstate(invalid="ignore"):
        n2, n5 = (d < 2).sum(1) / 3.0, (d < 5).sum(1) / 5.0
    cx = ex._nanmean(xy[:, :, 0], 1)
    v = ex.speed(xy)
    v[v > 12] = np.nan
    return np.stack(near + [n2, n5,
                            np.nan_to_num((cx - length / 2) * sign / (length / 2)),
                            np.nan_to_num(np.nanstd(xy[:, :, 0], 1) / 10),
                            np.nan_to_num(np.nanstd(xy[:, :, 1], 1) / 10),
                            np.nan_to_num(ex._nanmean(v, 1) / 5)], axis=1)


def _continuous_features(frames):
    """(n_frames, n_features()) per-frame features of one stretch of play (a half)."""
    length, width = frames["pitch"]
    ball = frames["ball"]
    seen = ~np.isnan(ball[:, 0])
    sign = attack_signs(frames)
    v = np.full(len(ball), np.nan)
    v[SPEED_LAG:] = np.linalg.norm(ball[SPEED_LAG:] - ball[:-SPEED_LAG], axis=1) * FPS / SPEED_LAG
    line = ex.distance_to_line(np.nan_to_num(ball, nan=-99.0), frames["pitch"])
    with np.errstate(invalid="ignore"):
        ball_block = np.stack([seen.astype(float),
                               np.where(seen, (ball[:, 0] - length / 2) * sign / (length / 2), 0),
                               np.where(seen, np.abs(ball[:, 1] - width / 2) / (width / 2), 0),
                               np.nan_to_num(np.clip(v, 0, 40) / 10),
                               np.where(seen, np.clip(line, -5, 20) / 10, 0),
                               (seen & (line < 0)).astype(float)], axis=1)
    blocks = [_team_block(frames["home"], ball, sign, length, width),
              _team_block(frames["away"], ball, -sign, length, width)]
    if "touch" in TEAM_FEATURES:
        accel, turn = _ball_change(ball)
        ball_block = np.concatenate([ball_block, np.stack([np.clip(accel / 50, 0, 2), turn], axis=1)], axis=1)
        for k, team in enumerate(("home", "away")):
            with np.errstate(invalid="ignore"):
                d1 = np.nanmin(np.linalg.norm(frames[team] - ball[:, None, :], axis=2), axis=1)
            blocks[k] = np.concatenate([blocks[k], (np.clip(accel / 50, 0, 2) * np.exp(-np.nan_to_num(d1, nan=99.0) / 1.5))[:, None]], axis=1)
    with np.errstate(invalid="ignore"):
        per_frame = np.concatenate([ball_block] + blocks, axis=1)
    per_frame = np.nan_to_num(per_frame)
    return per_frame


def features(frames):
    """Period-safe features and centres, including partial blocks at each half's end."""
    period = frames["period"]
    per_frame = np.zeros((len(period), n_features()), dtype=np.float32)
    for a, b in periods.slices(period):
        part = dict(frames, **{key: frames[key][a:b] for key in ("period", "ball", "home", "away")})
        per_frame[a:b] = _continuous_features(part)
    starts, stops, centres = periods.blocks(period, STEP)
    steps = np.asarray([per_frame[a:b].mean(0) for a, b in zip(starts, stops)], dtype=np.float32)
    return steps.reshape(-1, n_features()), centres


def _ball_change(ball, lag=2):
    """Per frame, the ball's acceleration (m/s^2) and change of direction (0..1 of a half turn),
    from its velocity `lag` frames before and after; 0 where the ball is not seen or barely moving."""
    vel = np.full_like(ball, np.nan)
    vel[1:] = np.diff(ball, axis=0) * FPS
    before, after = np.full_like(vel, np.nan), np.full_like(vel, np.nan)
    before[lag:] = vel[:-lag]
    after[:-lag] = vel[lag:]
    accel = np.linalg.norm(after - before, axis=1) * FPS / (2 * lag)
    nb, na = np.linalg.norm(before, axis=1), np.linalg.norm(after, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(before * after, axis=1) / (nb * na)
        turn = np.where((nb > 1) & (na > 1), np.arccos(np.clip(cos, -1, 1)) / np.pi, 0)
    return np.nan_to_num(accel), np.nan_to_num(turn)


def swap_teams(x):
    """The same situation with home and away exchanged (features), for training symmetry."""
    y = x.copy()
    nb, nt = len(BALL_FEATURES), len(TEAM_FEATURES)
    y[..., 1] = -x[..., 1]  # ball x in the other team's direction
    y[..., nb:nb + nt], y[..., nb + nt:] = x[..., nb + nt:], x[..., nb:nb + nt]
    return y


class Block(nn.Module):
    def __init__(self, channels, dilation, kernel, dropout):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel, dilation=dilation, padding=dilation * (kernel - 1) // 2)
        self.mix = nn.Conv1d(channels, channels, 1)
        self.norm = nn.GroupNorm(8, channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.mix(self.drop(torch.nn.functional.gelu(self.norm(self.conv(x)))))


class TCN(nn.Module):
    def __init__(self, hp=None):
        super().__init__()
        hp = dict(DEFAULTS, **(hp or {}))
        c = hp["channels"]
        self.inp = nn.Conv1d(n_features(), c, 1)
        self.blocks = nn.Sequential(*[Block(c, 2 ** i, hp["kernel"], hp["dropout"]) for i in range(hp["layers"])])
        self.out = nn.Conv1d(c, len(CLASSES), 1)

    def forward(self, x):  # x: (batch, time, features)
        return self.out(self.blocks(self.inp(x.transpose(1, 2)))).transpose(1, 2)


class BiLSTM(nn.Module):
    """The alternative to the TCN: a bidirectional LSTM reads each half forwards and backwards with
    memory, instead of looking through fixed dilated windows. Same inputs and outputs."""

    def __init__(self, hp=None):
        super().__init__()
        hp = dict(DEFAULTS, **(hp or {}))
        h = hp["lstm_hidden"]
        self.inp = nn.Linear(n_features(), h)
        self.lstm = nn.LSTM(h, h, num_layers=hp["lstm_layers"], batch_first=True, bidirectional=True,
                            dropout=hp["dropout"] if hp["lstm_layers"] > 1 else 0.0)
        self.out = nn.Linear(2 * h, len(CLASSES))

    def forward(self, x):  # x: (batch, time, features)
        y, _ = self.lstm(torch.nn.functional.gelu(self.inp(x)))
        return self.out(y)


class TemporalTransformer(nn.Module):
    """A convolution-then-attention hybrid: two convolutions read local patterns (a touch, a
    control), then a transformer encoder relates moments up to a window apart. Attention over a
    whole half (13,500 steps) does not fit in memory, so a sequence longer than the window is read
    in windows overlapping by half, each keeping its central part."""

    def __init__(self, hp=None, local=True):
        """local=False: the pure transformer, without the two convolutions -- attention alone."""
        super().__init__()
        hp = dict(DEFAULTS, **(hp or {}))
        d = hp["tf_dim"]
        self.window = hp["tf_window"]
        self.inp = nn.Conv1d(n_features(), d, 1)
        self.local = (nn.Sequential(nn.Conv1d(d, d, 5, padding=2), nn.GELU(), nn.Conv1d(d, d, 5, padding=2), nn.GELU())
                      if local else nn.Identity())
        layer = nn.TransformerEncoderLayer(d, hp["tf_heads"], dim_feedforward=2 * d, dropout=hp["dropout"],
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, hp["tf_layers"], enable_nested_tensor=False)
        self.out = nn.Linear(d, len(CLASSES))

    def _positions(self, length, d, device):
        pos = torch.arange(length, device=device, dtype=torch.float32)[:, None]
        freq = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32) * (-np.log(10000.0) / d))
        pe = torch.zeros(length, d, device=device)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * freq), torch.cos(pos * freq)
        return pe

    def _forward(self, x):
        h = self.local(self.inp(x.transpose(1, 2))).transpose(1, 2)
        h = h + self._positions(h.shape[1], h.shape[2], h.device)
        return self.out(self.encoder(h))

    def forward(self, x):  # x: (batch, time, features)
        n, w = x.shape[1], self.window
        if n <= w:
            return self._forward(x)
        out = torch.zeros(x.shape[0], n, len(CLASSES), device=x.device)
        step = w // 2
        starts = list(range(0, n - w + 1, step))
        if starts[-1] != n - w:
            starts.append(n - w)
        for i, a in enumerate(starts):
            y = self._forward(x[:, a:a + w])
            lo = 0 if i == 0 else w // 4           # keep each window's centre
            hi = w if i == len(starts) - 1 else w - w // 4
            out[:, a + lo:a + hi] = y[:, lo:hi]
        return out


def build_model(hp=None):
    hp = dict(DEFAULTS, **(hp or {}))
    if hp["arch"] == "lstm":
        return BiLSTM(hp)
    if hp["arch"] in ("transformer", "transformer_pure"):
        return TemporalTransformer(hp, local=hp["arch"] == "transformer")
    return TCN(hp)


def halves(feats, labels, centres, period):
    """Split a game into per-half (features, label indices) sequences; -1 = not graded."""
    out = []
    step_period = period[centres]
    lab = labels[centres]
    idx = np.full(len(lab), -1)
    for k, c in enumerate(CLASSES):
        idx[lab == c] = k
    for half in (1, 2):
        m = step_period == half
        if m.sum() > 100:
            out.append((feats[m], idx[m]))
    return out


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def _on_device(sequences, device):
    """All training halves padded into one (n, T, F) tensor and one (n, T) label tensor on the
    device, so batches are cut there instead of being built on the CPU and copied every step."""
    longest = max(len(x) for x, _ in sequences)
    xs = np.zeros((len(sequences), longest, n_features()), dtype=np.float32)
    ys = np.full((len(sequences), longest), -1, dtype=np.int64)
    lengths = np.zeros(len(sequences), dtype=np.int64)
    for i, (x, y) in enumerate(sequences):
        xs[i, :len(x)], ys[i, :len(y)], lengths[i] = x, y, len(x)
    return torch.tensor(xs, device=device), torch.tensor(ys, device=device), lengths


def _batches(data, rng, hp, gen):
    """Random crops, CROPS_PER_HALF per half per epoch, half of them team-swapped, with the
    augmentations hp asks for (feature noise, the ball removed in blocks)."""
    xs, ys, lengths = data
    crop, batch = hp["crop"], hp["batch"]
    picks = np.repeat(np.arange(len(lengths)), CROPS_PER_HALF)
    rng.shuffle(picks)
    swap_index = torch.tensor(_swap_permutation(), device=xs.device)
    for i in range(0, len(picks), batch):
        rows = picks[i:i + batch]
        starts = np.array([rng.integers(0, max(1, lengths[r] - crop)) for r in rows])
        offsets = torch.arange(crop, device=xs.device)
        index = torch.tensor(starts, device=xs.device)[:, None] + offsets[None]
        index = index.clamp(max=xs.shape[1] - 1)
        row_t = torch.tensor(rows, device=xs.device)[:, None]
        x, y = xs[row_t, index], ys[row_t, index]
        swap = torch.tensor(rng.random(len(rows)) < 0.5, device=xs.device)
        x = torch.where(swap[:, None, None], x[..., swap_index] * _swap_signs(xs.device), x)
        y = torch.where(swap[:, None] & (y >= 0) & (y < 2), 1 - y, y)
        if hp["noise"] > 0:
            x = x + hp["noise"] * torch.randn(x.shape, device=x.device, generator=gen)
        if hp["ball_drop"] > 0:
            x = _drop_ball(x, rng, hp["ball_drop"])
        yield x, y


def _drop_ball(x, rng, share):
    """Remove the ball from `share` of the crops, in 1-3 blocks of 2-15 s, writing what the
    features say when the tracker has no ball: not seen, nobody near, no distances."""
    nb, nt = len(BALL_FEATURES), len(TEAM_FEATURES)
    x = x.clone()
    for r in np.flatnonzero(rng.random(x.shape[0]) < share):
        for _ in range(rng.integers(1, 4)):
            length = int(rng.integers(2 * FPS // STEP, 15 * FPS // STEP + 1))
            a = int(rng.integers(0, max(1, x.shape[1] - length)))
            x[r, a:a + length, :nb] = 0
            for base in (nb, nb + nt):
                x[r, a:a + length, base:base + 3] = 2.0   # d1..d3 when there is no ball (clipped 20 m / 10)
                x[r, a:a + length, base + 3:base + 5] = 0  # n2, n5
                if "touch" in TEAM_FEATURES:
                    x[r, a:a + length, base + nt - 1] = 0
    return x


def _weights(y, trans_weight):
    """Per-step loss weight: trans_weight within 1 s of a label change, else 1."""
    w = torch.ones(y.shape, device=y.device)
    if trans_weight == 1:
        return w
    change = torch.zeros(y.shape, dtype=torch.bool, device=y.device)
    change[:, 1:] = (y[:, 1:] != y[:, :-1]) & (y[:, 1:] >= 0) & (y[:, :-1] >= 0)
    half = FPS // STEP
    near = torch.nn.functional.max_pool1d(change.float()[:, None], 2 * half + 1, stride=1, padding=half)[:, 0] > 0
    w[near] = trans_weight
    return w


def _swap_permutation():
    """Feature index map that exchanges the two team blocks (see swap_teams)."""
    nb, nt = len(BALL_FEATURES), len(TEAM_FEATURES)
    return list(range(nb)) + list(range(nb + nt, nb + 2 * nt)) + list(range(nb, nb + nt))


def _swap_signs(device):
    signs = torch.ones(n_features(), device=device)
    signs[1] = -1.0  # ball x in the other team's direction
    return signs


def _loss(model, sequences, device):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for x, y in sequences:
            logits = model(torch.tensor(x)[None].to(device))[0]
            target = torch.tensor(y).to(device)
            keep = target >= 0
            total += nn.functional.cross_entropy(logits[keep], target[keep], reduction="sum").item()
            count += int(keep.sum())
    return total / max(count, 1)


def train(train_seqs, val_seqs, log=print, hp=None, provenance=None):
    """Fit a TCN on per-half sequences, stopping when the validation loss stops improving."""
    hp = dict(DEFAULTS, **(hp or {}))
    _seed()
    device = _device()
    model = build_model(hp).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, COSINE_EPOCHS) if hp["schedule"] == "cosine" else None
    rng = np.random.default_rng(SEED)
    gen = torch.Generator(device=device).manual_seed(SEED)
    data = _on_device(train_seqs, device)
    best, best_state, waited = np.inf, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for x, y in _batches(data, rng, hp, gen):
            logits = model(x)
            per_step = nn.functional.cross_entropy(logits.reshape(-1, len(CLASSES)), y.reshape(-1).clamp(min=0),
                                                   reduction="none")
            valid = (y.reshape(-1) >= 0).float()
            w = _weights(y, hp["trans_weight"]).reshape(-1) * valid
            loss = (per_step * w).sum() / w.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if sched is not None and epoch < COSINE_EPOCHS:
            sched.step()
        if epoch % VAL_EVERY:
            continue
        val = _loss(model, val_seqs, device)
        if val < best - 1e-4:
            best, waited = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += VAL_EVERY
        if epoch % 20 == 0 or waited == 0:
            log("    epoch {:3d}  validation loss {:.4f}{}".format(epoch, val, "  *" if waited == 0 else ""))
        if waited >= PATIENCE:
            log("    stopped at epoch {} (best validation loss {:.4f})".format(epoch, best))
            break
    model.load_state_dict(best_state)
    model.eval()
    model.experiment_metadata = {"hp": hp, "seed": SEED, "feature_schema": feature_schema(),
                                 "data": provenance or {}, "source": records.source_snapshot(),
                                 "runtime": records.runtime()}
    return model


def step_probabilities(model, frames, feats=None):
    """(steps, 3) P(H), P(A), P(D) at 5 fps, the 25 fps frame each step stands for, and its half."""
    feats, centres = features(frames) if feats is None else feats
    device = _device()
    step_p = np.zeros((len(feats), 3))
    with torch.no_grad():
        for half in (1, 2):
            m = frames["period"][centres] == half
            if m.any():
                logits = model(torch.tensor(feats[m])[None].to(device))[0]
                step_p[m] = torch.softmax(logits, -1).cpu().numpy()
    return step_p, centres, frames["period"][centres]


def probabilities(model, frames, feats=None):
    """(n, 3) P(H), P(A), P(D) at 25 fps; 0 outside play."""
    feats, centres = features(frames) if feats is None else feats
    device = _device()
    n = len(frames["period"])
    out = np.zeros((n, 3))
    step_p = np.zeros((len(feats), 3))
    with torch.no_grad():
        for half in (1, 2):
            m = frames["period"][centres] == half
            if m.any():
                logits = model(torch.tensor(feats[m])[None].to(device))[0]
                step_p[m] = torch.softmax(logits, -1).cpu().numpy()
    out = periods.expand(step_p, frames["period"], STEP)
    out[frames["period"] == 0] = 0
    return out


MODEL_DIR = os.path.join(os.path.dirname(HERE), "external", "pff_wc2022", "models")


def _viterbi_varying(score, cost):
    """Viterbi over additive state scores where leaving a state at step t costs cost[t]."""
    n = len(score)
    back = np.zeros((n, 3), dtype=np.int8)
    total = score[0].copy()
    for t in range(1, n):
        best = int(np.argmax(total))
        switch = total[best] - cost[t]
        stay = total >= switch
        back[t] = np.where(stay, np.arange(3), best)
        total = np.where(stay, total, switch) + score[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = np.argmax(total)
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def save_model(model, path):
    """New checkpoints carry their architecture, feature schema, split and input identities."""
    meta = getattr(model, "experiment_metadata", None)
    if (not meta or not meta["data"].get("training") or not meta["data"].get("validation")
            or not meta["data"].get("inputs")):
        raise ValueError("Checkpoint requires explicit training, validation and input-hash provenance")
    if set(meta["data"]["training"]) & set(meta["data"]["validation"]):
        raise ValueError("Training and validation games overlap")
    import io
    buffer = io.BytesIO()
    torch.save({"format_version": 1, "metadata": meta, "state_dict": model.state_dict()}, buffer)
    records.atomic_bytes(path, buffer.getvalue())
    records.write_json(str(path) + ".json", dict(meta, weights_sha256=records.file_hash(path)))


def load_models(directory=MODEL_DIR, hp=None, prefix="tcn", allow_legacy=False):
    """Use recorded configuration. Bare historical weights require explicit opt-in and full hp."""
    models = []
    for name in sorted(f for f in os.listdir(directory) if f.startswith(prefix + "_fold") and f.endswith(".pt")):
        path = os.path.join(directory, name)
        payload = torch.load(path, map_location=_device(), weights_only=True)
        if isinstance(payload, dict) and payload.get("format_version") == 1:
            meta = payload["metadata"]
            if meta["feature_schema"] != feature_schema():
                raise ValueError("Checkpoint feature schema/source mismatch: " + name)
            config = meta["hp"]
            if hp is not None and hp != config:
                raise ValueError("Explicit hyperparameters disagree with checkpoint: " + name)
            state = payload["state_dict"]
        else:
            if not allow_legacy or hp is None or set(hp) != set(DEFAULTS):
                raise ValueError("Legacy weights need allow_legacy=True and a full frozen hp configuration: " + name)
            config, state = hp, payload
        model = build_model(config).to(_device())
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    if not models:
        raise ValueError("No checkpoints found for " + prefix)
    return models


def ensemble_method(models, broadcast_ball=False, lam=LAMBDA):
    """A method averaging the step probabilities of several networks, then Viterbi at 5 fps."""
    def run(frames):
        if broadcast_ball:
            frames = data.with_broadcast_ball(frames)
        feats = features(frames)
        step_p = np.mean([step_probabilities(m, frames, feats)[0] for m in models], axis=0)
        return decode_steps(frames, step_p, frames["period"][feats[1]], lam)
    run.__name__ = "tcn_sync" if broadcast_ball else "tcn"
    return run


def decode(frames, probs, lam=LAMBDA):
    return vote.combine_soft([probs], frames["period"], lam)


def decode_steps(frames, step_p, step_period, lam=LAMBDA, score_mode="probability", epsilon=1e-8,
                 beta=0.0, window_s=0.6):
    """Viterbi at 5 fps (the network's own resolution, five times less work), then each step's
    label spread over its 5 frames. lam is per 25 fps frame, so it is the same knob as decode's.
    beta > 0 makes a change cheaper where the network itself changes its mind: the cost at a step is
    lam * exp(-beta * e), e the total-variation distance between the mean probabilities in the
    window_s after it and the window_s before it (0 = no change, 1 = a certain one); it decodes
    probabilities as scores, whatever score_mode says."""
    if beta == 0:
        steps = vote.decode_probabilities(step_p, step_period, lam / STEP, score_mode, epsilon)
    else:
        steps = np.full(len(step_p), "X", dtype="<U1")
        w = max(1, int(round(window_s * FPS / STEP)))
        for a, b in periods.slices(step_period):
            p = step_p[a:b]
            c = np.vstack([np.zeros((1, 3)), np.cumsum(p, axis=0)])
            i = np.arange(len(p))
            lo, hi = np.maximum(i - w, 0), np.minimum(i + w, len(p))
            before = (c[i] - c[lo]) / np.maximum(i - lo, 1)[:, None]
            after = (c[hi] - c[i]) / np.maximum(hi - i, 1)[:, None]
            change = 0.5 * np.abs(after - before).sum(1)
            steps[a:b] = vote.STATES[_viterbi_varying(p, lam / STEP * np.exp(-beta * change))]
    n = len(frames["period"])
    expected_period = frames["period"][periods.blocks(frames["period"], STEP)[2]]
    if not np.array_equal(step_period, expected_period):
        raise ValueError("Step periods do not match the period-safe sampling grid")
    out = periods.expand(steps, frames["period"], STEP)
    out[frames["period"] == 0] = "X"
    return out
