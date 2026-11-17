"""The touch detector's probabilities as possession-network inputs, for any frames (the match included).

enable() appends the per-team "touch" feature to temporal.TEAM_FEATURES, which is what the networks of
bench_touch_input.py were trained with; call it before temporal.load_models on them. augment() puts the
detector's per-step maximum after each team's block. method() is temporal.ensemble_method with the
detector run on the frames first (the four fold detectors averaged, on the frames at 10 fps as they
were trained).
"""

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import frames as data  # noqa: E402
import periods  # noqa: E402
import temporal  # noqa: E402
import touch_detector as td  # noqa: E402

NB, NT = len(temporal.BALL_FEATURES), len(temporal.TEAM_FEATURES)  # the plain layout, before enable()


def enable():
    if "touch" not in temporal.TEAM_FEATURES:
        temporal.TEAM_FEATURES.append("touch")


def base_features(frames):
    """temporal.features with the plain 24 features, whatever enable() did."""
    added = temporal.TEAM_FEATURES[-1] == "touch" and len(temporal.TEAM_FEATURES) == NT + 1
    if added:
        temporal.TEAM_FEATURES.pop()
    try:
        return temporal.features(frames)
    finally:
        if added:
            temporal.TEAM_FEATURES.append("touch")


def augment(feats, touch_p, period):
    """(steps, 24) features and (frames, 3) detector probabilities -> (steps, 26)."""
    starts, stops, _ = periods.blocks(period, temporal.STEP)
    touch = np.stack([touch_p[a:b, 1:].max(0) for a, b in zip(starts, stops)]).astype(np.float32)
    return np.concatenate([feats[:, :NB + NT], touch[:, :1], feats[:, NB + NT:], touch[:, 1:]], axis=1)


def detector_probabilities(frames, detector_dir):
    x = td.touch_features(td.at_10fps(frames))
    out = []
    for k in range(4):
        model = td.Detector().to(temporal._device())
        model.load_state_dict(torch.load(os.path.join(detector_dir, "fold{}.pt".format(k)), map_location=temporal._device(),
                                         weights_only=True))
        model.eval()
        out.append(td.predict(model, x))
    return np.mean(out, axis=0)


def method(models, detector_dir, lam, broadcast_ball=True):
    def run(frames):
        if broadcast_ball:
            frames = data.with_broadcast_ball(frames)
        feats, centres = base_features(frames)
        x = augment(feats, detector_probabilities(frames, detector_dir), frames["period"])
        step_p = np.mean([temporal.step_probabilities(m, frames, (x, centres))[0] for m in models], axis=0)
        return temporal.decode_steps(frames, step_p, frames["period"][centres], lam)
    return run
