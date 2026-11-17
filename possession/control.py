"""The control rule: possession changes hands only after a player has sustained control of the ball.

Shared by methods.control (the rule on its own) and hmm.PossessionHMM (the rule as evidence).
"""

import numpy as np

HOME, AWAY, NOBODY = 0, 1, -1

# Control rule parameters, chosen by tune_control.py on the realistic external games only.
CONTROL_R = 2.0     # m: a player this close to the ball can be controlling it
CONTROL_T = 0.8     # s: ...for this long before possession changes hands
CONTROL_V = np.inf  # m/s: ...with the ball no faster than this (inf: no speed condition)
SPEED_SPAN = 5      # frames over which ball speed is measured, so tracking noise does not dominate


def ball_speed(ball):
    """m/s over SPEED_SPAN frames; NaN where either end is missing."""
    v = np.full(len(ball), np.nan)
    v[SPEED_SPAN:] = np.linalg.norm(ball[SPEED_SPAN:] - ball[:-SPEED_SPAN], axis=1) * 25 / SPEED_SPAN
    return v


def control_codes(period, dead, d_home, d_away, speed, r, t, v):
    """Possession by sustained control. A team is a candidate on a frame when its nearest player is
    within r m of the ball, nearer than the other team's, and the ball is slower than v m/s. It
    takes possession only after t s of candidacy uninterrupted by the other team (frames with no
    candidate do not break it), backdated to where that run began. Dead frames are dead; after a
    stoppage, the first team to take control has the ball from the end of the stoppage.
    """
    n = len(period)
    with np.errstate(invalid="ignore"):
        h = (d_home < r) & ~(d_away < d_home) & ~(speed > v)
        a = (d_away < r) & ~(d_home < d_away) & ~(speed > v)
    candidate = np.where(h, HOME, np.where(a, AWAY, NOBODY))
    need = max(1, int(round(t * 25)))
    out = np.full(n, NOBODY, dtype=np.int8)
    state, run_team, run_start, run_len, open_from = NOBODY, NOBODY, 0, 0, 0
    for i in range(n):
        if i and period[i] != period[i - 1]:
            state, run_team, run_len, open_from = NOBODY, NOBODY, 0, i
        if period[i] == 0 or dead[i]:
            state, run_team, run_len, open_from = NOBODY, NOBODY, 0, i + 1
            continue
        c = candidate[i]
        if c != NOBODY and c != state:
            if c != run_team:
                run_team, run_start, run_len = c, i, 0
            run_len += 1
            if run_len >= need:
                begin = open_from if state == NOBODY else run_start
                out[begin:i] = c
                state, run_team, run_len = c, NOBODY, 0
        elif c == state and c != NOBODY:
            run_team, run_len = NOBODY, 0
        out[i] = state
    return out


