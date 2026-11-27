"""Continuous-time Markov model of core damage on the lattice.

Every cell is in one of five states

    0 intact  ->  1 clad failed  ->  2 damaged  ->  3 molten  ->  4 relocated

with transition hazards that depend on the local, deterministically evolving state (clad and fuel
temperature, oxidation, fuel enthalpy, the state of the cell below). Between deterministic steps the
hazards are frozen, so the jump process is simulated either exactly (Gillespie direct method, events
in continuous time within the step, propensities recomputed after each event) or by tau-leaping
(each cell jumps with probability 1 - exp(-lambda dt)). Relocation moves the cell's fuel, decay-heat
inventory and enthalpy into the first non-relocated cell below, or into the lower head.
"""
from __future__ import annotations

import numpy as np

INTACT, FAILED, DAMAGED, MOLTEN, RELOCATED = range(5)
STATE_NAMES = ["intact", "clad failed", "damaged", "molten", "relocated"]


def _sig(x):
    """Logistic switch, cut to zero far below the threshold (no spurious failures in normal operation)."""
    x = np.asarray(x, float)
    return np.where(x < -6.0, 0.0, 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60))))


def hazards(rt, s, Tc, Tf, ecr, H_cal, melt_frac, below_state, quench_shock, dH_cal=None):
    """Hazard (1/s) of the next transition of each cell, and its target state."""
    lam = np.zeros(len(s))
    tgt = np.minimum(s + 1, RELOCATED)
    Tmax = np.maximum(Tc, Tf)
    # 0 -> 1: burst / creep rupture, PCMI in a power excursion, embrittlement
    m = s == INTACT
    l0 = (rt.clad_fail_rate * _sig((Tc - rt.T_clad_fail) / rt.T_clad_fail_w)
          + 2.0 * _sig(((H_cal if dH_cal is None else dH_cal) - rt.H_pcmi_fail) / 8.0)
          + 0.05 * _sig((ecr - rt.ECR_limit) / 0.02))
    lam[m] = l0[m]
    # 1 -> 2: loss of rod geometry: oxidation embrittlement, high temperature, fragmentation, quench
    m = s == FAILED
    l1 = (rt.damage_rate * (_sig((ecr - rt.ECR_limit) / 0.03) + _sig((Tc - rt.T_damage - 150.0) / 60.0))
          + 2.0 * _sig((H_cal - rt.H_fragment) / 8.0)
          + 0.2 * quench_shock * _sig((ecr - rt.ECR_limit) / 0.03))
    lam[m] = l1[m]
    # 2 -> 3: liquefaction / melting
    m = s == DAMAGED
    l2 = rt.melt_rate * _sig((Tmax - rt.T_liquefy) / rt.T_liquefy_w) + 0.5 * (melt_frac > 0.5)
    lam[m] = l2[m]
    # 2 -> 4: loss of support when the cell below has gone
    sup = m & (below_state == RELOCATED)
    lam[sup] += 1.0 / 600.0
    tgt[sup & (l2 < 1.0 / 600.0)] = RELOCATED
    # 3 -> 4: slumping of molten material; 3 -> 2: refreezing into a crust
    m = s == MOLTEN
    path = np.where(below_state < 0, 1.0, np.where(below_state >= MOLTEN, 3.0, 1.0))
    l3 = path / rt.reloc_tau
    lam[m] = l3[m]
    frz = m & (Tmax < rt.T_liquefy - 350.0)
    lam[frz] = 0.01
    tgt[frz] = DAMAGED
    lam[s == RELOCATED] = 0.0
    return lam, tgt


def tau_leap(lam, tgt, dt, rng):
    p = -np.expm1(-lam * dt)
    hit = rng.random(len(lam)) < p
    idx = np.nonzero(hit)[0]
    # bottom-up so relocations cascade in a physical order
    return [(int(i), int(tgt[i]), dt) for i in idx]


def gillespie(lam_fn, lam, tgt, dt, rng, max_events=10000):
    """Direct-method SSA over [0, dt] with propensities frozen between events.
    lam_fn(events_so_far) -> (lam, tgt) recomputes propensities after each event."""
    t = 0.0
    out = []
    while len(out) < max_events:
        R = lam.sum()
        if R <= 0:
            break
        t += rng.exponential(1.0 / R)
        if t > dt:
            break
        c = np.cumsum(lam)
        i = int(np.searchsorted(c, rng.random() * R, side="right"))
        i = min(i, len(lam) - 1)
        out.append((i, int(tgt[i]), t))
        lam, tgt = lam_fn(out)
    return out
