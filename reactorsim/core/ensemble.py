"""Monte Carlo ensembles of the stochastic lattice model.

Each member is an independent realisation of the continuous-time Markov chain of cell damage
(its own random stream) and, optionally, of uncertain damage-progression parameters sampled
from log-normal / normal distributions. Members run in parallel processes.

The outcome statistics are estimated from the members: probabilities of each end state with
Wilson 95 % intervals, and empirical distributions of the times of the first failure, first
melt, relocation and lower-head failure.
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np

# uncertain inputs: name -> (kind, spread). "log": multiplicative log-normal with sigma = spread;
# "abs": additive normal with sigma = spread (K)
UNCERTAIN = {
    "clad_fail_rate": ("log", 0.5),
    "damage_rate": ("log", 0.5),
    "melt_rate": ("log", 0.5),
    "reloc_tau": ("log", 0.4),
    "T_clad_fail": ("abs", 25.0),
    "T_damage": ("abs", 40.0),
    "T_liquefy": ("abs", 40.0),
}

EVENTS = ("first cladding failure", "first loss", "first molten", "first relocation",
          "lower-head failure")
EVENT_LABELS = {"first cladding failure": "clad failure", "first loss": "loss of rod geometry",
                "first molten": "first melt", "first relocation": "relocation",
                "lower-head failure": "vessel breach"}


def sample_overrides(rt, rng, scale=1.0):
    out = {}
    for k, (kind, s) in UNCERTAIN.items():
        v = getattr(rt, k)
        if kind == "log":
            out[k] = float(v * math.exp(rng.normal(0.0, s * scale)))
        else:
            out[k] = float(v + rng.normal(0.0, s * scale))
    return out


SERIES = ("frac_intact", "H2", "T_clad_max", "T_fuel_max", "P_total", "p", "release", "frac_molten",
          "frac_relocated")


def _member(args):
    key, params, seed, uncertain, nx, nz = args
    from .simulator import Simulator, SCENARIOS
    from .reactors import REACTORS
    sc = SCENARIOS[key] if isinstance(key, str) else key
    ov = None
    if uncertain > 0:
        rt = REACTORS[sc.reactor]
        ov = sample_overrides(rt, np.random.default_rng(10_000 + seed), uncertain)
    sim = Simulator(sc, params=params, seed=seed, nx=nx, nz=nz, overrides=ov)
    r = sim.run()
    s = r.summary
    step = max(1, len(r.t) // 400)
    fuel = r.snapshots[-1].fields
    Tf_peak = np.nanmax(np.stack([sn.fields["T_fuel"] for sn in r.snapshots]), axis=0)
    out = dict(seed=seed, overrides=ov or {}, times=dict(s["times"]),
               peak_power=s["peak_power"], peak_fuel_T=s["peak_fuel_T"], peak_clad_T=s["peak_clad_T"],
               H2_kg=s["H2_kg"], molten_max=s["molten_max"], relocated=s["relocated"],
               failed=s["failed_or_worse"], damaged=s["damaged_or_worse"], release=s["release"],
               vessel_failed=s["vessel_failed"], t=r.t[::step], runtime=r.runtime,
               nx=r.nx, nz=r.nz, fuel_mask=r.fuel_mask, cell_times=r.cell_times.astype(np.float32),
               state_end=fuel["state"].astype(np.float32), Tf_peak=Tf_peak.astype(np.float32))
    out["series"] = {k: np.asarray(r.series[k])[::step] for k in SERIES if k in r.series}
    out["frac_intact"], out["H2"] = out["series"]["frac_intact"], out["series"]["H2"]
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(c - h, 0.0), min(c + h, 1.0)


@dataclass
class EnsembleResult:
    scenario: str
    members: list = field(default_factory=list)

    @property
    def n(self):
        return len(self.members)

    def probability(self, what):
        """P(outcome) with a Wilson 95 % interval. what: 'clad failure', 'melt', 'relocation',
        'vessel breach' or any key of EVENTS."""
        alias = {"clad failure": "first cladding failure", "damage": "first loss",
                 "melt": "first molten", "relocation": "first relocation",
                 "vessel breach": "lower-head failure"}
        key = alias.get(what, what)
        k = sum(1 for m in self.members if key in m["times"])
        return wilson(k, self.n)

    def times(self, what):
        alias = {"clad failure": "first cladding failure", "damage": "first loss",
                 "melt": "first molten", "relocation": "first relocation",
                 "vessel breach": "lower-head failure"}
        key = alias.get(what, what)
        return np.sort([m["times"][key] for m in self.members if key in m["times"]])

    def table(self):
        rows = []
        for ev in EVENTS:
            p, lo, hi = self.probability(ev)
            t = self.times(ev)
            med = f"{np.median(t) / 3600:.2f} h" if len(t) else "—"
            spread = f"{np.percentile(t, 5) / 3600:.2f}–{np.percentile(t, 95) / 3600:.2f} h" if len(t) > 2 else "—"
            rows.append((EVENT_LABELS[ev], f"{p:.2f} [{lo:.2f}, {hi:.2f}]", med, spread))
        return rows

    def metric(self, name):
        return np.array([m[name] for m in self.members], float)


def run_ensemble(key, n=16, params=None, uncertain=1.0, seed0=0, workers=None, nx=None, nz=None,
                 progress=None, cancelled=None, on_member=None):
    """Run n members of scenario `key` (a key or a Scenario). uncertain scales the parameter
    spreads (0 = only the stochastic damage chain varies). progress(i, n) and on_member(member)
    after each member."""
    args = [(key, params, seed0 + i, uncertain, nx, nz) for i in range(n)]
    res = EnsembleResult(key if isinstance(key, str) else key.key)
    workers = workers or max(1, min(n, (os.cpu_count() or 2) - 1))
    if workers == 1:
        for i, a in enumerate(args):
            if cancelled is not None and cancelled():
                break
            res.members.append(_member(a))
            if on_member:
                on_member(res.members[-1])
            if progress:
                progress(i + 1, n)
        return res
    ex = ProcessPoolExecutor(max_workers=workers)
    try:
        futs = [ex.submit(_member, a) for a in args]
        for i, f in enumerate(as_completed(futs)):
            if cancelled is not None and cancelled():
                break
            res.members.append(f.result())
            if on_member:
                on_member(res.members[-1])
            if progress:
                progress(i + 1, n)
    finally:
        # on cancel (or an exception from progress) drop the queued members without waiting
        ex.shutdown(wait=len(res.members) == n, cancel_futures=True)
    res.members.sort(key=lambda m: m["seed"])
    return res
