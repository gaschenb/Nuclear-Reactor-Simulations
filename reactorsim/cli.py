"""Command-line interface.

    reactorsim gui [--dark]
    reactorsim reactors
    reactorsim scenarios [--reactor pwr]
    reactorsim run pwr-sbo [--param t_afw=3600] [--seed 3] [--method tau-leap] [--csv out.csv] [--png out.png]
    reactorsim ensemble pwr-sbo -n 32 [--uncertain 1.0] [--workers 4] [--csv members.csv]
    reactorsim xenon --reactor pwr [--power 1.0] [--after 0.0] [--hours 72]
    reactorsim kinetics --reactor pwr --rho 0.2 0.5 -0.5 1.2
    reactorsim decay --times 1 60 3600 86400 [--days 365]
    reactorsim boiloff --reactor bwr
Times are in seconds unless stated, reactivities in dollars.
"""
from __future__ import annotations

import argparse
import csv
import sys

import numpy as np


def _table(rows, header=None):
    rows = [tuple(str(c) for c in r) for r in rows]
    if header:
        rows = [tuple(header)] + rows
    w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for j, r in enumerate(rows):
        print("  " + "  ".join(c.ljust(w[i]) for i, c in enumerate(r)).rstrip())
        if header and j == 0:
            print("  " + "  ".join("-" * x for x in w))


def _hms(t):
    if t is None or not np.isfinite(t):
        return "—"
    if t < 120:
        return f"{t:.1f} s"
    if t < 2 * 3600:
        return f"{t / 60:.1f} min"
    return f"{t / 3600:.2f} h"


def _rt(key):
    from .core.reactors import REACTORS
    if key not in REACTORS:
        raise SystemExit(f"unknown reactor {key!r}; one of: {', '.join(REACTORS)}")
    return REACTORS[key]


def cmd_reactors(a):
    from .core.analytics import reactor_table
    rows = []
    for r in reactor_table():
        rows.append((r["key"], f"{r['power_MW']:.0f}", r["coolant"], r["moderator"] or "—", r["fuel"],
                     f"{r['beta_eff'] * 1e5:.0f}", f"{r['Lambda']:.1e}", f"{r['alpha_D'] * 1e5:+.2f}",
                     f"{r['alpha_void'] * 1e2:+.2f}", f"{r['xe_eq'] * 1e2:+.2f}"))
    _table(rows, ("key", "MWt", "coolant", "moderator", "fuel", "beta (pcm)", "Lambda (s)",
                  "Doppler (pcm/K)", "void (%dk/k)", "Xe eq (%dk/k)"))


def cmd_scenarios(a):
    from .core.scenarios import SCENARIOS
    rows = [(k, s.reactor, s.name) for k, s in SCENARIOS.items() if a.reactor in (None, s.reactor)]
    _table(rows, ("key", "reactor", "description"))


def _params(items):
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        out[k] = float(v)
    return out


def _check_scenario(key):
    from .core.scenarios import SCENARIOS
    if key not in SCENARIOS:
        raise SystemExit(f"unknown scenario {key!r}; see `reactorsim scenarios`")
    return SCENARIOS[key]


def cmd_run(a):
    from .core.simulator import Simulator
    sc = _check_scenario(a.scenario)
    print(f"{sc.name}  ({sc.reactor})")
    last = [0.0]

    def prog(f):
        if f - last[0] >= 0.1:
            last[0] = f
            print(f"  … {f * 100:.0f} %", file=sys.stderr)

    sim = Simulator(sc, params=_params(a.param), method=a.method, seed=a.seed)
    r = sim.run(progress=None if a.quiet else prog)
    s = r.summary
    print("\nEvents")
    for t, msg in r.events:
        print(f"  {_hms(t):>10}  {msg}")
    print("\nSummary")
    _table([("peak power (rated)", f"{s['peak_power']:.3g}"),
            ("peak fuel temperature", f"{s['peak_fuel_T']:.0f} K"),
            ("peak cladding temperature", f"{s['peak_clad_T']:.0f} K"),
            ("peak fuel enthalpy", f"{s['peak_enthalpy']:.0f} cal/g"),
            ("hydrogen", f"{s['H2_kg']:.1f} kg"),
            ("cells failed or worse", f"{s['failed_or_worse'] * 100:.1f} %"),
            ("cells molten (peak)", f"{s['molten_max'] * 100:.1f} %"),
            ("cells relocated", f"{s['relocated'] * 100:.1f} %"),
            ("fission products released", f"{s['release']:.2f} % of inventory"),
            ("vessel failed", str(s["vessel_failed"])),
            ("run time", f"{r.runtime:.1f} s, {s['steps']} steps")])
    if a.csv:
        keys = list(r.series)
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t"] + keys)
            for i in range(len(r.t)):
                w.writerow([f"{r.t[i]:.6g}"] + [f"{r.series[k][i]:.6g}" for k in keys])
        print(f"wrote {a.csv}")
    if a.png:
        from . import plots
        plots.scenario_figure(r, a.png)
        print(f"wrote {a.png}")


def cmd_ensemble(a):
    from .core.ensemble import run_ensemble
    sc = _check_scenario(a.scenario)
    print(f"{sc.name}: {a.n} members, uncertainty scale {a.uncertain}")
    res = run_ensemble(a.scenario, n=a.n, params=_params(a.param), uncertain=a.uncertain,
                       seed0=a.seed, workers=a.workers,
                       progress=lambda i, n: print(f"  member {i}/{n}", file=sys.stderr))
    _table(res.table(), ("outcome", "probability [95 % CI]", "median time", "5–95 %"))
    h2 = res.metric("H2_kg")
    print(f"\n  hydrogen: median {np.median(h2):.1f} kg, 5–95 % {np.percentile(h2, 5):.1f}–{np.percentile(h2, 95):.1f} kg")
    if a.csv:
        from .core.ensemble import EVENTS
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["seed", "peak_clad_T", "H2_kg", "molten_max", "relocated", "release"] + list(EVENTS))
            for m in res.members:
                w.writerow([m["seed"], f"{m['peak_clad_T']:.1f}", f"{m['H2_kg']:.2f}", f"{m['molten_max']:.4f}",
                            f"{m['relocated']:.4f}", f"{m['release']:.4f}"]
                           + [f"{m['times'][e]:.1f}" if e in m["times"] else "" for e in EVENTS])
        print(f"wrote {a.csv}")


def cmd_xenon(a):
    from .core import analytics as an
    rt = _rt(a.reactor)
    t = np.linspace(0, a.hours * 3600, 2000)
    r = an.poison_transient(rt, t, a.power, a.after)
    i = int(np.argmin(r["xenon"]))
    print(f"{rt.name}")
    rows = [("equilibrium Xe worth at power", f"{an.xenon_equilibrium_worth(rt, a.power) * 1e2:+.3f} %dk/k"),
            ("minimum Xe worth (peak poisoning)", f"{r['xenon'][i] * 1e2:+.3f} %dk/k at {_hms(t[i])}"),
            (f"Xe worth after {a.hours:g} h", f"{r['xenon'][-1] * 1e2:+.3f} %dk/k"),
            (f"Sm worth after {a.hours:g} h", f"{r['samarium'][-1] * 1e2:+.3f} %dk/k")]
    if a.after == 0.0:
        tp, w, t0, t1 = an.restart_window(rt, p_before=a.power)
        rows.append(("excess reactivity", f"{rt.excess * 1e2:.2f} %dk/k"))
        rows.append(("restart impossible (xenon dead time)",
                     f"{_hms(t0)} to {_hms(t1)}" if t0 is not None else "never"))
    _table(rows)


def cmd_kinetics(a):
    from .core import analytics as an
    rt = _rt(a.reactor)
    rho, T = an.inhour_curve(rt, np.array(a.rho))
    rows = []
    for r_, T_ in zip(rho, T):
        pj = an.prompt_jump(r_)
        nf = an.nordheim_fuchs(rt, r_) if r_ > 1 else None
        extra = f"burst: peak {nf['peak_rel']:.3g}× rated, FWHM {nf['fwhm_s'] * 1e3:.3g} ms, ΔT {nf['dT_fuel']:.0f} K" if nf else \
            (f"prompt jump ×{float(pj):.3g}" if r_ < 1 else "")
        rows.append((f"{r_:+.3g} $", f"{T_:.4g} s", extra))
    print(rt.name)
    _table(rows, ("reactivity", "stable period", ""))


def cmd_decay(a):
    from .core.nuclear_data import decay_heat_fraction, way_wigner
    t = np.array(a.times, float)
    P = decay_heat_fraction(t, a.days * 86400.0)
    _table([(_hms(ti), f"{p * 100:.3f} %", f"{way_wigner(np.array(ti)) * 100:.3f} %") for ti, p in zip(t, P)],
           ("after shutdown", f"ANS-5.1 ({a.days:g} d)", "Way–Wigner"))


def cmd_boiloff(a):
    from .core import analytics as an
    rt = _rt(a.reactor)
    r = an.boil_off(rt, a.after)
    if r is None:
        raise SystemExit(f"{rt.key}: not a water-cooled design")
    _table([("core uncovery (top of fuel)", _hms(r["t_uncover"])), ("core dry (bottom of fuel)", _hms(r["t_dry"])),
            ("adiabatic heat-up 1 h after trip", f"{an.adiabatic_heatup_rate(rt, 3600):.3f} K/s")])


def cmd_gui(a):
    from .gui.app import main
    return main(["reactorsim"] + (["--dark"] if a.dark else []))


def build_parser():
    p = argparse.ArgumentParser(prog="reactorsim", description="Nuclear reactor transient and severe-accident simulations")
    sp = p.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gui", help="desktop application"); g.add_argument("--dark", action="store_true")
    g.set_defaults(fn=cmd_gui)
    sp.add_parser("reactors", help="built-in reactor designs").set_defaults(fn=cmd_reactors)
    s = sp.add_parser("scenarios", help="built-in scenarios"); s.add_argument("--reactor")
    s.set_defaults(fn=cmd_scenarios)
    r = sp.add_parser("run", help="run one scenario")
    r.add_argument("scenario"); r.add_argument("--param", "-p", action="append", metavar="NAME=VALUE")
    r.add_argument("--seed", type=int, default=0); r.add_argument("--method", choices=("gillespie", "tau-leap"),
                                                                   default="gillespie")
    r.add_argument("--csv"); r.add_argument("--png"); r.add_argument("--quiet", "-q", action="store_true")
    r.set_defaults(fn=cmd_run)
    e = sp.add_parser("ensemble", help="Monte Carlo ensemble of one scenario")
    e.add_argument("scenario"); e.add_argument("-n", type=int, default=16)
    e.add_argument("--uncertain", type=float, default=1.0, help="scale of the parameter spreads (0 = chain only)")
    e.add_argument("--param", "-p", action="append", metavar="NAME=VALUE")
    e.add_argument("--seed", type=int, default=0); e.add_argument("--workers", type=int)
    e.add_argument("--csv"); e.set_defaults(fn=cmd_ensemble)
    x = sp.add_parser("xenon", help="Xe-135 / Sm-149 transient after a power change")
    x.add_argument("--reactor", default="pwr"); x.add_argument("--power", type=float, default=1.0)
    x.add_argument("--after", type=float, default=0.0); x.add_argument("--hours", type=float, default=72.0)
    x.set_defaults(fn=cmd_xenon)
    k = sp.add_parser("kinetics", help="inhour periods, prompt jump, Nordheim–Fuchs bursts")
    k.add_argument("--reactor", default="pwr"); k.add_argument("--rho", type=float, nargs="+", required=True)
    k.set_defaults(fn=cmd_kinetics)
    d = sp.add_parser("decay", help="ANS-5.1 decay heat")
    d.add_argument("--times", type=float, nargs="+", default=[1, 10, 60, 600, 3600, 36000, 86400, 864000])
    d.add_argument("--days", type=float, default=365.0); d.set_defaults(fn=cmd_decay)
    b = sp.add_parser("boiloff", help="time to uncover a water-cooled core with no injection")
    b.add_argument("--reactor", default="pwr"); b.add_argument("--after", type=float, default=60.0)
    b.set_defaults(fn=cmd_boiloff)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    raise SystemExit(main())
