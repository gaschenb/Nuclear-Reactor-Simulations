"""Matplotlib figures of scenario results, ensembles and the analytic models (CLI --png and the
README gallery)."""
from __future__ import annotations

import numpy as np

STATE_COLORS = ["#4e79a7", "#edc948", "#f28e2b", "#e15759", "#7f3b08"]
STATE_NAMES = ["intact", "failed", "damaged", "molten", "relocated"]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _time_axis(t):
    te = float(t[-1]) if len(t) else 1.0
    if te > 3 * 3600:
        return np.asarray(t) / 3600, "time (h)"
    if te > 300:
        return np.asarray(t) / 60, "time (min)"
    return np.asarray(t), "time (s)"


def _state_map(ax, res, snap, title):
    from matplotlib.colors import ListedColormap, BoundaryNorm
    st = snap.fields["state"].reshape(res.nz, res.nx)
    cm = ListedColormap(STATE_COLORS)
    ax.imshow(np.ma.masked_invalid(st), origin="lower", cmap=cm, norm=BoundaryNorm(np.arange(-0.5, 5.5), 5),
              aspect="auto")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])


def scenario_figure(res, path, dpi=130):
    """Power and reactivity, temperatures, coolant, damage fractions, plus the lattice damage map
    at the end of the run."""
    plt = _plt()
    S = res.series
    t, tl = _time_axis(res.t)
    fig = plt.figure(figsize=(11, 6.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.75])
    ax = fig.add_subplot(gs[0, 0])
    ax.semilogy(t, np.maximum(S["P_total"], 1e-3), label="total")
    ax.semilogy(t, np.maximum(S["P_decay"], 1e-3), "--", label="decay heat")
    ax.set_ylabel("power (MW)"); ax.set_xlabel(tl); ax.legend(fontsize=8)
    ax2 = ax.twinx(); ax2.plot(t, S["rho"], color="#8c8c8c", lw=0.8); ax2.set_ylabel("reactivity ($)")
    ax2.grid(False)
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, S["T_fuel_max"], label="fuel, max"); ax.plot(t, S["T_clad_max"], label="clad, max")
    ax.plot(t, S["T_cool_max"], label="coolant, max")
    ax.set_ylabel("temperature (K)"); ax.set_xlabel(tl); ax.legend(fontsize=8)
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t, S["p"], label="pressure (MPa)")
    ax.set_xlabel(tl); ax.set_ylabel("pressure (MPa)")
    ax3 = ax.twinx(); ax3.plot(t, S["H2"], color="#e15759", label="H₂"); ax3.set_ylabel("hydrogen (kg)", color="#e15759")
    ax3.grid(False)
    ax = fig.add_subplot(gs[1, 1])
    keys = ["frac_intact", "frac_failed", "frac_damaged", "frac_molten", "frac_relocated"]
    ax.stackplot(t, *[S[k] for k in keys], colors=STATE_COLORS, labels=STATE_NAMES)
    ax.set_ylim(0, 1); ax.set_xlabel(tl); ax.set_ylabel("fraction of fuel cells"); ax.legend(fontsize=7, loc="lower left")
    ax = fig.add_subplot(gs[:, 2])
    _state_map(ax, res, res.snapshots[-1], "damage state at end")
    fig.suptitle(f"{res.scenario}  ({res.reactor})", fontsize=10)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def lattice_figure(res, path, times, field="T_fuel", dpi=130, cmap="inferno", label="fuel temperature (K)"):
    """A row of lattice maps of one field at several times."""
    plt = _plt()
    fig, axs = plt.subplots(1, len(times), figsize=(2.3 * len(times) + 1, 3.6), constrained_layout=True)
    vals = [res.at(tt).fields[field] for tt in times]
    lo = np.nanmin([np.nanmin(v) for v in vals]); hi = np.nanmax([np.nanmax(v) for v in vals])
    for ax, tt, v in zip(np.atleast_1d(axs), times, vals):
        snap = res.at(tt)
        if field == "state":
            _state_map(ax, res, snap, f"t = {snap.t / 3600:.2f} h")
            continue
        im = ax.imshow(np.ma.masked_invalid(v.reshape(res.nz, res.nx)), origin="lower", cmap=cmap,
                       vmin=lo, vmax=hi, aspect="auto")
        ax.set_title(f"t = {snap.t / 3600:.2f} h" if snap.t > 600 else f"t = {snap.t:.1f} s")
        ax.set_xticks([]); ax.set_yticks([])
    if field != "state":
        fig.colorbar(im, ax=axs, label=label, shrink=0.85)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def ensemble_figure(ens, path, dpi=130):
    """Empirical distributions of the event times and the spread of intact fraction and hydrogen."""
    plt = _plt()
    from .core.ensemble import EVENTS, EVENT_LABELS
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    ax = axs[0]
    for i, ev in enumerate(EVENTS):
        tt = ens.times(ev)
        if len(tt):
            ax.step(np.r_[0, tt / 3600], np.r_[0, np.arange(1, len(tt) + 1) / ens.n], where="post",
                    label=EVENT_LABELS[ev])
    ax.set_xlabel("time (h)"); ax.set_ylabel("fraction of members"); ax.set_ylim(0, 1.02); ax.legend(fontsize=8)
    ax.set_title("time to event (empirical CDF)")
    for m in ens.members:
        axs[1].plot(np.asarray(m["t"]) / 3600, m["frac_intact"], color="#4e79a7", alpha=0.3, lw=0.8)
        axs[2].plot(np.asarray(m["t"]) / 3600, m["H2"], color="#e15759", alpha=0.3, lw=0.8)
    axs[1].set_xlabel("time (h)"); axs[1].set_ylabel("intact fuel cells"); axs[1].set_title(f"{ens.n} members")
    axs[2].set_xlabel("time (h)"); axs[2].set_ylabel("hydrogen (kg)"); axs[2].set_title("in-vessel hydrogen")
    fig.suptitle(ens.scenario, fontsize=10)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def xenon_figure(path, keys=("pwr", "bwr", "rbmk", "candu", "htgr"), dpi=130):
    plt = _plt()
    from .core import analytics as an
    from .core.reactors import REACTORS
    t = np.linspace(0, 72 * 3600, 1500)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for k in keys:
        rt = REACTORS[k]
        r = an.poison_transient(rt, t, 1.0, 0.0)
        axs[0].plot(t / 3600, r["xenon"] * 1e2, label=rt.key.upper())
        axs[1].plot(t / 3600, r["samarium"] * 1e2, label=rt.key.upper())
    axs[0].set_xlabel("time after trip (h)"); axs[0].set_ylabel("Xe-135 worth (% Δk/k)")
    axs[0].set_title("xenon pit after a trip from equilibrium"); axs[0].legend(fontsize=8)
    axs[1].set_xlabel("time after trip (h)"); axs[1].set_ylabel("Sm-149 worth (% Δk/k)")
    axs[1].set_title("samarium build-up (stable)")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def kinetics_figure(path, keys=("pwr", "bwr", "rbmk", "candu", "sfr-metal"), dpi=130):
    plt = _plt()
    from .core import analytics as an
    from .core.reactors import REACTORS
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    rho = np.geomspace(0.01, 1.6, 120)
    for k in keys:
        rt = REACTORS[k]
        _, T = an.inhour_curve(rt, rho)
        axs[0].loglog(rho, T, label=rt.key.upper())
    axs[0].axvline(1.0, color="#8c8c8c", ls="--", lw=0.8)
    axs[0].set_xlabel("reactivity ($)"); axs[0].set_ylabel("stable period (s)"); axs[0].legend(fontsize=8)
    axs[0].set_title("inhour equation")
    t = np.geomspace(1, 1e7, 400)
    for lab, y in an.decay_heat_curves(t).items():
        axs[1].loglog(t, y * 100, label=lab)
    axs[1].set_xlabel("time after shutdown (s)"); axs[1].set_ylabel("decay power (% of rated)")
    axs[1].set_title("ANS-5.1 decay heat by operating time"); axs[1].legend(fontsize=8)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
