"""Fission-product poisoning: I-135 → Xe-135 and Pm-149 → Sm-149 after a power change, the
xenon pit / dead time and the equilibrium xenon worth versus power."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from . import widgets as W
from .workspace import Workspace
from ..core import analytics as an
from ..core.reactors import REACTORS

DESIGNS = [(rt.name, k) for k, rt in REACTORS.items()]


class PoisonPage(Workspace):
    title = "Xenon and samarium"
    scenario_label = "Example"
    PARAMS = [
        W.P("p0", "power before (× rated)", 0.0, 1.2, 1.0, decimals=2),
        W.P("p1", "power after (× rated)", 0.0, 1.2, 0.0, decimals=2),
        W.P("hours", "time span (h)", 6.0, 200.0, 72.0, decimals=1),
        W.P("excess", "excess reactivity available (%Δk/k)", 0.0, 10.0, 6.0, decimals=2,
            tip="reactivity the operators can withdraw to override the extra xenon"),
        W.P("phi", "flux at rated power (10¹³ n/cm²s)", 0.1, 30.0, 3.0, decimals=2, log=True),
    ]
    CHOICES = [W.C("reactor", "design", DESIGNS, "pwr")]
    SCENARIOS = [
        dict(name="PWR trip from full power", desc="Xenon peaks ~8–10 h after the trip; the large excess "
             "reactivity of a fresh PWR core overrides it.", values=dict(reactor="pwr", p0=1, p1=0, excess=6.0, phi=3.0)),
        dict(name="RBMK trip: xenon dead time", desc="A high-flux, low-excess graphite core cannot restart "
             "for about a day after a trip from full power (the operating margin is a few % Δk/k).",
             values=dict(reactor="rbmk", p0=1, p1=0, excess=1.0, phi=5.0)),
        dict(name="Power reduction 100 → 50 %", desc="The xenon worth first rises (iodine keeps feeding it "
             "while burn-out falls), then settles at the new equilibrium.", values=dict(reactor="pwr", p0=1, p1=0.5, excess=6.0, phi=3.0)),
        dict(name="Power increase 50 → 100 %", desc="Burn-out outruns production: xenon drops first, then "
             "builds to the higher equilibrium.", values=dict(reactor="pwr", p0=0.5, p1=1.0, excess=6.0, phi=3.0)),
        dict(name="CANDU trip (poison-out)", desc="Natural-uranium CANDU has little excess reactivity: "
             "without adjuster withdrawal it poisons out ~30–40 min after a trip.",
             values=dict(reactor="candu", p0=1, p1=0, excess=1.5, phi=7.0)),
        dict(name="Fast reactor", desc="A fast spectrum has a tiny Xe-135 absorption share: no xenon pit.",
             values=dict(reactor="sfr-metal", p0=1, p1=0, excess=1.0, phi=10.0)),
    ]

    def build_views(self):
        self.pl_w = W.plot("worth after the power change", "time (h)", "reactivity (% Δk/k)", legend=True)
        self.pl_n = W.plot("concentrations relative to the initial equilibrium", "time (h)", "N / N₀", legend=True)
        self.pl_eq = W.plot("equilibrium xenon worth versus power", "power (× rated)", "% Δk/k", legend=True)
        return [("Reactivity", self.pl_w), ("Concentrations", self.pl_n), ("Equilibrium", self.pl_eq)]

    def equations(self):
        return [("h", "Iodine–xenon chain"),
                r"$\frac{dI}{dt}=\gamma_I\Sigma_f\phi-\lambda_II$",
                r"$\frac{dX}{dt}=\gamma_X\Sigma_f\phi+\lambda_II-(\lambda_X+\sigma_X\phi)X$",
                r"$X_\infty=\frac{(\gamma_I+\gamma_X)\Sigma_f\phi}{\lambda_X+\sigma_X\phi},\quad\rho_X=-\frac{\gamma_I+\gamma_X}{\nu}\,\frac{\phi/\phi_X}{1+\phi/\phi_X}$",
                r"$\phi_X=\lambda_X/\sigma_X\approx0.76\times10^{13}\,\mathrm{cm^{-2}s^{-1}}$",
                "After a trip the iodine inventory keeps producing xenon that is no longer burnt out: the "
                "worth passes through a maximum (the xenon pit) before decaying with T½ = 9.14 h.",
                ("h", "Promethium–samarium chain"),
                r"$\frac{dP}{dt}=\gamma_P\Sigma_f\phi-\lambda_PP,\quad\frac{dS}{dt}=\lambda_PP-\sigma_S\phi S$",
                "Sm-149 is stable: after shutdown it only builds up (no decay), by the Pm-149 inventory."]

    def compute(self, p, heavy, ctx):
        from dataclasses import replace
        rt = replace(REACTORS[p["reactor"]], phi_nom=p["phi"] * 1e13)
        t = np.linspace(0, p["hours"] * 3600, 1500)
        r = an.poison_transient(rt, t, p["p0"], p["p1"])
        pw = np.linspace(0.0, 1.2, 121)
        eq = np.array([an.xenon_equilibrium_worth(rt, x) for x in pw])
        extra = r["xenon"] - r["xenon"][0]
        lock = np.nonzero(-extra > p["excess"] / 100)[0]
        out = dict(t=t, r=r, pw=pw, eq=eq, extra=extra, excess=p["excess"] / 100,
                   lock=(t[lock[0]], t[lock[-1]]) if lock.size and p["p1"] < p["p0"] else None,
                   eq0=an.xenon_equilibrium_worth(rt, p["p0"]), eq1=an.xenon_equilibrium_worth(rt, p["p1"]),
                   rt=rt, p=p)
        return out

    def show_result(self, res):
        t = res["t"] / 3600
        r = res["r"]
        pi = self.pl_w.getPlotItem(); pi.clear(); pi.legend.clear()
        W.curve(pi, t, r["xenon"] * 100, "xenon", name="Xe-135")
        W.curve(pi, t, r["samarium"] * 100, "samarium", name="Sm-149")
        W.curve(pi, t, (r["xenon"] + r["samarium"]) * 100, "fg", name="total", style="dash")
        W.hline(pi, (r["xenon"][0] - res["excess"]) * 100, "limit", label="restart limit (excess)")
        if res["lock"]:
            a, b = res["lock"]
            reg = pg.LinearRegionItem((a / 3600, b / 3600), movable=False,
                                      brush=pg.mkBrush(225, 87, 89, 40), pen=pg.mkPen(None))
            pi.addItem(reg)
        pi = self.pl_n.getPlotItem(); pi.clear(); pi.legend.clear()
        W.curve(pi, t, r["iodine"], "iodine", name="I-135")
        W.curve(pi, t, r["xe_rel"] / max(r["xe_rel"][0], 1e-30), "xenon", name="Xe-135")
        pi = self.pl_eq.getPlotItem(); pi.clear(); pi.legend.clear()
        W.curve(pi, res["pw"], res["eq"] * 100, "xenon", name="Xe-135 at equilibrium")
        W.vline(pi, res["p"]["p0"], "ref", label="before")
        W.vline(pi, res["p"]["p1"], "limit", label="after")

    def summary(self, res):
        r, t = res["r"], res["t"]
        i = int(np.argmin(r["xenon"]))
        rows = [("equilibrium Xe worth before", f"{res['eq0'] * 100:+.3f} % Δk/k"),
                ("equilibrium Xe worth after", f"{res['eq1'] * 100:+.3f} % Δk/k"),
                ("most negative Xe worth", f"{r['xenon'][i] * 100:+.3f} % Δk/k"),
                ("at", f"{t[i] / 3600:.2f} h"),
                ("Sm-149 worth at end", f"{r['samarium'][-1] * 100:+.3f} % Δk/k"),
                ("flux / φ_X", f"{res['rt'].phi_nom / (2.09e-5 / 2.65e-18) :.2f}")]
        if res["lock"]:
            a, b = res["lock"]
            rows.append(("restart impossible", f"{a / 3600:.2f} h – {b / 3600:.2f} h"))
        elif res["p"]["p1"] < res["p"]["p0"]:
            rows.append(("restart impossible", "never (excess suffices)"))
        return rows

    def export_data(self, path):
        import csv
        res = self.res
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t_s", "xenon_dk", "samarium_dk", "iodine_rel", "xenon_rel"])
            for row in zip(res["t"], res["r"]["xenon"], res["r"]["samarium"], res["r"]["iodine"], res["r"]["xe_rel"]):
                w.writerow([f"{v:.6g}" for v in row])
