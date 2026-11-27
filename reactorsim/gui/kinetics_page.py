"""Reactor kinetics: point-kinetics response to a reactivity step (six delayed groups, exact
matrix exponential), the inhour relation, prompt jump and the Nordheim–Fuchs prompt burst."""
from __future__ import annotations

import numpy as np
from dataclasses import replace

from . import widgets as W
from .workspace import Workspace
from ..core import analytics as an
from ..core.neutronics import PointKinetics
from ..core.nuclear_data import delayed_groups
from ..core.reactors import REACTORS

DESIGNS = [(rt.name, k) for k, rt in REACTORS.items()]


def response(rt, rho_d, t_end, n_pts=400):
    beta, lam = delayed_groups(rt.fissions, rt.beta_eff)
    pk = PointKinetics(beta, lam, rt.Lambda, source=0.0)
    pk.set_steady(1.0)
    t = np.concatenate([[0.0], np.geomspace(1e-4 * max(t_end, 1e-3), t_end, n_pts)])
    n = [1.0]
    for a, b in zip(t[:-1], t[1:]):
        pk.step(rho_d * rt.beta_eff, b - a)
        n.append(pk.n)
    return t, np.array(n)


class KineticsPage(Workspace):
    title = "Kinetics"
    PARAMS = [
        W.P("rho", "reactivity step ($)", -3.0, 2.0, 0.3, decimals=3),
        W.P("t_end", "time span (s)", 0.01, 600.0, 60.0, log=True, decimals=3),
        W.P("beta", "β_eff (pcm)", 200.0, 800.0, 650.0, decimals=0),
        W.P("Lambda", "generation time Λ (µs)", 0.1, 2000.0, 20.0, log=True, decimals=2),
    ]
    CHOICES = [W.C("reactor", "design (delayed-neutron data)", DESIGNS, "pwr")]
    SCENARIOS = [
        dict(name="PWR, +30 ¢ step", desc="Prompt jump to 1/(1−ρ) followed by a stable period of ~25 s.",
             values=dict(reactor="pwr", rho=0.3, t_end=60, beta=600, Lambda=20)),
        dict(name="PWR scram (−5 $)", desc="Prompt drop to β/(β−ρ) ≈ 17 %, then the −80 s period of the "
             "longest-lived precursor group.", values=dict(reactor="pwr", rho=-5.0, t_end=600, beta=600, Lambda=20)),
        dict(name="Prompt critical, 1.2 $", desc="Above one dollar the prompt neutrons alone are supercritical: "
             "the period is set by Λ and the burst is ended by Doppler feedback (Nordheim–Fuchs).",
             values=dict(reactor="pwr", rho=1.2, t_end=0.05, beta=600, Lambda=20)),
        dict(name="RBMK, +2 $ void excursion", desc="Graphite-moderated: Λ ≈ 1 ms, so even a superprompt "
             "excursion develops over tens of milliseconds.", values=dict(reactor="rbmk", rho=2.0, t_end=0.5, beta=480, Lambda=1000)),
        dict(name="Fast reactor, +0.5 $", desc="β ≈ 350 pcm (Pu-239) and Λ ≈ 0.4 µs: same delayed period, "
             "but prompt-critical transients are thousands of times faster.",
             values=dict(reactor="sfr-metal", rho=0.5, t_end=60, beta=350, Lambda=0.4)),
    ]

    def build_views(self):
        self.pl_n = W.plot("relative power after the step", "time (s)", "n(t) / n₀", legend=True)
        self.pl_ih = W.plot("inhour equation: stable period", "reactivity ($)", "period (s)", legend=True)
        self.pl_nf = W.plot("Nordheim–Fuchs burst (adiabatic Doppler feedback)", "time (ms)", "power (× rated)", legend=True)
        return [("Power response", self.pl_n), ("Inhour", self.pl_ih), ("Prompt burst", self.pl_nf)]

    def on_scenario(self):
        pass

    def equations(self):
        return [("h", "Point kinetics, six delayed groups"),
                r"$\frac{dn}{dt}=\frac{\rho-\beta}{\Lambda}n+\sum_i\lambda_iC_i,\quad\frac{dC_i}{dt}=\frac{\beta_i}{\Lambda}n-\lambda_iC_i$",
                "Solved exactly over each step with the matrix exponential.",
                ("h", "Inhour equation and prompt jump"),
                r"$\rho=\omega\Lambda+\sum_i\frac{\beta_i\,\omega}{\omega+\lambda_i},\quad T=1/\omega$",
                r"$\frac{n_+}{n_-}=\frac{\beta}{\beta-\rho}$",
                ("h", "Nordheim–Fuchs burst"),
                r"$\rho(t)=\rho_0-\gamma E(t),\quad\alpha_0=\frac{\rho_0-\beta}{\Lambda}$",
                r"$P_{max}=\frac{\alpha_0^2\Lambda}{2\gamma},\quad E=\frac{2\alpha_0\Lambda}{\gamma},\quad\mathrm{FWHM}=\frac{3.52}{\alpha_0}$"]

    def _rt(self, p):
        return replace(REACTORS[p["reactor"]], beta_eff=p["beta"] * 1e-5, Lambda=p["Lambda"] * 1e-6)

    def compute(self, p, heavy, ctx):
        rt = self._rt(p)
        rho = p["rho"]
        t_end = p["t_end"]
        if rho > 1.0:
            # without feedback the prompt-supercritical response is a pure exponential: cap the span
            alpha = (rho - 1.0) * rt.beta_eff / rt.Lambda
            t_end = min(t_end, 30.0 / alpha)
        t, n = response(rt, rho, t_end)
        rr = np.concatenate([-np.geomspace(5, 0.005, 60), np.geomspace(0.005, 1.8, 80)])
        _, T = an.inhour_curve(rt, rr)
        T_here = float(an.inhour_curve(rt, np.array([rho]))[1][0]) if abs(rho) > 1e-6 else np.inf
        nf = an.nordheim_fuchs(rt, rho) if rho > 1.0 else None
        nf_curve = an.nordheim_fuchs_curve(nf) if nf else None
        return dict(t=t, n=n, rr=rr, T=T, T_here=T_here, nf=nf, nf_curve=nf_curve, rt=rt, p=p, t_end=t_end)

    def show_result(self, res):
        p = res["p"]
        pi = self.pl_n.getPlotItem(); pi.clear(); pi.legend.clear()
        pi.setLogMode(x=False, y=True)
        W.curve(pi, res["t"], np.maximum(res["n"], 1e-12), "power", name="point kinetics")
        if p["rho"] < 1.0:
            pj = 1.0 / (1.0 - p["rho"])
            W.hline(pi, np.log10(pj), "ref", label=f"prompt jump ×{pj:.3g}")
            if np.isfinite(res["T_here"]):
                tt = res["t"]
                W.curve(pi, tt, pj * np.exp(tt / res["T_here"]), "limit", name="stable period asymptote",
                        style="dash", width=1.0)
        pi = self.pl_ih.getPlotItem(); pi.clear(); pi.legend.clear()
        pi.setLogMode(x=False, y=True)
        rr, T = res["rr"], res["T"]
        pos = rr > 0
        W.curve(pi, rr[pos], T[pos], "power", name="positive period")
        W.curve(pi, rr[~pos], -T[~pos], "decay", name="|negative period|")
        W.vline(pi, 1.0, "ref", label="prompt critical")
        if np.isfinite(res["T_here"]) and res["T_here"] != 0:
            pi.plot([p["rho"]], [abs(res["T_here"])], pen=None, symbol="o", symbolSize=9,
                    symbolBrush=W.theme.color("limit"))
        pi = self.pl_nf.getPlotItem(); pi.clear(); pi.legend.clear()
        if res["nf_curve"] is not None:
            t, P = res["nf_curve"]
            W.curve(pi, t * 1e3, P / (res["rt"].power_MW * 1e6), "power", name="P(t) = P_max sech²(α₀t/2)")
        else:
            W.text(pi, 0, 0, "the burst model applies above one dollar")

    def summary(self, res):
        p, rt = res["p"], res["rt"]
        rows = [("β_eff", f"{rt.beta_eff * 1e5:.0f} pcm"), ("Λ", f"{rt.Lambda * 1e6:.3g} µs"),
                ("reactivity", f"{p['rho']:+.3f} $ = {p['rho'] * rt.beta_eff * 1e5:+.0f} pcm"),
                ("stable period", W.fmt(res["T_here"], "s"))]
        if p["rho"] < 1.0:
            rows.append(("prompt jump n₊/n₋", f"{1 / (1 - p['rho']):.4g}"))
        rows.append((f"n after {res['t_end']:.3g} s", f"{res['n'][-1]:.4g}"))
        nf = res["nf"]
        if nf:
            rows += [("prompt period 1/α₀", f"{nf['period'] * 1e3:.3g} ms"),
                     ("burst peak", f"{nf['peak_rel']:.3g} × rated"),
                     ("burst width (FWHM)", f"{nf['fwhm_s'] * 1e3:.3g} ms"),
                     ("burst energy", f"{nf['energy_J'] / 1e6:.4g} MJ"),
                     ("core-average fuel heat-up", f"{nf['dT_fuel']:.0f} K")]
        return rows
