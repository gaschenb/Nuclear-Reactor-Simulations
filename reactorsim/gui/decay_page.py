"""Decay heat after shutdown (ANS-5.1 and Way–Wigner), the boil-off time of a water-cooled core
with no injection and the adiabatic heat-up rate of an uncooled core."""
from __future__ import annotations

import numpy as np
from dataclasses import replace

from . import widgets as W
from .workspace import Workspace
from ..core import analytics as an
from ..core.nuclear_data import decay_heat_fraction, way_wigner
from ..core.reactors import REACTORS

DESIGNS = [(rt.name, k) for k, rt in REACTORS.items()]


def _trapz(y, x):
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


class DecayPage(Workspace):
    title = "Decay heat"
    PARAMS = [
        W.P("days", "operating time before shutdown (d)", 0.1, 3000.0, 365.0, log=True, decimals=1),
        W.P("t_mark", "time after shutdown (s)", 1.0, 1e7, 3600.0, log=True, decimals=0),
        W.P("power", "rated power (MWt)", 10.0, 5000.0, 3411.0, log=True, decimals=0),
    ]
    CHOICES = [W.C("reactor", "design", DESIGNS, "pwr")]
    SCENARIOS = [
        dict(name="PWR, one year at power", desc="About 6.5 % of rated power one second after the trip, "
             "1.5 % after an hour and 0.5 % after a day.", values=dict(reactor="pwr", days=365, t_mark=3600, power=3411)),
        dict(name="BWR boil-off without injection", desc="A BWR's large water inventory above the core buys "
             "about an hour before the fuel uncovers.", values=dict(reactor="bwr", days=500, t_mark=3600, power=2381)),
        dict(name="Short run (1 day)", desc="After a short irradiation the long-lived fission products have "
             "not built up: decay heat falls much faster after the first hours.",
             values=dict(reactor="pwr", days=1, t_mark=86400, power=3411)),
        dict(name="HTGR, low power density", desc="Same decay fraction, but 350 MWt spread through hundreds "
             "of tonnes of graphite: the heat-up is a fraction of a kelvin per minute.",
             values=dict(reactor="htgr", days=500, t_mark=3600, power=350)),
    ]

    def build_views(self):
        self.pl_d = W.plot("decay power after shutdown", "time after shutdown (s)", "% of rated power", legend=True)
        self.pl_b = W.plot("water boiled off by decay heat (no injection)", "time after shutdown (h)",
                           "water boiled (t)", legend=True)
        self.pl_h = W.plot("adiabatic heat-up rate of an uncooled core", "time after shutdown (s)", "K/s", legend=True)
        return [("Decay heat", self.pl_d), ("Boil-off", self.pl_b), ("Heat-up", self.pl_h)]

    def on_scenario(self):
        pass

    def equations(self):
        return [("h", "ANS-5.1 decay heat"),
                r"$\frac{P_d(t,T)}{P_0}=\sum_{n=1}^{23}\frac{\alpha_n}{\lambda_n}\left(1-e^{-\lambda_nT}\right)e^{-\lambda_nt}\,/\,Q$",
                "23-exponential fit of the U-235 fission-product decay power (ANS-5.1-1979) plus the "
                "U-239 and Np-239 capture products (R = 0.6 per fission).",
                ("h", "Way–Wigner"),
                r"$\frac{P_d}{P_0}=0.0622\left[t^{-0.2}-(t+T)^{-0.2}\right]$",
                ("h", "Boil-off and heat-up"),
                r"$m_{boiled}(t)=\frac{1}{h_{fg}}\int_{t_s}^{t}P_d\,dt',\qquad\frac{dT}{dt}=\frac{P_d}{m_fc_{p,f}+m_cc_{p,c}}$"]

    def compute(self, p, heavy, ctx):
        rt = replace(REACTORS[p["reactor"]], power_MW=p["power"], t_op_days=p["days"])
        t = np.geomspace(1.0, 1e7, 400)
        T = p["days"] * 86400.0
        ans = decay_heat_fraction(t, T)
        ww = way_wigner(t, T)
        inf = decay_heat_fraction(t, np.inf)
        bo = an.boil_off(rt, 60.0)
        hr = np.array([an.adiabatic_heatup_rate(rt, x) for x in t])
        return dict(t=t, ans=ans, ww=ww, inf=inf, bo=bo, hr=hr, rt=rt, p=p,
                    at=float(decay_heat_fraction(np.array(p["t_mark"]), T)))

    def show_result(self, res):
        p = res["p"]
        pi = self.pl_d.getPlotItem(); pi.clear(); pi.legend.clear()
        pi.setLogMode(x=True, y=True)
        W.curve(pi, res["t"], res["ans"] * 100, "power", name=f"ANS-5.1, {p['days']:g} d")
        W.curve(pi, res["t"], res["inf"] * 100, "ref", name="ANS-5.1, infinite operation", style="dash")
        W.curve(pi, res["t"], res["ww"] * 100, "decay", name="Way–Wigner", style="dot")
        W.vline(pi, np.log10(p["t_mark"]), "limit")
        pi = self.pl_b.getPlotItem(); pi.clear(); pi.legend.clear()
        bo = res["bo"]
        if bo is not None:
            W.curve(pi, bo["t"] / 3600, bo["boiled"] / 1e3, "cool", name="boiled")
            rt = res["rt"]
            W.hline(pi, (rt.M_nom - rt.M_top), "decay", label="top of fuel uncovered")
            W.hline(pi, (rt.M_nom - rt.M_bot), "limit", label="core dry")
            pi.setXRange(0, min(bo["t_dry"] * 1.6, 30 * 86400) / 3600)
            pi.setYRange(0, (rt.M_nom - rt.M_bot) * 1.3)
        else:
            W.text(pi, 0, 0, "not a water-cooled design")
        pi = self.pl_h.getPlotItem(); pi.clear(); pi.legend.clear()
        pi.setLogMode(x=True, y=True)
        W.curve(pi, res["t"], res["hr"], "fuel", name="fuel + cladding, no cooling")

    def summary(self, res):
        p, rt = res["p"], res["rt"]
        rows = [(f"decay power at {W.fmt(p['t_mark'], 's')}", f"{res['at'] * 100:.3f} % = {res['at'] * rt.power_MW:.3g} MW"),
                ("at 1 s / 1 h / 1 d", " / ".join(f"{float(decay_heat_fraction(np.array(x), p['days'] * 86400)) * 100:.2f} %"
                                                  for x in (1.0, 3600.0, 86400.0))),
                ("energy in the first day", f"{_trapz(res['ans'][res['t'] <= 86400] * rt.power_MW, res['t'][res['t'] <= 86400]) / 1e3:.3g} GJ"),
                ("adiabatic heat-up at the mark", f"{an.adiabatic_heatup_rate(rt, p['t_mark']):.3g} K/s")]
        bo = res["bo"]
        if bo is not None:
            rows += [("core uncovery (no injection, trip at 60 s)", f"{bo['t_uncover'] / 3600:.2f} h"),
                     ("core dry", f"{bo['t_dry'] / 3600:.2f} h")]
        return rows
