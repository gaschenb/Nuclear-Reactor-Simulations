"""Reactor designs: key data and reactivity coefficients of the built-in reactor types, and the
feedback model used by the simulator."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from . import theme
from . import widgets as W
from .workspace import Workspace
from ..core import analytics as an
from ..core.reactors import REACTORS

DESIGNS = [(rt.name, k) for k, rt in REACTORS.items()]


class DesignsPage(Workspace):
    title = "Reactor designs"
    scenario_label = "Design"
    PARAMS = [W.P("dT", "temperature rise for the comparison (K)", 10.0, 500.0, 100.0, decimals=0),
              W.P("dvoid", "coolant void for the comparison (%)", 1.0, 100.0, 50.0, decimals=0)]
    CHOICES = [W.C("reactor", "design", DESIGNS, "pwr")]
    SCENARIOS = [dict(name=rt.name, desc=rt.description, values=dict(reactor=k)) for k, rt in REACTORS.items()]

    def build_views(self):
        self.pl_c = W.plot("reactivity of a uniform change ($)", None, "$", legend=True)
        self.pl_x = W.plot("equilibrium Xe-135 worth versus power", "power (× rated)", "% Δk/k", legend=True)
        self.tbl = QtTable()
        return [("Feedback comparison", self.pl_c), ("Xenon by design", self.pl_x), ("Design data", self.tbl)]

    def on_scenario(self):
        pass

    def equations(self):
        return [("h", "Feedback model (per cell)"),
                r"$\rho_D=2\alpha_D\sqrt{T_0}\,(\sqrt{T_f}-\sqrt{T_0})$",
                "Doppler broadening of U-238 resonances grows with √T.",
                r"$\rho_M=\alpha_M\,(T_m-T_{m,0}),\qquad\rho_V=\alpha_V\,(\alpha-\alpha_0)$",
                r"$\rho_R=\alpha_R\left[0.3\,\Delta T_{in}+0.7\,\Delta T_{out}\right]$",
                "Moderator / coolant temperature, void and (fast reactors) radial-expansion terms. "
                "The signs are design specific: light-water lattices are under-moderated (void and "
                "moderator temperature remove reactivity), RBMK and CANDU have a positive void "
                "coefficient, sodium voiding adds reactivity in a large fast core.",
                ("h", "Cross sections"),
                r"$\nu\Sigma_f=\nu\Sigma_{f,0}\,m\,(1+\rho_D+\rho_M+\rho_V+\rho_R)$",
                "m is the fuel inventory of the cell (relocation moves it)."]

    def compute(self, p, heavy, ctx):
        rows = []
        for k, rt in REACTORS.items():
            dop = 2 * rt.alpha_D * np.sqrt(rt.T_f_ref) * (np.sqrt(rt.T_f_ref + p["dT"]) - np.sqrt(rt.T_f_ref))
            dop += rt.alpha_fexp * p["dT"]
            rows.append(dict(key=k, doppler=dop / rt.beta_eff, mod=rt.alpha_mod * p["dT"] / rt.beta_eff,
                             void=rt.alpha_void * p["dvoid"] / 100 / rt.beta_eff,
                             radial=rt.alpha_radial * p["dT"] / rt.beta_eff))
        pw = np.linspace(0, 1.2, 61)
        xe = {k: np.array([an.xenon_equilibrium_worth(rt, x) for x in pw]) for k, rt in REACTORS.items()}
        return dict(rows=rows, pw=pw, xe=xe, p=p, table=an.reactor_table())

    def show_result(self, res):
        sel = res["p"]["reactor"]
        pi = self.pl_c.getPlotItem(); pi.clear(); pi.legend.clear()
        n = len(res["rows"])
        x = np.arange(n)
        wdt = 0.2
        for j, (key, lab) in enumerate([("doppler", f"fuel +{res['p']['dT']:.0f} K"),
                                        ("mod", f"moderator/coolant +{res['p']['dT']:.0f} K"),
                                        ("void", f"void {res['p']['dvoid']:.0f} %"),
                                        ("radial", f"core expansion +{res['p']['dT']:.0f} K")]):
            y = np.array([r[key] for r in res["rows"]])
            bar = pg.BarGraphItem(x=x + (j - 1.5) * wdt, height=y, width=wdt, brush=theme.color(j), pen=None)
            pi.addItem(bar)
            pi.legend.addItem(bar, lab)
        ax = pi.getAxis("bottom")
        short = {"rbmk-mod": "RBMK-M", "sfr-metal": "SFR-M", "sfr-oxide": "SFR-O"}
        ax.setTicks([[(i, short.get(r["key"], r["key"].upper())) for i, r in enumerate(res["rows"])]])
        k = [r["key"] for r in res["rows"]].index(sel)
        pi.addItem(pg.LinearRegionItem((k - 0.5, k + 0.5), movable=False, brush=pg.mkBrush(128, 128, 128, 30),
                                       pen=pg.mkPen(None)))
        W.hline(pi, 0.0, "ref", style=None)
        pi = self.pl_x.getPlotItem(); pi.clear(); pi.legend.clear()
        for i, (kk, y) in enumerate(res["xe"].items()):
            W.curve(pi, res["pw"], y * 100, i, name=kk.upper(), width=2.6 if kk == sel else 1.2)
        self.tbl.set_data(res["table"], sel)

    def summary(self, res):
        rt = REACTORS[res["p"]["reactor"]]
        return [("thermal power", f"{rt.power_MW:.0f} MW"), ("coolant / moderator", f"{rt.coolant} / {rt.moderator or rt.coolant}"),
                ("fuel / cladding", f"{rt.fuel} / {rt.clad}"), ("core height × width", f"{rt.height_m:.2f} × {rt.width_m:.2f} m"),
                ("lattice", f"{rt.nx} × {rt.nz} cells"), ("β_eff", f"{rt.beta_eff * 1e5:.0f} pcm"),
                ("Λ", f"{rt.Lambda * 1e6:.3g} µs"), ("Doppler α_D", f"{rt.alpha_D * 1e5:+.2f} pcm/K"),
                ("moderator α_M", f"{rt.alpha_mod * 1e5:+.2f} pcm/K"), ("void α_V", f"{rt.alpha_void * 1e2:+.2f} % Δk/k per unit void"),
                ("rated flux", f"{rt.phi_nom:.3g} n/cm²s"), ("Xe-135 at equilibrium", f"{an.xenon_equilibrium_worth(rt) * 100:+.2f} % Δk/k"),
                ("excess reactivity", f"{rt.excess * 100:.1f} % Δk/k"), ("scram worth", f"{rt.scram_worth:.1f} $")]


class QtTable(W.QtWidgets.QTableWidget):
    COLS = [("key", "design", "{}"), ("power_MW", "MWt", "{:.0f}"), ("coolant", "coolant", "{}"),
            ("moderator", "moderator", "{}"), ("fuel", "fuel", "{}"), ("beta_eff", "β (pcm)", "{:.0f}"),
            ("Lambda", "Λ (s)", "{:.1e}"), ("alpha_D", "α_D (pcm/K)", "{:+.2f}"),
            ("alpha_mod", "α_M (pcm/K)", "{:+.2f}"), ("alpha_void", "α_V (%Δk/k)", "{:+.2f}"),
            ("phi", "φ (n/cm²s)", "{:.2g}"), ("xe_eq", "Xe eq (%Δk/k)", "{:+.2f}")]
    SCALE = dict(beta_eff=1e5, alpha_D=1e5, alpha_mod=1e5, alpha_void=1e2, xe_eq=1e2)

    def __init__(self):
        super().__init__(0, len(self.COLS))
        self.setHorizontalHeaderLabels([c[1] for c in self.COLS])
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(W.QtWidgets.QAbstractItemView.NoEditTriggers)
        self.horizontalHeader().setSectionResizeMode(W.QtWidgets.QHeaderView.ResizeToContents)

    def set_data(self, rows, sel):
        self.setRowCount(len(rows))
        for i, r in enumerate(rows):
            for j, (k, _, f) in enumerate(self.COLS):
                v = r[k]
                if v is None:
                    v = "—"
                elif k in self.SCALE:
                    v = v * self.SCALE[k]
                it = W.QtWidgets.QTableWidgetItem(f.format(v) if not isinstance(v, str) else v)
                if r["key"] == sel:
                    font = it.font(); font.setBold(True); it.setFont(font)
                self.setItem(i, j, it)
