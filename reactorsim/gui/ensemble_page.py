"""Monte Carlo ensembles: many realisations of the stochastic damage chain (and optionally of
uncertain damage-progression parameters) run in parallel processes. The page fills in live as
members finish: cell-by-cell damage-probability maps through time, percentile fans of the plant
variables, the spread of event times, outcome probabilities with confidence intervals and the
sensitivity of the outcome to each uncertain input."""
from __future__ import annotations

import time

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

from . import theme
from . import widgets as W
from .fields import STATE_NAMES, Field, fmt
from .lattice import LatticeView
from .scenario_editor import ScenarioEditor
from ..core.ensemble import EVENTS, EVENT_LABELS, UNCERTAIN, EnsembleResult, wilson
from ..core.reactors import REACTORS
from ..core.scenarios import SCENARIOS

DAMAGE_SCENARIOS = [k for k, s in SCENARIOS.items()
                    if any(w in k for w in ("sbo", "lbloca-nolpi", "loecc", "tmi", "1986", "ulof"))]

SERIES = [("frac_intact", "intact fuel cells", "fraction", 1.0),
          ("frac_molten", "molten fuel cells", "fraction", 1.0),
          ("frac_relocated", "relocated fuel cells", "fraction", 1.0),
          ("H2", "hydrogen generated", "kg", 1.0),
          ("T_clad_max", "peak cladding temperature", "K", 1.0),
          ("T_fuel_max", "peak fuel temperature", "K", 1.0),
          ("P_total", "thermal power", "MW", 1.0),
          ("p", "primary pressure", "MPa", 1.0),
          ("release", "volatile fission products released", "%", 1.0)]
METRICS = [("H2_kg", "hydrogen generated (kg)"), ("peak_clad_T", "peak cladding temperature (K)"),
           ("peak_fuel_T", "peak fuel temperature (K)"), ("molten_max", "peak molten fraction (%)"),
           ("relocated", "relocated fraction (%)"), ("failed", "failed-or-worse fraction (%)"),
           ("release", "fission products released (%)"), ("peak_power", "peak power (× rated)")]
METRICS += [("t:" + e, f"time of {EVENT_LABELS[e]} (h)") for e in EVENTS]
STATES = [(1, "clad failed or worse"), (2, "damaged or worse"), (3, "molten or worse"), (4, "relocated")]


def _metric(m, key):
    if key.startswith("t:"):
        t = m["times"].get(key[2:])
        return t / 3600 if t is not None else np.nan
    v = float(m[key])
    return v * 100 if key in ("molten_max", "relocated", "failed") else v


def _ranks(x):
    r = np.empty(len(x)); r[np.argsort(x, kind="mergesort")] = np.arange(len(x))
    return r


def spearman(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return np.nan
    return float(np.corrcoef(_ranks(x[ok]), _ranks(y[ok]))[0, 1])


class EnsemblePage(QtWidgets.QWidget):
    status = QtCore.pyqtSignal(str)
    title = "Ensemble"
    MIN_SIZE = (1450, 800)
    member_signal = QtCore.pyqtSignal(int, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("page")
        self.res = None
        self.last_error = None
        self.runner = W.Runner(self)
        self.heavy_runner = self.runner
        self.runner.done.connect(self._done)
        self.runner.failed.connect(self._failed)
        self.runner.progressed.connect(self._progress)
        self.member_signal.connect(self._member)
        self._cache = None
        self._build()

    # ------------------------------------------------------------------ layout
    def _build(self):
        h = QtWidgets.QHBoxLayout(self); h.setContentsMargins(10, 10, 10, 10); h.setSpacing(10)
        self.split = QtWidgets.QSplitter(Qt.Horizontal); self.split.setHandleWidth(8)
        h.addWidget(self.split)
        left = QtWidgets.QScrollArea(); left.setWidgetResizable(True); left.setMinimumWidth(300)
        left.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        lw = QtWidgets.QWidget(); lv = QtWidgets.QVBoxLayout(lw)
        lv.setContentsMargins(0, 0, 4, 0); lv.setSpacing(10)
        f, v = W.card("Scenario")
        self.cb_scn = QtWidgets.QComboBox()
        self.cb_scn.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cb_scn.setMinimumContentsLength(12)
        for k in DAMAGE_SCENARIOS + [k for k in SCENARIOS if k not in DAMAGE_SCENARIOS]:
            s = SCENARIOS[k]
            self.cb_scn.addItem(f"{REACTORS[s.reactor].key.upper()} · {s.name}", k)
        self.cb_scn.currentIndexChanged.connect(self._scenario_changed)
        v.addWidget(self.cb_scn)
        self.lab_desc = W.muted("")
        v.addWidget(self.lab_desc)
        lv.addWidget(f)
        f, v = W.card("Ensemble")
        import os
        ncpu = max(1, (os.cpu_count() or 2) - 1)
        self.params = W.ParamPanel([
            W.P("n", "members", 2, 400, 24, integer=True),
            W.P("uncertain", "parameter uncertainty (× nominal spread)", 0.0, 3.0, 1.0, decimals=2,
                tip="0: only the random damage chain differs between members"),
            W.P("workers", "parallel processes", 1, 64, ncpu, integer=True),
            W.P("seed", "first seed", 0, 100000, 0, integer=True)])
        v.addWidget(self.params)
        row = QtWidgets.QHBoxLayout()
        self.btn_run = QtWidgets.QPushButton("Run ensemble"); self.btn_run.setObjectName("primary")
        self.btn_run.clicked.connect(lambda: self.run_all())
        b = QtWidgets.QPushButton("Cancel"); b.clicked.connect(self.cancel)
        row.addWidget(self.btn_run, 1); row.addWidget(b)
        v.addLayout(row)
        self.progress = QtWidgets.QProgressBar(); self.progress.setRange(0, 100); self.progress.setVisible(False)
        v.addWidget(self.progress)
        unc = "<br>".join(f"{k}: {'× e^N(0, ' + str(s) + ')' if kind == 'log' else '± ' + str(s) + ' K'}"
                          for k, (kind, s) in UNCERTAIN.items())
        lab = W.muted("Uncertain inputs sampled per member:<br>" + unc)
        lab.setTextFormat(Qt.RichText)
        v.addWidget(lab)
        lv.addWidget(f)
        f, v = W.card("Edit scenario", "The same edits apply to every member.")
        self.editor = ScenarioEditor()
        v.addWidget(self.editor)
        lv.addWidget(f)
        lv.addStretch(1)
        left.setWidget(lw)
        self.split.addWidget(left)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_maps(), "Damage probability map")
        self.tabs.addTab(self._build_fans(), "Time evolution")
        self.tabs.addTab(self._build_events(), "Event timing")
        self.tabs.addTab(self._build_outcomes(), "Outcomes and sensitivity")
        self.tabs.currentChanged.connect(lambda _: self.refresh())
        self.split.addWidget(self.tabs)

        right = QtWidgets.QScrollArea(); right.setWidgetResizable(True); right.setMinimumWidth(300)
        right.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        rw = QtWidgets.QWidget(); rv = QtWidgets.QVBoxLayout(rw)
        rv.setContentsMargins(0, 0, 4, 0); rv.setSpacing(10)
        f, v = W.card("Outcome probabilities", "Wilson 95 % interval; times over the members that reach the event.")
        self.table = self._table(["outcome", "probability [95 % CI]", "median time (5–95 %)"])
        self.table.setMinimumHeight(190)
        v.addWidget(self.table)
        rv.addWidget(f)
        f, v = W.card("Sensitivity", "Spearman rank correlation of the selected outcome with each uncertain input.")
        self.sens = self._table(["input", "ρ", ""])
        self.sens.setMinimumHeight(220)
        v.addWidget(self.sens)
        rv.addWidget(f)
        f, v = W.card("Model")
        self.eq = W.EquationView(self.equations())
        v.addWidget(self.eq)
        rv.addWidget(f)
        rv.addStretch(1)
        right.setWidget(rw)
        self.split.addWidget(right)
        self.split.setStretchFactor(1, 1)
        self.split.setSizes([370, 1010, 440])
        self._scenario_changed()

    def _table(self, headers):
        t = QtWidgets.QTableWidget(0, len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.verticalHeader().setVisible(False)
        t.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        t.horizontalHeader().setStretchLastSection(True)
        t.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        t.verticalHeader().setDefaultSectionSize(22)
        return t

    def _build_maps(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w); v.setContentsMargins(6, 6, 6, 6)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("probability that a cell is"))
        self.cb_state = QtWidgets.QComboBox()
        for s, name in STATES:
            self.cb_state.addItem(name, s)
        self.cb_state.setCurrentIndex(2)
        self.cb_state.currentIndexChanged.connect(lambda _: self._show_maps())
        bar.addWidget(self.cb_state)
        bar.addSpacing(16)
        bar.addWidget(QtWidgets.QLabel("second map"))
        self.cb_map2 = QtWidgets.QComboBox()
        self.cb_map2.addItem("median time to reach the state", "tmed")
        self.cb_map2.addItem("mean peak fuel temperature", "Tpk")
        self.cb_map2.currentIndexChanged.connect(lambda _: self._show_maps())
        bar.addWidget(self.cb_map2)
        bar.addStretch(1)
        self.lab_cell = W.muted("hover over a cell for its probability")
        self.lab_cell.setWordWrap(False)
        bar.addWidget(self.lab_cell)
        v.addLayout(bar)
        g = QtWidgets.QHBoxLayout(); g.setSpacing(8)
        self.map_p = LatticeView(Field("prob", "probability", "%", "magma", digits=3))
        self.map_t = LatticeView(Field("tmed", "median time reached", "h", "viridis_r", digits=3))
        for m in (self.map_p, self.map_t):
            m.hovered.connect(self._hover)
            g.addWidget(m)
        v.addLayout(g, 1)
        trow = QtWidgets.QHBoxLayout()
        self.btn_play = QtWidgets.QToolButton(); self.btn_play.setCheckable(True)
        self.btn_play.setText("▶"); self.btn_play.setFixedWidth(40)
        self.btn_play.toggled.connect(self._play)
        self.slider = QtWidgets.QSlider(Qt.Horizontal); self.slider.setRange(0, 199); self.slider.setValue(199)
        self.slider.valueChanged.connect(lambda _: self._show_maps())
        self.lab_time = QtWidgets.QLabel("t = —"); self.lab_time.setObjectName("readout")
        self.lab_time.setMinimumWidth(110)
        trow.addWidget(self.btn_play); trow.addWidget(self.slider, 1); trow.addWidget(self.lab_time)
        v.addLayout(trow)
        self.pl_stack = pg.PlotWidget(); self.pl_stack.setFixedHeight(210)
        pi = self.pl_stack.getPlotItem(); pi.setMenuEnabled(False)
        W.style(pi, "expected core state (mean over members)", "time (h)", "fraction of fuel cells")
        W.side_legend(pi)
        v.addWidget(self.pl_stack)
        self._timer = QtCore.QTimer(self); self._timer.setInterval(60)
        self._timer.timeout.connect(self._tick)
        return w

    def _build_fans(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w); v.setContentsMargins(6, 6, 6, 6)
        bar = QtWidgets.QHBoxLayout()
        self.cb_series = QtWidgets.QComboBox()
        for k, name, unit, _ in SERIES:
            self.cb_series.addItem(f"{name} ({unit})", k)
        self.cb_series.currentIndexChanged.connect(lambda _: self._show_fans())
        self.chk_members = QtWidgets.QCheckBox("show members"); self.chk_members.setChecked(True)
        self.chk_members.toggled.connect(lambda _: self._show_fans())
        self.chk_logy = QtWidgets.QCheckBox("log scale")
        self.chk_logy.toggled.connect(lambda _: self._show_fans())
        bar.addWidget(self.cb_series); bar.addWidget(self.chk_members); bar.addWidget(self.chk_logy)
        bar.addStretch(1)
        v.addLayout(bar)
        self.pl_fan = pg.PlotWidget()
        pi = self.pl_fan.getPlotItem(); pi.setMenuEnabled(False)
        W.side_legend(pi)
        v.addWidget(self.pl_fan, 1)
        self.pl_fan2 = pg.PlotWidget()
        pi = self.pl_fan2.getPlotItem(); pi.setMenuEnabled(False)
        W.side_legend(pi)
        self.pl_fan2.setXLink(self.pl_fan)
        v.addWidget(self.pl_fan2, 1)
        return w

    def _build_events(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w); v.setContentsMargins(6, 6, 6, 6)
        self.pl_strip = pg.PlotWidget()
        pi = self.pl_strip.getPlotItem(); pi.setMenuEnabled(False)
        v.addWidget(self.pl_strip, 3)
        self.pl_cdf = pg.PlotWidget()
        pi = self.pl_cdf.getPlotItem(); pi.setMenuEnabled(False)
        W.side_legend(pi)
        self.pl_cdf.setXLink(self.pl_strip)
        v.addWidget(self.pl_cdf, 2)
        return w

    def _build_outcomes(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w); v.setContentsMargins(6, 6, 6, 6)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("outcome"))
        self.cb_metric = QtWidgets.QComboBox()
        for k, name in METRICS:
            self.cb_metric.addItem(name, k)
        self.cb_metric.currentIndexChanged.connect(lambda _: self._show_outcomes())
        bar.addWidget(self.cb_metric)
        bar.addWidget(QtWidgets.QLabel("against input"))
        self.cb_input = QtWidgets.QComboBox()
        for k in UNCERTAIN:
            self.cb_input.addItem(k, k)
        self.cb_input.currentIndexChanged.connect(lambda _: self._show_outcomes())
        bar.addWidget(self.cb_input)
        bar.addStretch(1)
        v.addLayout(bar)
        top = QtWidgets.QHBoxLayout()
        self.pl_prob = pg.PlotWidget(); self.pl_prob.getPlotItem().setMenuEnabled(False)
        self.pl_hist = pg.PlotWidget(); self.pl_hist.getPlotItem().setMenuEnabled(False)
        top.addWidget(self.pl_prob, 1); top.addWidget(self.pl_hist, 1)
        v.addLayout(top, 1)
        self.pl_scatter = pg.PlotWidget(); self.pl_scatter.getPlotItem().setMenuEnabled(False)
        v.addWidget(self.pl_scatter, 1)
        return w

    def equations(self):
        return ["Each member is an independent realisation of the continuous-time Markov chain of "
                "cell damage, with its own random stream and sampled damage parameters.",
                ("h", "Damage probability of a cell"),
                r"$\hat p_c(t)=\frac{1}{n}\sum_{m=1}^{n}\mathbf{1}[t^{(m)}_{c,s}\leq t]$",
                "the fraction of members in which cell c has reached state s by time t.",
                ("h", "Outcome probability"),
                r"$\hat p=\frac{k}{n},\qquad \frac{\hat p+\frac{z^2}{2n}\pm z\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^2}{4n^2}}}{1+z^2/n}$",
                "Wilson score interval (z = 1.96).",
                ("h", "Event times"),
                r"$\hat F(t)=\frac{1}{n}\sum_{m=1}^{n}\mathbf{1}[t_m\leq t]$",
                ("h", "Sensitivity"),
                r"$\rho_S=\mathrm{corr}\left(\mathrm{rank}(x),\,\mathrm{rank}(y)\right)$",
                "Spearman rank correlation between an uncertain input x and the outcome y."]

    # ------------------------------------------------------------------ run
    def _scenario_changed(self):
        k = self.cb_scn.currentData()
        if k:
            self.lab_desc.setText(SCENARIOS[k].description)
            self.editor.set_scenario(k)

    def run_all(self, auto=False):
        p = self.params.values()
        sc, params = self.editor.scenario()
        nx, nz = self.editor.lattice()
        self._t0 = time.perf_counter()
        self.progress.setValue(0); self.progress.setVisible(True)
        self.res = EnsembleResult(sc.key)
        self._cache = None
        self._n_target = int(p["n"])
        self.status.emit(f"ensemble of {int(p['n'])} × {sc.key} running in {int(p['workers'])} processes …")
        bridge = self.member_signal

        def job(ctx):
            from ..core.ensemble import run_ensemble
            gen = ctx.gen
            return run_ensemble(sc, n=int(p["n"]), params=params, uncertain=p["uncertain"], seed0=int(p["seed"]),
                                workers=int(p["workers"]), nx=nx, nz=nz, progress=lambda i, n: ctx.progress(i, n),
                                cancelled=lambda: ctx.cancelled, on_member=lambda m: bridge.emit(gen, m))

        self.runner.submit(job)

    def cancel(self):
        self.runner.cancel()
        self.progress.setVisible(False)
        self.status.emit("cancelled")

    def set_live(self, on):
        pass

    def _member(self, gen, m):
        if not self.runner.is_current(gen) or self.res is None:
            return
        self.res.members.append(m)
        self._cache = None
        self.status.emit(f"ensemble: {self.res.n} of {self._n_target} members done")
        self.refresh()

    def _done(self, gen, res):
        if not self.runner.is_current(gen) or res is None:
            return
        self.res = res
        self._cache = None
        self.progress.setVisible(False)
        # scatter against the most influential input by default
        if res.n >= 4:
            y = np.array([_metric(m, self.cb_metric.currentData()) for m in res.members], float)
            r = [(abs(spearman(np.array([m["overrides"].get(k, np.nan) for m in res.members], float), y)), k)
                 for k in UNCERTAIN]
            r = [x for x in r if np.isfinite(x[0])]
            if r:
                self.cb_input.blockSignals(True)
                self.cb_input.setCurrentIndex(self.cb_input.findData(max(r)[1]))
                self.cb_input.blockSignals(False)
        self.refresh()
        self.status.emit(f"ensemble finished: {res.n} members in {time.perf_counter() - self._t0:.1f} s")

    def _progress(self, gen, i, n):
        if self.runner.is_current(gen):
            self.progress.setValue(int(100 * i / max(n, 1)))

    def _failed(self, gen, msg):
        self.last_error = msg
        self.progress.setVisible(False)
        self.status.emit("error: " + msg.splitlines()[0])
        print(msg)

    # ------------------------------------------------------------------ derived data
    def data(self):
        """Common time grid and the members' series on it (cached until a member is added)."""
        if self._cache is not None:
            return self._cache
        ms = self.res.members
        t_end = max(float(np.asarray(m["t"])[-1]) for m in ms)
        ts, unit = (3600.0, "h") if t_end > 2 * 3600 else ((60.0, "min") if t_end > 600 else (1.0, "s"))
        tg = np.linspace(0.0, t_end, 300)
        series = {}
        for k, *_ in SERIES:
            rows = []
            for m in ms:
                s = m.get("series", m)
                if k in s:
                    tm = np.asarray(m["t"], float)
                    y = np.interp(tg, tm, np.asarray(s[k], float))     # a member that stopped holds its last value
                    rows.append(y)
            if rows:
                series[k] = np.array(rows)
        self._cache = dict(t=tg, ts=ts, unit=unit, series=series, t_end=t_end)
        return self._cache

    def geom(self):
        m = self.res.members[0]
        rt = REACTORS[SCENARIOS[self.res.scenario].reactor]
        return (int(m["nx"]), int(m["nz"]), rt.width_m, rt.height_m)

    # ------------------------------------------------------------------ display
    def refresh(self):
        res = self.res
        if res is None or not res.members:
            return
        try:
            i = self.tabs.currentIndex()
            [self._show_maps, self._show_fans, self._show_events, self._show_outcomes][i]()
            self._show_tables()
        except Exception as e:
            import traceback
            self.last_error = traceback.format_exc()
            traceback.print_exc()
            self.status.emit(f"display error: {e}")

    def _t_now(self):
        d = self.data()
        return d["t_end"] * self.slider.value() / max(self.slider.maximum(), 1)

    def _show_maps(self):
        if self.res is None or not self.res.members or "cell_times" not in self.res.members[0]:
            return
        d = self.data()
        g = self.geom()
        s = self.cb_state.currentData()
        t = self._t_now()
        ms = [m for m in self.res.members if "cell_times" in m]
        ct = np.stack([np.asarray(m["cell_times"][s], float) for m in ms])          # (n, N)
        fuel = np.asarray(ms[0]["fuel_mask"], bool)
        P = np.where(fuel, (ct <= t).mean(0) * 100, np.nan)
        reached = np.isfinite(ct)
        import warnings
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            tmed = np.where(fuel & (reached.sum(0) > 0),
                            np.nanmedian(np.where(reached, ct, np.nan), axis=0) / d["ts"], np.nan)
        Tpk = np.where(fuel, np.nanmean(np.stack([m["Tf_peak"] for m in ms]), 0), np.nan)
        st = STATES[self.cb_state.currentIndex()][1]
        if self.cb_map2.currentData() == "tmed":
            self.map_t.set_field(Field("tmed", f"median time to {st}", d["unit"], "viridis_r", digits=3))
            self.map_t.show(tmed, g)
            t2 = f"median time to {st} ({d['unit']})"
        else:
            self.map_t.set_field(Field("Tpk", "mean peak fuel temperature", "K", "inferno"))
            self.map_t.show(Tpk, g)
            t2 = "mean peak fuel temperature over the run (K)"
        self.map_p.show(P, g, (0.0, 100.0))
        W.style(self.map_p.pi, f"probability that a cell is {st} at t (%)")
        W.style(self.map_t.pi, t2)
        self.lab_time.setText(f"t = {t / d['ts']:.2f} {d['unit']}")
        self._maps = dict(P=P, tmed=tmed, Tpk=Tpk)
        # expected state fractions
        pi = self.pl_stack.getPlotItem(); pi.clear(); pi.legend.clear()
        W.style(pi, "expected core state (mean over members)", f"time ({d['unit']})", "fraction of fuel cells")
        tg = d["t"]
        allct = np.stack([np.asarray(m["cell_times"], float)[:, fuel] for m in ms])   # (n, 5, nF)
        frac_ge = np.array([[(allct[:, k, :] <= tt).mean() for tt in tg] for k in range(5)])
        x = tg / d["ts"]
        prev = pg.PlotCurveItem(x, np.zeros_like(x), pen=pg.mkPen(None)); pi.addItem(prev)
        base = np.zeros_like(x)
        for k in range(5):
            share = frac_ge[k] - (frac_ge[k + 1] if k < 4 else 0.0)
            top = base + share
            c1 = pg.PlotCurveItem(x, top, pen=pg.mkPen(theme.STATE_COLORS[k], width=1))
            pi.addItem(c1)
            pi.addItem(pg.FillBetweenItem(prev, c1, brush=pg.mkBrush(theme.STATE_COLORS[k])))
            pi.legend.addItem(pg.ScatterPlotItem([], [], symbol="s", size=10, brush=theme.STATE_COLORS[k], pen=None),
                              STATE_NAMES[k])
            prev, base = c1, top
        pi.setYRange(0, 1, padding=0)
        pi.addItem(pg.InfiniteLine(t / d["ts"], angle=90, pen=pg.mkPen(theme.current["accent"], width=2)))

    def _hover(self, idx):
        for m in (self.map_p, self.map_t):
            m.highlight(idx, show_tip=(m is self.sender()))
        if idx < 0 or not hasattr(self, "_maps"):
            return
        M = self._maps
        nx = self.geom()[0]
        st = STATES[self.cb_state.currentIndex()][1]
        self.lab_cell.setText(f"cell ({idx % nx + 1}, {idx // nx + 1}):  P({st}) = {M['P'][idx]:.0f} %,  "
                              f"median time {fmt(self.map_t.f, M['tmed'][idx])},  "
                              f"mean peak fuel {M['Tpk'][idx]:.0f} K")

    def _play(self, on):
        if on:
            if self.slider.value() >= self.slider.maximum():
                self.slider.setValue(0)
            self._timer.start()
            self.btn_play.setText("❚❚")
        else:
            self._timer.stop()
            self.btn_play.setText("▶")

    def _tick(self):
        if self.slider.value() >= self.slider.maximum():
            self.btn_play.setChecked(False)
            return
        self.slider.setValue(self.slider.value() + 1)

    def _show_fans(self):
        d = self.data()
        k = self.cb_series.currentData()
        name, unit = next((n, u) for kk, n, u, _ in SERIES if kk == k)
        x = d["t"] / d["ts"]
        for pw, key, nm, un in ((self.pl_fan, k, name, unit), (self.pl_fan2, "H2" if k != "H2" else "frac_intact",
                                                            None, None)):
            pi = pw.getPlotItem(); pi.clear(); pi.legend.clear()
            if nm is None:
                nm, un = next((n, u) for kk, n, u, _ in SERIES if kk == key)
            W.style(pi, f"{nm} — median and percentile bands over {self.res.n} members", f"time ({d['unit']})", un)
            Y = d["series"].get(key)
            if Y is None:
                continue
            logy = self.chk_logy.isChecked() and pw is self.pl_fan
            pi.setLogMode(y=logy)
            Yp = np.where(Y > 0, Y, np.nan) if logy else Y
            c = theme.color(0 if pw is self.pl_fan else "h2")
            qc = QtGui.QColor(c)
            if self.chk_members.isChecked():
                qm = QtGui.QColor(c); qm.setAlpha(45)
                for i, y in enumerate(Yp):
                    pi.plot(x, y, pen=pg.mkPen(qm, width=1), name="members" if i == 0 else None)
            import warnings
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                q = np.nanpercentile(Yp, [5, 25, 50, 75, 95], axis=0)
            if logy:
                q = np.log10(q)
            for (lo, hi), a, lab in (((0, 4), 50, "5–95 %"), ((1, 3), 100, "25–75 %")):
                c1 = pg.PlotCurveItem(x, q[lo], pen=pg.mkPen(None)); c2 = pg.PlotCurveItem(x, q[hi], pen=pg.mkPen(None))
                qa = QtGui.QColor(qc); qa.setAlpha(a)
                fb = pg.FillBetweenItem(c1, c2, brush=pg.mkBrush(qa))
                pi.addItem(fb)
                pi.legend.addItem(pg.ScatterPlotItem([], [], symbol="s", size=10, brush=qa, pen=None), lab)
            med = pg.PlotCurveItem(x, q[2], pen=pg.mkPen(qc, width=2.5))
            pi.addItem(med)
            pi.legend.addItem(med, "median")

    def _show_events(self):
        res = self.res
        d = self.data()
        pi = self.pl_strip.getPlotItem(); pi.clear()
        W.style(pi, f"time of each event in each member ({res.n} members; dots: members, box: quartiles, "
                    f"bar: median)", f"time ({d['unit']})", None)
        rng = np.random.default_rng(0)
        ticks = []
        for i, ev in enumerate(EVENTS):
            y0 = len(EVENTS) - 1 - i
            tt = res.times(ev) / d["ts"]
            p, lo, hi = res.probability(ev)
            ticks.append((y0, f"{EVENT_LABELS[ev]}  ({p * 100:.0f} %)"))
            if not len(tt):
                continue
            c = theme.color(i)
            q1, q2, q3 = np.percentile(tt, [25, 50, 75])
            box = QtWidgets.QGraphicsRectItem(q1, y0 - 0.28, max(q3 - q1, 1e-9), 0.56)
            qa = QtGui.QColor(c); qa.setAlpha(60)
            box.setBrush(pg.mkBrush(qa)); box.setPen(pg.mkPen(c, width=1.2))
            pi.addItem(box)
            pi.plot([q2, q2], [y0 - 0.32, y0 + 0.32], pen=pg.mkPen(c, width=3))
            jit = rng.uniform(-0.22, 0.22, len(tt))
            pi.addItem(pg.ScatterPlotItem(tt, y0 + jit, size=7, brush=c, pen=pg.mkPen(theme.current["plot_bg"], width=0.8),
                                          hoverable=True, tip=lambda x, y, data: f"{x:.3g} {d['unit']}"))
        pi.getAxis("left").setTicks([ticks])
        pi.getAxis("left").setWidth(240)
        pi.setYRange(-0.6, len(EVENTS) - 0.4, padding=0)
        pi = self.pl_cdf.getPlotItem(); pi.clear(); pi.legend.clear()
        W.style(pi, "cumulative probability (empirical CDF; ends below 1 if not every member reaches the event)",
                f"time ({d['unit']})", "probability")
        pi.getAxis("left").setWidth(240)
        for i, ev in enumerate(EVENTS):
            tt = res.times(ev) / d["ts"]
            if len(tt):
                x = np.r_[0.0, np.repeat(tt, 2), d["t_end"] / d["ts"]]
                y = np.r_[0.0, 0.0, np.repeat(np.arange(1, len(tt) + 1) / res.n, 2)]
                W.curve(pi, x, y, i, name=EVENT_LABELS[ev], width=2)
        pi.setYRange(0, 1.02)

    def _show_outcomes(self):
        res = self.res
        mk = self.cb_metric.currentData()
        mname = self.cb_metric.currentText()
        # probabilities with Wilson intervals
        pi = self.pl_prob.getPlotItem(); pi.clear()
        W.style(pi, f"probability of each outcome ({res.n} members)", "probability (bar: 95 % interval)", None)
        ticks = []
        for i, ev in enumerate(EVENTS):
            y0 = len(EVENTS) - 1 - i
            p, lo, hi = res.probability(ev)
            ticks.append((y0, EVENT_LABELS[ev]))
            pi.addItem(pg.BarGraphItem(x0=[0], y=[y0], height=0.6, width=[p], brush=theme.color(i), pen=None))
            pi.addItem(pg.ErrorBarItem(x=np.array([p]), y=np.array([y0]), left=np.array([p - lo]),
                                       right=np.array([hi - p]), beam=0.25, pen=pg.mkPen(theme.current["plot_fg"], width=1.5)))
            t = pg.TextItem(f"{p * 100:.0f} %", color=theme.current["plot_fg"], anchor=(0, 0.5))
            t.setPos(min(hi, 1.0) + 0.02, y0)
            pi.addItem(t)
        pi.getAxis("left").setTicks([ticks])
        pi.setXRange(0, 1.28, padding=0); pi.setYRange(-0.6, len(EVENTS) - 0.4, padding=0)
        # histogram of the selected outcome
        y = np.array([_metric(m, mk) for m in res.members], float)
        fin = y[np.isfinite(y)]
        pi = self.pl_hist.getPlotItem(); pi.clear()
        W.style(pi, f"distribution over members ({len(fin)} of {res.n} with a value)", mname, "members")
        if fin.size:
            if np.ptp(fin) > 0:
                cnt, edges = np.histogram(fin, bins=min(30, max(5, fin.size // 2)))
            else:
                cnt, edges = np.array([fin.size]), np.array([fin[0] - 0.5, fin[0] + 0.5])
            pi.addItem(pg.BarGraphItem(x0=edges[:-1], x1=edges[1:], height=cnt, brush=theme.color(0),
                                       pen=pg.mkPen(theme.current["plot_bg"])))
            med = float(np.median(fin))
            W.vline(pi, med, "fg", label=f"median {med:.3g}", pos=0.9)
        # scatter against the chosen input
        ik = self.cb_input.currentData()
        x = np.array([m["overrides"].get(ik, np.nan) for m in res.members], float)
        pi = self.pl_scatter.getPlotItem(); pi.clear()
        rho = spearman(x, y)
        W.style(pi, f"{mname} against {ik}  (Spearman ρ = {rho:+.2f})" if np.isfinite(rho)
                else f"{mname} against {ik}", ik, mname)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.any():
            pi.addItem(pg.ScatterPlotItem(x[ok], y[ok], size=9, brush=theme.color(0),
                                          pen=pg.mkPen(theme.current["plot_bg"], width=0.8), hoverable=True,
                                          data=[f"seed {m['seed']}" for m, k in zip(res.members, ok) if k],
                                          tip=lambda x, y, data: f"{data}: {ik} = {x:.3g}, {y:.3g}"))
        self._sens = [(k, spearman(np.array([m["overrides"].get(k, np.nan) for m in res.members], float), y))
                      for k in UNCERTAIN]

    def _show_tables(self):
        res = self.res
        d = self.data()
        rows = []
        for ev in EVENTS:
            p, lo, hi = res.probability(ev)
            t = res.times(ev) / d["ts"]
            med = f"{np.median(t):.2f} {d['unit']}" if len(t) else "—"
            if len(t) > 2:
                med += f" ({np.percentile(t, 5):.2f}–{np.percentile(t, 95):.2f})"
            rows.append((EVENT_LABELS[ev], f"{p:.2f} [{lo:.2f}, {hi:.2f}]", med))
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            for j, c in enumerate(r):
                self.table.setItem(i, j, QtWidgets.QTableWidgetItem(c))
        mk = self.cb_metric.currentData()
        y = np.array([_metric(m, mk) for m in res.members], float)
        sens = [(k, spearman(np.array([m["overrides"].get(k, np.nan) for m in res.members], float), y))
                for k in UNCERTAIN]
        sens.sort(key=lambda r: -abs(r[1]) if np.isfinite(r[1]) else 0)
        self.sens.setRowCount(len(sens))
        for i, (k, r) in enumerate(sens):
            self.sens.setItem(i, 0, QtWidgets.QTableWidgetItem(k))
            self.sens.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{r:+.2f}" if np.isfinite(r) else "—"))
            bar = "█" * int(round(abs(r) * 12)) if np.isfinite(r) else ""
            it = QtWidgets.QTableWidgetItem(bar)
            it.setForeground(QtGui.QColor(theme.color(3 if np.isfinite(r) and r < 0 else 0)))
            self.sens.setItem(i, 2, it)

    # ------------------------------------------------------------------ misc
    def invalidate(self):
        for m in (self.map_p, self.map_t):
            m.restyle()
        self.eq.set_html(self.equations())
        self.refresh()

    def current_view(self):
        return self.tabs.currentWidget()

    def export_png(self, path):
        W.grab_png(self.current_view(), path)

    def export_data(self, path):
        import csv
        if self.res is None:
            return
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["seed", "peak_clad_T", "H2_kg", "molten_max", "relocated", "release"] + list(EVENTS)
                       + list(UNCERTAIN))
            for m in self.res.members:
                w.writerow([m["seed"], m["peak_clad_T"], m["H2_kg"], m["molten_max"], m["relocated"], m["release"]]
                           + [m["times"].get(e, "") for e in EVENTS] + [m["overrides"].get(k, "") for k in UNCERTAIN])

    def busy(self):
        return self.runner.running()

    def settle(self, ms=600_000):
        app = QtWidgets.QApplication.instance()
        t = time.time()
        while time.time() - t < ms / 1000:
            app.processEvents()
            if not (self.runner.running() or self.runner.threads):
                break
            time.sleep(0.005)
        for _ in range(5):
            app.processEvents()
