"""Core accident simulator: pick a design and scenario, edit it, run the coupled lattice model in
a background thread and watch the core evolve live. The lattice maps show physical fields with
units; hovering reads every quantity of a cell, clicking pins cells whose history is plotted; the
timeline below shows power, the damage stack and the events, and its cursor scrubs time."""
from __future__ import annotations

import csv
import time

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt

from . import theme
from . import widgets as W
from .fields import BY_KEY, STATE_NAMES
from .lattice import PIN_COLORS, CellInspector, MapPanel, Timeline
from .scenario_editor import ScenarioEditor
from ..core.reactors import REACTORS
from ..core.scenarios import SCENARIOS

DEFAULT_FIELDS = ("q3", "T_fuel", "state", "T_clad")
LAYOUTS = [("2 × 2 maps", 4), ("2 maps", 2), ("1 large map", 1)]
SPEEDS = [("¼×", 0.25), ("½×", 0.5), ("1×", 1), ("2×", 2), ("4×", 4), ("8×", 8)]


def _time_scale(t_end):
    if t_end > 6 * 3600:
        return 3600.0, "h"
    if t_end > 600:
        return 60.0, "min"
    return 1.0, "s"


def _fmt_t(t):
    if t < 120:
        return f"{t:.2f} s" if t < 10 else f"{t:.1f} s"
    if t < 2 * 3600:
        return f"{t / 60:.1f} min"
    return f"{t / 3600:.2f} h"


class SimPage(QtWidgets.QWidget):
    status = QtCore.pyqtSignal(str)
    title = "Core accident"
    MIN_SIZE = (1500, 820)
    live_signal = QtCore.pyqtSignal(int, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("page")
        self.res = None
        self.last_error = None
        self.live = True
        self.pins = []              # [(cell index, colour)]
        self.hover = -1
        self._range_cache = {}
        self.runner = W.Runner(self)
        self.heavy_runner = self.runner
        self.runner.done.connect(self._done)
        self.runner.failed.connect(self._failed)
        self.runner.progressed.connect(self._progress)
        self.live_signal.connect(self._live_update)
        self._t0 = 0.0
        self._follow = True
        self._frac = 0.0
        self._build()
        self.cb_reactor.setCurrentIndex(0)
        self._reactor_changed()

    # ------------------------------------------------------------------ layout
    def _build(self):
        h = QtWidgets.QHBoxLayout(self); h.setContentsMargins(10, 10, 10, 10); h.setSpacing(10)
        self.split = QtWidgets.QSplitter(Qt.Horizontal); self.split.setChildrenCollapsible(True)
        self.split.setHandleWidth(8)
        h.addWidget(self.split)
        left = QtWidgets.QScrollArea(); left.setWidgetResizable(True); left.setMinimumWidth(300)
        left.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        lw = QtWidgets.QWidget(); lv = QtWidgets.QVBoxLayout(lw)
        lv.setContentsMargins(0, 0, 4, 0); lv.setSpacing(10)
        f, v = W.card("Scenario")
        self.cb_reactor = QtWidgets.QComboBox()
        self.cb_reactor.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cb_reactor.setMinimumContentsLength(12)
        self.cb_reactor.addItem("All designs", None)
        for k, rt in REACTORS.items():
            self.cb_reactor.addItem(rt.name, k)
        self.cb_reactor.currentIndexChanged.connect(self._reactor_changed)
        v.addWidget(self.cb_reactor)
        self.cb_scn = QtWidgets.QComboBox()
        self.cb_scn.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cb_scn.setMinimumContentsLength(12)
        self.cb_scn.currentIndexChanged.connect(self._scenario_changed)
        v.addWidget(self.cb_scn)
        self.lab_desc = W.muted("")
        v.addWidget(self.lab_desc)
        self.opts = W.ParamPanel([W.P("seed", "random seed", 0, 9999, 0, integer=True,
                                      tip="seed of the stochastic damage chain")],
                                 [W.C("method", "damage chain", [("Gillespie (exact)", "gillespie"),
                                                                 ("τ-leaping", "tau-leap")], "gillespie")])
        v.addWidget(self.opts)
        row = QtWidgets.QHBoxLayout()
        self.btn_run = QtWidgets.QPushButton("Run"); self.btn_run.setObjectName("primary")
        self.btn_run.clicked.connect(lambda: self.run_all())
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.cancel)
        row.addWidget(self.btn_run, 1); row.addWidget(self.btn_cancel)
        v.addLayout(row)
        self.progress = QtWidgets.QProgressBar(); self.progress.setRange(0, 1000)
        self.progress.setVisible(False)
        v.addWidget(self.progress)
        lv.addWidget(f)
        f, v = W.card("Edit scenario", "Changes apply to the next run; Reset restores the built-in scenario.")
        self.editor = ScenarioEditor()
        self.editor.changed.connect(self._edited)
        v.addWidget(self.editor)
        self.lab_mod = W.muted("")
        v.addWidget(self.lab_mod)
        lv.addWidget(f)
        f, v = W.card("Design")
        self.lab_design = W.muted("")
        v.addWidget(self.lab_design)
        lv.addWidget(f)
        lv.addStretch(1)
        left.setWidget(lw)
        self.split.addWidget(left)

        # ---- centre: tabs
        self.tabs = QtWidgets.QTabWidget()
        lat = QtWidgets.QWidget(); lg = QtWidgets.QVBoxLayout(lat); lg.setContentsMargins(6, 6, 6, 6)
        lg.setSpacing(6)
        bar = QtWidgets.QHBoxLayout(); bar.setSpacing(10)
        self.cb_layout = QtWidgets.QComboBox()
        for name, n in LAYOUTS:
            self.cb_layout.addItem(name, n)
        self.cb_layout.currentIndexChanged.connect(self._layout)
        self.cb_range = QtWidgets.QComboBox()
        self.cb_range.addItem("colour range: this frame", "frame")
        self.cb_range.addItem("colour range: whole run", "run")
        self.cb_range.setToolTip("Whole run: the colour scale is fixed over all frames, so frames can be compared")
        self.cb_range.setCurrentIndex(0)
        self.cb_range.currentIndexChanged.connect(lambda _: self._show_maps())
        self.chk_smooth = QtWidgets.QCheckBox("smooth shading")
        self.chk_smooth.toggled.connect(self._smooth)
        for w in (self.cb_layout, self.cb_range, self.chk_smooth):
            bar.addWidget(w)
        bar.addStretch(1)
        self.lab_hint = W.muted("hover: read a cell · click: pin it")
        self.lab_hint.setWordWrap(False)
        bar.addWidget(self.lab_hint)
        lg.addLayout(bar)
        self.grid = QtWidgets.QGridLayout(); self.grid.setSpacing(8)
        self.panels = []
        for i, fld in enumerate(DEFAULT_FIELDS):
            m = MapPanel(fld)
            m.changed.connect(self._show_maps)
            m.view.hovered.connect(lambda idx, j=i: self._hovered(j, idx))
            m.view.clicked.connect(self._pin)
            self.grid.addWidget(m, i // 2, i % 2)
            self.panels.append(m)
        lg.addLayout(self.grid, 1)
        trow = QtWidgets.QHBoxLayout(); trow.setSpacing(4)
        self.btn_start = QtWidgets.QToolButton(); self.btn_start.setText("|◀")
        self.btn_back = QtWidgets.QToolButton(); self.btn_back.setText("◀")
        self.btn_play = QtWidgets.QToolButton(); self.btn_play.setText("▶"); self.btn_play.setCheckable(True)
        self.btn_fwd = QtWidgets.QToolButton(); self.btn_fwd.setText("▶|")
        for b, tip in ((self.btn_start, "first frame"), (self.btn_back, "previous frame"),
                       (self.btn_play, "play / pause"), (self.btn_fwd, "next frame")):
            b.setToolTip(tip); b.setFixedWidth(40); trow.addWidget(b)
        self.btn_start.clicked.connect(lambda: self.slider.setValue(0))
        self.btn_back.clicked.connect(lambda: self.slider.setValue(self.slider.value() - 1))
        self.btn_fwd.clicked.connect(lambda: self.slider.setValue(self.slider.value() + 1))
        self.btn_play.toggled.connect(self._play)
        self.slider = QtWidgets.QSlider(Qt.Horizontal); self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self._scrub)
        trow.addWidget(self.slider, 1)
        self.cb_speed = QtWidgets.QComboBox()
        for n, s in SPEEDS:
            self.cb_speed.addItem(n, s)
        self.cb_speed.setCurrentIndex(2)
        self.cb_speed.setToolTip("playback speed")
        trow.addWidget(self.cb_speed)
        self.chk_loop = QtWidgets.QCheckBox("loop")
        trow.addWidget(self.chk_loop)
        self.lab_time = QtWidgets.QLabel("t = 0"); self.lab_time.setObjectName("readout")
        self.lab_time.setMinimumWidth(120)
        trow.addWidget(self.lab_time)
        lg.addLayout(trow)
        self.timeline = Timeline()
        self.timeline.seek.connect(self._seek)
        lg.addWidget(self.timeline)
        self.tabs.addTab(lat, "Core lattice")
        self._timer = QtCore.QTimer(self); self._timer.setInterval(60)
        self._timer.timeout.connect(self._tick)

        plant_w = QtWidgets.QWidget(); pv = QtWidgets.QVBoxLayout(plant_w); pv.setContentsMargins(6, 6, 6, 6)
        prow = QtWidgets.QHBoxLayout()
        self.chk_logt = QtWidgets.QCheckBox("log time axis")
        self.chk_logp = QtWidgets.QCheckBox("log power axis"); self.chk_logp.setChecked(True)
        self.chk_cursor = QtWidgets.QCheckBox("show lattice time"); self.chk_cursor.setChecked(True)
        for w in (self.chk_logt, self.chk_logp, self.chk_cursor):
            w.toggled.connect(lambda _: self._show_plant())
            prow.addWidget(w)
        prow.addStretch(1)
        pv.addLayout(prow)
        plant = pg.GraphicsLayoutWidget()
        self.p_power = plant.addPlot(row=0, col=0); self.p_rho = plant.addPlot(row=0, col=1)
        self.p_temp = plant.addPlot(row=1, col=0); self.p_press = plant.addPlot(row=1, col=1)
        self.p_dmg = plant.addPlot(row=2, col=0); self.p_h2 = plant.addPlot(row=2, col=1)
        self.plant_plots = [self.p_power, self.p_rho, self.p_temp, self.p_press, self.p_dmg, self.p_h2]
        for p in self.plant_plots[1:]:
            p.setXLink(self.p_power)
        plant.ci.layout.setHorizontalSpacing(14)
        plant.ci.layout.setColumnStretchFactor(0, 1)
        plant.ci.layout.setColumnStretchFactor(1, 1)
        plant.ci.layout.setVerticalSpacing(4)
        for i, p in enumerate(self.plant_plots):
            W.nice_axes(p, ("left", "bottom", "right"))
            W.top_legend(p, 3 if p is not self.p_rho else 4)
            p.showAxis("right")
            p.getAxis("left").setWidth(W.LEFT_W)
            p.getAxis("right").setWidth(W.LEFT_W if i % 2 else 10)    # right column: twin axes
            p.getAxis("right").setStyle(showValues=False)
            p.setMenuEnabled(False)
        self.tw_press = W.Twin(self.p_press)
        self.tw_h2 = W.Twin(self.p_h2)
        pv.addWidget(plant, 1)
        self.tabs.addTab(plant_w, "Plant")
        self.plant = plant
        self.ev_table = QtWidgets.QTableWidget(0, 2)
        self.ev_table.setHorizontalHeaderLabels(["time", "event"])
        self.ev_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.ev_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.ev_table.verticalHeader().setVisible(False)
        self.ev_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.ev_table.cellDoubleClicked.connect(self._event_jump)
        self.ev_table.setToolTip("double-click an event to show the core at that time")
        self.tabs.addTab(self.ev_table, "Event log")
        self.tabs.currentChanged.connect(lambda _: self.refresh())
        self.split.addWidget(self.tabs)

        right = QtWidgets.QScrollArea(); right.setWidgetResizable(True); right.setMinimumWidth(300)
        right.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        rw = QtWidgets.QWidget(); rv = QtWidgets.QVBoxLayout(rw)
        rv.setContentsMargins(0, 0, 4, 0); rv.setSpacing(10)
        f, v = W.card("Cell inspector")
        self.inspector = CellInspector()
        self.inspector.unpin_all.connect(self._unpin_all)
        self.inspector.box.currentIndexChanged.connect(lambda _: self._show_history())
        v.addWidget(self.inspector)
        rv.addWidget(f)
        f, v = W.card("Results")
        self.table = W.ResultTable(); self.table.setMinimumHeight(300)
        v.addWidget(self.table)
        rv.addWidget(f)
        f, v = W.card("Model")
        self.eq = W.EquationView(self.equations())
        v.addWidget(self.eq)
        rv.addWidget(f)
        rv.addStretch(1)
        right.setWidget(rw)
        self.split.addWidget(right)
        self.split.setStretchFactor(0, 0); self.split.setStretchFactor(1, 1); self.split.setStretchFactor(2, 0)
        self.split.setSizes([350, 1100, 400])
        self.restyle()

    # ------------------------------------------------------------------ content
    def equations(self):
        return [("h", "Neutronics"),
                r"$-\nabla\cdot D\nabla\phi+\Sigma_a\phi=\frac{1}{k}\nu\Sigma_f\phi$",
                "One-group diffusion on the x–z lattice gives the flux shape; reactivity changes "
                "are first-order perturbations on that shape:",
                r"$\delta\rho=\frac{\langle\phi^2\,(\delta\nu\Sigma_f/k-\delta\Sigma_a)\rangle}{\langle\phi^2\,\nu\Sigma_f\rangle}$",
                r"$\frac{dn}{dt}=\frac{\rho-\beta}{\Lambda}n+\sum_{i=1}^{6}\lambda_iC_i,\quad\frac{dC_i}{dt}=\frac{\beta_i}{\Lambda}n-\lambda_iC_i$",
                "Power density of a cell is its fission plus decay power over the core volume it "
                "represents; the linear heat rate is per metre of fuel rod:",
                r"$q'''=\frac{P_{cell}}{V_{cell}},\qquad q'=\frac{P_{cell}}{L_{rod}}$",
                ("h", "Poisons (per cell)"),
                r"$\dot I=\gamma_I\Sigma_f\phi-\lambda_II,\quad\dot X=\gamma_X\Sigma_f\phi+\lambda_II-(\lambda_X+\sigma_X\phi)X$",
                ("h", "Damage chain (continuous-time Markov)"),
                "Each fuel cell moves intact → failed → damaged → molten → relocated with "
                "temperature-, oxidation- and enthalpy-dependent hazards λ(T) (the 'damage transition "
                "rate' map); the next event is drawn exactly (Gillespie):",
                r"$\tau=-\frac{\ln u_1}{\Lambda_{tot}},\quad\Lambda_{tot}=\sum_j\lambda_j,\quad P(j)=\frac{\lambda_j}{\Lambda_{tot}}$",
                ("h", "Oxidation and decay heat"),
                r"$\frac{d(w^2)}{dt}=A\,e^{-Q/RT}$",
                "Cathcart–Pawel / Baker–Just Zr–steam kinetics (steam limited), ANS-5.1 decay heat."]

    def _reactor_changed(self):
        key = self.cb_reactor.currentData()
        self.cb_scn.blockSignals(True)
        self.cb_scn.clear()
        for k, s in SCENARIOS.items():
            if key is None or s.reactor == key:
                self.cb_scn.addItem(f"{REACTORS[s.reactor].key.upper()} · {s.name}", k)
        self.cb_scn.blockSignals(False)
        self.cb_scn.setCurrentIndex(0)
        self._scenario_changed()

    def scenario(self):
        return SCENARIOS[self.cb_scn.currentData()]

    @property
    def params(self):
        return self.editor.params

    def _scenario_changed(self):
        if self.cb_scn.currentData() is None:
            return
        sc = self.scenario()
        rt = REACTORS[sc.reactor]
        self.lab_desc.setText(sc.description)
        self.lab_design.setText(f"<b>{rt.name}</b><br>{rt.description}")
        self.lab_design.setTextFormat(Qt.RichText)
        self.editor.set_scenario(sc.key)

    def _edited(self):
        d = self.editor.summary()
        self.lab_mod.setText(f"<b>Modified:</b> {d}" if d else "")
        self.lab_mod.setTextFormat(Qt.RichText)

    def values(self):
        sc, params = self.editor.scenario()
        o = self.opts.values()
        return dict(params=params, method=o["method"], seed=int(o["seed"]), scenario=sc,
                    lattice=self.editor.lattice())

    # ------------------------------------------------------------------ running
    def run_all(self, auto=False):
        vals = self.values()
        key = vals["scenario"].key
        self._t0 = time.perf_counter()
        self.progress.setValue(0); self.progress.setVisible(True)
        self._follow = True
        self.res = None
        self._range_cache.clear()
        self.status.emit(f"running {key} …")
        bridge = self.live_signal
        nx, nz = vals["lattice"]

        def job(ctx):
            from ..core.simulator import Simulator
            sim = Simulator(vals["scenario"], params=vals["params"], method=vals["method"], seed=vals["seed"],
                            nx=nx, nz=nz)
            gen = ctx.gen
            return sim.run(progress=lambda f: ctx.progress(int(f * 1000), 1000),
                           live=lambda part: (not ctx.cancelled) and bridge.emit(gen, part))

        self.runner.submit(job)

    def cancel(self):
        self.runner.cancel()
        self.progress.setVisible(False)
        self.status.emit("cancelled")

    def set_live(self, on):
        self.live = on

    def _live_update(self, gen, part):
        if not self.runner.is_current(gen) or not self.live:
            return
        if self.res is None or self.res.nx * self.res.nz != part.nx * part.nz:
            self.pins = [(i, c) for i, c in self.pins if i < part.nx * part.nz]
        self.res = part
        self.refresh()
        self.status.emit(f"{part.scenario}: t = {_fmt_t(part.t[-1])}")

    def _done(self, gen, res):
        if not self.runner.is_current(gen) or res is None:
            return
        self.res = res
        self.pins = [(i, c) for i, c in self.pins if i < res.nx * res.nz]
        self.progress.setVisible(False)
        self._follow = True
        self.refresh()
        self.status.emit(f"{res.scenario}: finished in {time.perf_counter() - self._t0:.1f} s "
                         f"({res.summary['steps']} steps)")

    def _progress(self, gen, i, n):
        if self.runner.is_current(gen):
            self.progress.setValue(int(1000 * i / max(n, 1)))

    def _failed(self, gen, msg):
        self.last_error = msg
        self.progress.setVisible(False)
        self.status.emit("error: " + msg.splitlines()[0])
        print(msg)

    # ------------------------------------------------------------------ display
    def geom(self):
        rt = REACTORS[self.res.reactor]
        return (self.res.nx, self.res.nz, rt.width_m, rt.height_m)

    def frame(self):
        return min(self.slider.value(), len(self.res.snapshots) - 1)

    def refresh(self):
        if self.res is None:
            return
        try:
            n = len(self.res.snapshots)
            self.slider.blockSignals(True)
            self.slider.setRange(0, max(n - 1, 0))
            if self._follow:
                self.slider.setValue(n - 1)
            self.slider.blockSignals(False)
            i = self.tabs.currentIndex()
            if i == 0:
                ts, unit = _time_scale(max(float(self.res.t[-1]), 1.0))
                self.timeline.set_result(self.res, ts, unit)
                self._show_maps()
                self._show_history()
            elif i == 1:
                self._show_plant()
            else:
                self._show_events()
            self.table.set_rows(self.summary(self.res))
        except Exception as e:
            import traceback
            self.last_error = traceback.format_exc()
            traceback.print_exc()
            self.status.emit(f"display error: {e}")

    def _run_levels(self, view):
        """Colour range of a field over every stored frame (displayed units)."""
        res = self.res
        key = (id(res), len(res.snapshots), view.field, view.log)
        lv = self._range_cache.get(key)
        if lv is None:
            stack = np.stack([s.fields[view.field] for s in res.snapshots])
            v = view.display(stack)
            fin = v[np.isfinite(v)]
            lv = (float(fin.min()), float(fin.max())) if fin.size else (0.0, 1.0)
            self._range_cache[key] = lv
        return lv

    def _show_maps(self):
        res = self.res
        if res is None or not res.snapshots:
            return
        k = self.frame()
        snap = res.snapshots[k]
        g = self.geom()
        run = self.cb_range.currentData() == "run"
        for m in self.panels:
            if m.isHidden():
                continue
            v = m.view
            lv = self._run_levels(v) if run and v.field != "state" else None
            v.show(snap.fields[v.field], g, lv)
            v.set_pins(self.pins)
            v.highlight(self.hover)
        self.lab_time.setText(f"t = {_fmt_t(snap.t)}")
        self.timeline.set_cursor(snap.t)
        self._show_cell()
        if self.inspector.cursor is not None:
            ts, _ = _time_scale(max(float(res.t[-1]), 1.0))
            self.inspector.cursor.setValue(snap.t / ts)

    def _show_cell(self):
        if self.res is None:
            return
        idx = self.hover if self.hover >= 0 else (self.pins[-1][0] if self.pins else -1)
        no = next((n + 1 for n, (i, _) in enumerate(self.pins) if i == idx), None)
        self.inspector.show_cell(self.res, self.frame(), idx, self.geom(), no)

    def _show_history(self):
        if self.res is None:
            return
        ts, unit = _time_scale(max(float(self.res.t[-1]), 1.0))
        t_now = self.res.snapshots[self.frame()].t if self.res.snapshots else 0.0
        self.inspector.show_history(self.res, self.pins, t_now, ts, unit)

    def _hovered(self, j, idx):
        self.hover = idx
        for i, m in enumerate(self.panels):
            m.view.highlight(idx, show_tip=(i == j))
        self._show_cell()

    def _pin(self, idx):
        have = [i for i, _ in self.pins]
        if idx in have:
            self.pins = [(i, c) for i, c in self.pins if i != idx]
        else:
            used = {c for _, c in self.pins}
            c = next((c for c in PIN_COLORS if c not in used), PIN_COLORS[0])
            self.pins = (self.pins + [(idx, c)])[-len(PIN_COLORS):]
        for m in self.panels:
            m.view.set_pins(self.pins)
        self._show_history()
        self._show_cell()

    def _unpin_all(self):
        self.pins = []
        for m in self.panels:
            m.view.set_pins([])
        self._show_history()

    def _layout(self, _=None):
        n = self.cb_layout.currentData()
        for i, m in enumerate(self.panels):
            m.setVisible(i < n)
        for i, m in enumerate(self.panels):
            self.grid.removeWidget(m)
        cols = 1 if n == 1 else 2
        for i, m in enumerate(self.panels[:n]):
            self.grid.addWidget(m, i // cols, i % cols)
        self._show_maps()

    def _smooth(self, on):
        for m in self.panels:
            m.view.smooth = on
        self._show_maps()

    def _scrub(self, v):
        if self.res is None:
            return
        self._follow = v >= len(self.res.snapshots) - 1
        self._show_maps()

    def _seek(self, t):
        if self.res is None or not self.res.snapshots:
            return
        ts = np.array([s.t for s in self.res.snapshots])
        self.slider.setValue(int(np.argmin(np.abs(ts - t))))

    def _event_jump(self, r, _c):
        if self.res is None or r >= len(self.res.events):
            return
        self._seek(self.res.events[r][0])
        self.tabs.setCurrentIndex(0)

    def _play(self, on):
        if on:
            if self.res is not None and self.slider.value() >= self.slider.maximum():
                self.slider.setValue(0)
            self._frac = 0.0
            self._timer.start()
            self.btn_play.setText("❚❚")
        else:
            self._timer.stop()
            self.btn_play.setText("▶")

    def _tick(self):
        if self.slider.value() >= self.slider.maximum():
            if self.chk_loop.isChecked():
                self.slider.setValue(0)
                return
            self.btn_play.setChecked(False)
            return
        self._frac += self.cb_speed.currentData()
        step = int(self._frac)
        if step >= 1:
            self._frac -= step
            self.slider.setValue(min(self.slider.value() + step, self.slider.maximum()))

    def _show_plant(self):
        res = self.res
        if res is None:
            return
        S = res.series
        ts, unit = _time_scale(max(float(res.t[-1]), 1.0))
        logx = self.chk_logt.isChecked()
        t = np.maximum(res.t, 1e-3) if logx else res.t / ts
        for p in self.plant_plots:
            p.clear()
            if p.legend is not None:
                p.legend.clear()
            p.setLogMode(x=logx, y=False)
        xl = "time (s)" if logx else f"time ({unit})"
        rt = REACTORS[res.reactor]

        W.style(self.p_power, "Thermal power", None, "power (MW)")
        self.p_power.setLogMode(x=logx, y=self.chk_logp.isChecked())
        W.curve(self.p_power, t, np.maximum(S["P_total"], 1e-3), "power", name="total")
        W.curve(self.p_power, t, np.maximum(S["P_fission"], 1e-3), 2, name="fission", width=1.2)
        W.curve(self.p_power, t, np.maximum(S["P_decay"], 1e-3), "decay", name="decay heat", style="dash")
        P = np.concatenate([np.asarray(S[k], float) for k in ("P_total", "P_decay")])
        P = P[np.isfinite(P) & (P > 0)]
        if self.chk_logp.isChecked() and P.size:
            hi = float(P.max()); lo = max(float(P.min()), hi * 1e-5, 1e-3)
            self.p_power.setYRange(np.log10(lo), np.log10(hi), padding=0.06)
        else:
            self.p_power.enableAutoRange(y=True)

        W.style(self.p_rho, "Reactivity", None, "reactivity ($)")
        W.curve(self.p_rho, t, S["rho"], "fg", name="net", width=2)
        ys = [np.asarray(S["rho"], float)]
        for i, k in enumerate(["doppler", "moderator", "void", "xenon", "rods", "control"]):
            y = S.get("rho_" + k)
            if y is not None and np.nanmax(np.abs(y - y[0])) > 1e-3:
                W.curve(self.p_rho, t, y - y[0], i, name=f"Δ {k}", width=1.2)
                ys.append(y - y[0])
        # robust range: brief spikes (e.g. a voiding burst) must not flatten everything else
        Y = np.concatenate([np.asarray(v, float)[np.isfinite(v)] for v in ys])
        if Y.size:
            lo, hi = np.percentile(Y, [0.5, 99.5])
            net = np.asarray(S["rho"], float); net = net[np.isfinite(net)]
            if net.size:                              # components far below the net value are clipped
                lo = max(lo, min(-20.0, 1.5 * float(net.min())))
                hi = min(hi, max(5.0, 1.5 * float(net.max())))
            lo, hi = min(lo, 0.0), max(hi, 0.0)
            if hi - lo < 0.2:
                lo, hi = lo - 0.1, hi + 0.1
            self.p_rho.setYRange(lo, hi, padding=0.08)

        W.style(self.p_temp, "Peak temperatures", None, "temperature (K)")
        W.curve(self.p_temp, t, S["T_fuel_max"], "fuel", name="fuel")
        W.curve(self.p_temp, t, S["T_clad_max"], "clad", name="cladding")
        W.curve(self.p_temp, t, S["T_cool_max"], "cool", name="coolant")
        if rt.T_clad_fail:
            ln = W.hline(self.p_temp, rt.T_clad_fail, "limit")
            self.p_temp.legend.addItem(pg.PlotDataItem([], [], pen=ln.pen), f"clad failure {rt.T_clad_fail:.0f} K")

        W.style(self.p_press, "Primary pressure", None, "pressure (MPa)")
        W.curve(self.p_press, t, S["p"], 0, name="pressure")
        self.tw_press.clear()
        water = rt.coolant in ("water", "d2o") or rt.moderator == "d2o"
        self.tw_press.show(water, "coolant inventory (t)")
        if water:
            c = pg.PlotDataItem(t, S["M"], pen=theme.pen(2, 1.6, "dash"))
            c.setLogMode(logx, False)
            self.tw_press.vb.addItem(c)
            self.p_press.legend.addItem(c, "inventory (right axis)")
            self.tw_press.vb.enableAutoRange()
        p = np.asarray(S["p"], float)
        if p.size and np.ptp(p) < 1e-3 * max(abs(p.mean()), 1e-9):      # flat: avoid a 1e-9-wide axis
            self.p_press.setYRange(p.mean() * 0.95, p.mean() * 1.05, padding=0)
        else:
            self.p_press.enableAutoRange(y=True)

        W.style(self.p_dmg, "Core damage state", None, "fraction of fuel cells")
        keys = ["frac_intact", "frac_failed", "frac_damaged", "frac_molten", "frac_relocated"]
        base = np.zeros_like(t)
        tx = np.log10(t) if logx else t                 # PlotCurveItem ignores the plot's log mode
        prev = pg.PlotCurveItem(tx, base, pen=pg.mkPen(None))
        self.p_dmg.addItem(prev)
        for k, c, nm in zip(keys, theme.STATE_COLORS, STATE_NAMES):
            top = base + S[k]
            c1 = pg.PlotCurveItem(tx, top, pen=pg.mkPen(c, width=1))
            self.p_dmg.addItem(c1)
            self.p_dmg.addItem(pg.FillBetweenItem(prev, c1, brush=pg.mkBrush(c)))
            self.p_dmg.legend.addItem(pg.ScatterPlotItem([], [], symbol="s", size=10, brush=c, pen=None), nm)
            prev, base = c1, top
        self.p_dmg.setYRange(0, 1.0, padding=0.02)

        W.style(self.p_h2, "Hydrogen and FP release", None, "H₂ generated (kg)")
        W.curve(self.p_h2, t, S["H2"], "h2", name="H₂ (left axis)")
        self.tw_h2.clear()
        self.tw_h2.show(True, "volatile FP released (%)")
        c = pg.PlotDataItem(t, np.maximum(S["release"], 0.0), pen=theme.pen("xenon", 1.6, "dash"))
        c.setLogMode(logx, False)
        self.tw_h2.vb.addItem(c)
        self.p_h2.legend.addItem(c, "FP release (right axis)")
        for tw, y in ((self.tw_h2, S["release"]), (self.tw_press, S.get("M"))):
            if y is not None and np.nanmax(y) - np.nanmin(y) < 1e-6:
                v = float(np.nanmax(y))
                tw.vb.setYRange(0, 1) if abs(v) < 1e-6 else tw.vb.setYRange(v * 0.95, v * 1.05)
        if np.nanmax(S["H2"]) < 1e-6:
            self.p_h2.setYRange(0, 1, padding=0)
        else:
            self.p_h2.enableAutoRange(y=True)

        for p in self.plant_plots:
            p.getAxis("bottom").showLabel(False)
            W.fit_legend(p.legend, p.vb.width())
        for p in (self.p_dmg, self.p_h2):
            W.axis_label(p, "bottom", xl)
        if self.chk_cursor.isChecked() and res.snapshots:
            tc = res.snapshots[self.frame()].t
            x = np.log10(max(tc, 1e-3)) if logx else tc / ts
            for p in self.plant_plots:
                p.addItem(pg.InfiniteLine(x, angle=90, pen=pg.mkPen(theme.current["accent"], width=1.5)))

    def _show_events(self):
        ev = self.res.events
        self.ev_table.setRowCount(len(ev))
        for i, (t, msg) in enumerate(ev):
            self.ev_table.setItem(i, 0, QtWidgets.QTableWidgetItem(_fmt_t(t)))
            self.ev_table.setItem(i, 1, QtWidgets.QTableWidgetItem(msg))

    def summary(self, res):
        s = res.summary
        rt = REACTORS[res.reactor]
        rows = [("simulated time", _fmt_t(float(res.t[-1]))),
                ("peak power", f"{s['peak_power']:.3g} × rated ({s['peak_power'] * rt.power_MW:,.0f} MW)"),
                ("peak fuel temperature", W.fmt(s["peak_fuel_T"], "K")),
                ("peak cladding temperature", W.fmt(s["peak_clad_T"], "K")),
                ("peak fuel enthalpy", W.fmt(s["peak_enthalpy"], "cal/g", 3)),
                ("hydrogen generated", W.fmt(s["H2_kg"], "kg", 3)),
                ("max cladding oxidation", W.fmt(s["ECR_max"], "% ECR", 3)),
                ("cells failed or worse", f"{s['failed_or_worse'] * 100:.1f} %"),
                ("cells molten (peak)", f"{s['molten_max'] * 100:.1f} %"),
                ("cells relocated", f"{s['relocated'] * 100:.1f} %"),
                ("fission-product release", f"{s['release']:.2f} %"),
                ("vessel breached", "yes" if s["vessel_failed"] else "no")]
        names = {"scram": "reactor trip", "core uncovery": "core uncovery", "first cladding failure": "first clad failure",
                 "first loss": "first loss of geometry", "first molten": "first melt",
                 "first relocation": "first relocation", "lower-head failure": "vessel breach"}
        for k, lab in names.items():
            if k in s["times"]:
                rows.append((lab, _fmt_t(s["times"][k])))
        if np.isfinite(s.get("min_DNBR", np.nan)):
            rows.append(("minimum DNBR", f"{s['min_DNBR']:.2f}"))
        return rows

    # ------------------------------------------------------------------ misc
    def restyle(self):
        for m in self.panels:
            m.view.restyle()
        self.timeline.restyle()

    def invalidate(self):
        self.restyle()
        self.eq.set_html(self.equations())
        self.refresh()

    def current_view(self):
        return self.tabs.currentWidget()

    def export_png(self, path):
        W.grab_png(self.current_view(), path)

    def export_data(self, path):
        if self.res is None:
            return
        keys = list(self.res.series)
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t"] + keys)
            for i in range(len(self.res.t)):
                w.writerow([f"{self.res.t[i]:.6g}"] + [f"{self.res.series[k][i]:.6g}" for k in keys])

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
