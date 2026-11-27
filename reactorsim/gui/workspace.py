"""Base class of the analytic workspaces: example picker, live parameter sliders, background
computation (a fast live pass on every change and an optional heavy pass), result table,
typeset equations and a tab widget of plots."""
from __future__ import annotations

import csv
import time

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt

from . import widgets as W


class Workspace(QtWidgets.QWidget):
    status = QtCore.pyqtSignal(str)

    title = "Workspace"
    PARAMS: list = []
    CHOICES: list = []
    SCENARIOS: list = []           # dicts: name, desc, values (display units), data
    HEAVY_KEYS: tuple = ()         # result keys produced by the heavy pass only
    VIEW_KEYS: tuple = ()          # parameters that only change the display (no recomputation)
    auto_heavy = False             # run the heavy pass automatically after live updates
    run_label = "Run"
    scenario_label = "Example"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("page")
        self.res = {}
        self.last_error = None
        self.live = True
        self.scenario = self.SCENARIOS[0] if self.SCENARIOS else {}
        self.runner = W.Runner(self)
        self.heavy_runner = W.Runner(self)
        self.runner.done.connect(self._done)
        self.heavy_runner.done.connect(self._heavy_done)
        for r in (self.runner, self.heavy_runner):
            r.failed.connect(self._failed)
        self.heavy_runner.progressed.connect(self._progress)
        self._t0 = 0.0
        self._debounce = QtCore.QTimer(self); self._debounce.setSingleShot(True)
        self._debounce.setInterval(35)
        self._debounce.timeout.connect(self._launch)
        self._heavy_debounce = QtCore.QTimer(self); self._heavy_debounce.setSingleShot(True)
        self._heavy_debounce.setInterval(250)
        self._heavy_debounce.timeout.connect(lambda: self.run_all(auto=True))
        self._build()
        if self.SCENARIOS:
            self.cb_scn.setCurrentIndex(0)
            self._apply_scenario(0)

    # ------------------------------------------------------------------ layout
    def _build(self):
        h = QtWidgets.QHBoxLayout(self); h.setContentsMargins(10, 10, 10, 10); h.setSpacing(10)
        left = QtWidgets.QScrollArea(); left.setWidgetResizable(True); left.setFixedWidth(350)
        left.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        lw = QtWidgets.QWidget(); lv = QtWidgets.QVBoxLayout(lw)
        lv.setContentsMargins(0, 0, 4, 0); lv.setSpacing(10)
        f, v = W.card(self.scenario_label)
        self.cb_scn = QtWidgets.QComboBox()
        self.cb_scn.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cb_scn.setMinimumContentsLength(12)
        for s in self.SCENARIOS:
            self.cb_scn.addItem(s["name"])
        self.cb_scn.currentIndexChanged.connect(self._apply_scenario)
        v.addWidget(self.cb_scn)
        self.lab_desc = W.muted("")
        v.addWidget(self.lab_desc)
        lv.addWidget(f)
        f, v = W.card("Parameters")
        self.params = W.ParamPanel(self.PARAMS, self.CHOICES)
        self.params.changed.connect(self._changed)
        v.addWidget(self.params)
        row = QtWidgets.QHBoxLayout()
        self.btn_run = QtWidgets.QPushButton(self.run_label); self.btn_run.setObjectName("primary")
        self.btn_run.clicked.connect(lambda: self.run_all())
        self.btn_reset = QtWidgets.QPushButton("Reset")
        self.btn_reset.clicked.connect(lambda: self._apply_scenario(self.cb_scn.currentIndex()))
        row.addWidget(self.btn_run, 1); row.addWidget(self.btn_reset)
        v.addLayout(row)
        self.progress = QtWidgets.QProgressBar(); self.progress.setRange(0, 100)
        self.progress.setVisible(False)
        v.addWidget(self.progress)
        lv.addWidget(f)
        extra = self.build_controls()
        if extra is not None:
            lv.addWidget(extra)
        lv.addStretch(1)
        left.setWidget(lw)
        h.addWidget(left)

        self.tabs = QtWidgets.QTabWidget()
        for name, wdg in self.build_views():
            self.tabs.addTab(wdg, name)
        self.tabs.currentChanged.connect(lambda _: self.refresh())
        h.addWidget(self.tabs, 1)

        right = QtWidgets.QScrollArea(); right.setWidgetResizable(True); right.setFixedWidth(350)
        right.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        rw = QtWidgets.QWidget(); rv = QtWidgets.QVBoxLayout(rw)
        rv.setContentsMargins(0, 0, 4, 0); rv.setSpacing(10)
        f, v = W.card("Results")
        self.table = W.ResultTable()
        self.table.setMinimumHeight(260)
        v.addWidget(self.table)
        rv.addWidget(f)
        f, v = W.card("Model")
        self.eq = W.EquationView(self.equations())
        v.addWidget(self.eq)
        rv.addWidget(f)
        rv.addStretch(1)
        right.setWidget(rw)
        h.addWidget(right)

    # ------------------------------------------------------------------ overridables
    def build_controls(self):
        return None

    def build_views(self):
        return []

    def equations(self):
        return ""

    def compute(self, p, heavy, ctx):
        return {}

    def show_result(self, res):
        pass

    def summary(self, res):
        return []

    def export_data(self, path):
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["quantity", "value"])
            w.writerows(self.table.rows())

    # ------------------------------------------------------------------ plumbing
    def values(self):
        p = self.params.values()
        p["_scenario"] = self.scenario
        return p

    def _apply_scenario(self, i):
        if not self.SCENARIOS:
            return
        self.scenario = self.SCENARIOS[i]
        self.lab_desc.setText(self.scenario.get("desc", ""))
        vals = self.params.defaults()
        vals.update(self.scenario.get("values", {}))
        self.params.set_values(vals)
        self.res = {}
        self._update_visibility()
        self.on_scenario()
        self.eq.set_html(self.equations())
        self._launch()
        if self.auto_heavy or self.scenario.get("auto_run"):
            self._heavy_debounce.start()

    def on_scenario(self):
        pass

    def hidden_keys(self, p):
        """Parameters that do not apply for the current values (hidden in the panel)."""
        return ()

    def _update_visibility(self):
        self.params.set_hidden(self.hidden_keys(self.params.values()))

    def _changed(self, key):
        self._update_visibility()
        if key in self.VIEW_KEYS:
            self.refresh()
            return
        spec = self.params.specs.get(key)
        self.eq.set_html(self.equations())
        for k in self.HEAVY_KEYS:
            self.res.pop(k, None)
        self.heavy_runner.cancel()
        if spec is not None and spec.heavy:
            self._mark_stale()
            if self.auto_heavy:
                self._heavy_debounce.start()
            return
        if self.live:
            self._debounce.start()
        if self.auto_heavy and self.live:
            self._heavy_debounce.start()

    def _mark_stale(self):
        self.status.emit(f"{self.title}: parameters changed — press {self.run_label} "
                         f"(Ctrl+R) to update the full computation")
        self.refresh()

    def _launch(self):
        vals = self.values()
        self._t0 = time.perf_counter()
        self.runner.submit(lambda ctx: self.compute(vals, False, ctx))

    def run_all(self, auto=False):
        vals = self.values()
        self._th0 = time.perf_counter()
        self.progress.setValue(0); self.progress.setVisible(True)
        self.heavy_runner.submit(lambda ctx: self.compute(vals, True, ctx))
        if not auto:
            self.status.emit(f"{self.title}: running…")

    def cancel(self):
        self.runner.cancel(); self.heavy_runner.cancel()
        self.progress.setVisible(False)
        self.status.emit("cancelled")

    def set_live(self, on):
        self.live = on

    def _done(self, gen, res):
        if not self.runner.is_current(gen) or res is None:
            return
        keep = {k: v for k, v in self.res.items() if k in self.HEAVY_KEYS}
        self.res = dict(res, **keep)
        dt = (time.perf_counter() - self._t0) * 1e3
        self.refresh()
        self.status.emit(f"{self.title}: updated in {dt:.0f} ms")

    def _heavy_done(self, gen, res):
        if not self.heavy_runner.is_current(gen) or res is None:
            return
        self.res.update(res)
        self.progress.setVisible(False)
        dt = time.perf_counter() - self._th0
        self.refresh()
        self.status.emit(f"{self.title}: full computation finished in {dt:.2f} s")

    def _progress(self, gen, i, n):
        if self.heavy_runner.is_current(gen):
            self.progress.setValue(int(100 * i / max(n, 1)))

    def _failed(self, gen, msg):
        self.last_error = msg
        self.progress.setVisible(False)
        self.status.emit("error: " + msg.splitlines()[0])
        print(msg)

    def refresh(self):
        if not self.res:
            return
        try:
            self.show_result(self.res)
            self.table.set_rows(self.summary(self.res))
        except Exception as e:      # keep the GUI alive, report
            import traceback
            self.last_error = traceback.format_exc()
            traceback.print_exc()
            self.status.emit(f"display error: {e}")

    def invalidate(self):
        """Theme changed: rebuild plot styling by redrawing."""
        self.restyle()
        self.eq.set_html(self.equations())
        self.refresh()

    def restyle(self):
        pass

    def current_view(self):
        return self.tabs.currentWidget()

    def export_png(self, path):
        W.grab_png(self.current_view(), path)

    def busy(self):
        return self.runner.running() or self.heavy_runner.running() or self._debounce.isActive()

    def settle(self, ms=120_000):
        """Process events until all computation is finished (tests, screenshots)."""
        app = QtWidgets.QApplication.instance()
        t = time.time()
        while time.time() - t < ms / 1000:
            app.processEvents()
            if not (self.busy() or self._heavy_debounce.isActive() or self.runner.threads
                    or self.heavy_runner.threads):
                break
            time.sleep(0.005)
        for _ in range(5):
            app.processEvents()
