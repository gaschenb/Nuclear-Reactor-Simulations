"""Nuclear Reactor Simulations desktop app (PyQt5 + pyqtgraph).

    reactorsim-gui      or      python -m reactorsim.gui

Sidebar workspaces: the core accident simulator (live lattice maps while the coupled model runs
in a worker thread), Monte Carlo ensembles, xenon and samarium poisoning, reactor kinetics, decay
heat and the reactor designs. The analytic workspaces recompute live as sliders move.
"""
from __future__ import annotations

import os
import sys

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt

from . import theme
from . import widgets as W

APP_NAME = "Nuclear Reactor Simulations"

PAGES = [
    ("Core accident", "sim_page", "SimPage"),
    ("Ensemble", "ensemble_page", "EnsemblePage"),
    ("Xenon and samarium", "poison_page", "PoisonPage"),
    ("Kinetics", "kinetics_page", "KineticsPage"),
    ("Decay heat", "decay_page", "DecayPage"),
    ("Reactor designs", "designs_page", "DesignsPage"),
]


def restyle_plots(root):
    """Re-apply the theme colours to every pyqtgraph view (background, axes, titles)."""
    import pyqtgraph as pg
    t = theme.current
    for gv in root.findChildren(pg.GraphicsView):
        gv.setBackground(t["plot_bg"])
        for it in gv.scene().items():
            if isinstance(it, pg.AxisItem):
                it.setPen(pg.mkPen(t["plot_fg"])); it.setTextPen(pg.mkPen(t["plot_fg"]))
            elif isinstance(it, pg.PlotItem):
                lab = it.titleLabel
                if lab.text:
                    import re
                    txt = re.sub(r"color:[^;']+", f"color:{t['plot_fg']}", lab.text)
                    lab.setText(txt)
                if it.legend is not None:
                    W.legend_color(it.legend, t["plot_fg"])
            elif isinstance(it, pg.LegendItem):
                W.legend_color(it, t["plot_fg"])


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, theme_name="light", pages=None):
        super().__init__()
        app = QtWidgets.QApplication.instance()
        theme.apply(app, theme_name)
        import importlib
        self.setWindowTitle(APP_NAME)
        self.resize(1680, 960)
        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        h = QtWidgets.QHBoxLayout(central); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(0)
        side = QtWidgets.QWidget(); side.setFixedWidth(190)
        sv = QtWidgets.QVBoxLayout(side); sv.setContentsMargins(0, 0, 0, 0); sv.setSpacing(0)
        b = QtWidgets.QLabel(APP_NAME); b.setWordWrap(True); b.setObjectName("brand"); sv.addWidget(b)
        s = QtWidgets.QLabel("severe-accident simulation"); s.setObjectName("brandSub"); sv.addWidget(s)
        self.nav = QtWidgets.QListWidget(); self.nav.setObjectName("nav")
        self.nav.setFocusPolicy(Qt.NoFocus)
        sv.addWidget(self.nav, 1)
        h.addWidget(side)
        self.stack = QtWidgets.QStackedWidget()
        h.addWidget(self.stack, 1)
        self.pages = []
        for name, mod, cls in PAGES:
            if pages is not None and name not in pages:
                continue
            m = importlib.import_module(f".{mod}", __package__)
            p = getattr(m, cls)()
            p.status.connect(self.show_status)
            for r in (p.runner, p.heavy_runner):
                r.busy.connect(self._busy)
            # below the page's minimum size the page scrolls instead of squashing its panels
            sa = QtWidgets.QScrollArea(); sa.setWidgetResizable(True); sa.setFrameShape(QtWidgets.QFrame.NoFrame)
            sa.setWidget(p)
            p.setMinimumSize(getattr(p, "MIN_SIZE", (1180, 760))[0], getattr(p, "MIN_SIZE", (1180, 760))[1])
            self.stack.addWidget(sa)
            self.pages.append(p)
            self.nav.addItem(name)
        self.nav.currentRowChanged.connect(self._page_changed)

        tb = QtWidgets.QToolBar(); tb.setMovable(False); tb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.addToolBar(tb)
        tb.addAction("▶  Run", self.run).setShortcut("Ctrl+R")
        tb.addAction("■  Cancel", self.cancel).setShortcut("Esc")
        self.a_live = tb.addAction("Live updates"); self.a_live.setCheckable(True)
        self.a_live.setChecked(True); self.a_live.setShortcut("Ctrl+L")
        self.a_live.toggled.connect(lambda on: [p.set_live(on) for p in self.pages])
        tb.addSeparator()
        tb.addAction("Export plot…", self.export_png).setShortcut("Ctrl+E")
        tb.addAction("Export results…", self.export_csv).setShortcut("Ctrl+Shift+E")
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        tb.addWidget(spacer)
        self.a_theme = tb.addAction("Dark theme", self.toggle_theme); self.a_theme.setShortcut("Ctrl+T")
        tb.addAction("Help", self.about).setShortcut("F1")
        self.busy = QtWidgets.QProgressBar(); self.busy.setRange(0, 0); self.busy.setFixedWidth(120)
        self.busy.setVisible(False)
        self.statusBar().addPermanentWidget(self.busy)
        if theme_name == "dark":
            self.a_theme.setText("Light theme")
        self.nav.setCurrentRow(0)

    def _busy(self, _):
        self.busy.setVisible(any(r.running() for p in self.pages
                                 for r in (p.runner, p.heavy_runner)))

    def page(self):
        return self.pages[self.stack.currentIndex()]

    def _page_changed(self, i):
        self.stack.setCurrentIndex(i)
        self.page().refresh()

    def show_status(self, msg):
        self.statusBar().showMessage(msg)

    def run(self):
        self.page().run_all()

    def cancel(self):
        self.page().cancel()

    def export_png(self):
        name = self.page().tabs.tabText(self.page().tabs.currentIndex()).lower().replace(" ", "_")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export plot", f"{name}.png",
                                                        "PNG image (*.png)")
        if path:
            self.page().export_png(path); self.show_status(f"saved {path}")

    def export_csv(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export results", "results.csv",
                                                        "CSV (*.csv)")
        if path:
            self.page().export_data(path); self.show_status(f"saved {path}")

    def toggle_theme(self):
        new = "dark" if theme.current["name"] == "light" else "light"
        theme.apply(QtWidgets.QApplication.instance(), new)
        self.a_theme.setText("Light theme" if new == "dark" else "Dark theme")
        restyle_plots(self)
        for p in self.pages:
            p.invalidate()

    def about(self):
        QtWidgets.QMessageBox.information(self, APP_NAME, (
            f"{APP_NAME}\n\n"
            "Ctrl+R  run the scenario / ensemble / full computation of the workspace\n"
            "Esc     cancel\n"
            "Ctrl+L  toggle live updates\n"
            "Ctrl+E  export the current plot, Ctrl+Shift+E the results table\n"
            "Ctrl+T  light / dark theme\n\n"
            "Simulations run in a background thread and update the lattice maps live; the time "
            "slider scrubs through the stored snapshots. Analytic pages recompute as the sliders "
            "move. Plots: drag to pan, wheel to zoom, right-click for export and axis options."))

    def closeEvent(self, e):
        for p in self.pages:
            for r in (p.runner, p.heavy_runner):
                r.cancel()
                for t in list(r.threads):
                    t.wait(5000)
        super().closeEvent(e)


def main(argv=None):
    argv = sys.argv if argv is None else argv
    QtCore.QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QtCore.QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    app.setApplicationName(APP_NAME)
    f = app.font(); f.setPointSize(9)
    if sys.platform.startswith("win"):
        f.setFamily("Segoe UI")
    app.setFont(f)
    w = MainWindow("dark" if "--dark" in argv or os.environ.get("REACTORSIM_THEME") == "dark"
                   else "light")
    w.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
