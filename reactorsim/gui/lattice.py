"""Interactive lattice views: a core map with a physical colour scale, hover read-out, pinned
cells and synchronised highlighting; a timeline strip with events and a draggable time cursor;
the cell inspector (read-out of every field and the history of pinned cells)."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

from . import theme
from . import widgets as W
from .fields import BY_KEY, FIELDS, STATE_NAMES, Field, fmt

PIN_COLORS = ["#00e5ff", "#76ff03", "#ff4081", "#ffea00", "#e040fb", "#ffffff"]


def _upsample(g, k, nearest=False):
    """Bilinear (or nearest) refinement of a cell field by k, keeping missing cells missing."""
    if k <= 1:
        return g
    nz, nx = g.shape
    if nearest:
        return np.repeat(np.repeat(g, k, 0), k, 1)
    mask = np.isfinite(g)
    if not mask.any():
        return np.repeat(np.repeat(g, k, 0), k, 1)
    # fill holes with the nearest finite neighbour along rows so interpolation does not bleed NaN
    f = np.where(mask, g, np.nanmean(g))
    zc = (np.arange(nz * k) + 0.5) / k - 0.5
    xc = (np.arange(nx * k) + 0.5) / k - 0.5
    z0 = np.clip(np.floor(zc).astype(int), 0, nz - 1); z1 = np.clip(z0 + 1, 0, nz - 1)
    x0 = np.clip(np.floor(xc).astype(int), 0, nx - 1); x1 = np.clip(x0 + 1, 0, nx - 1)
    wz = np.clip(zc - np.floor(zc), 0, 1)[:, None]; wx = np.clip(xc - np.floor(xc), 0, 1)[None, :]
    out = ((1 - wz) * (1 - wx) * f[np.ix_(z0, x0)] + (1 - wz) * wx * f[np.ix_(z0, x1)]
           + wz * (1 - wx) * f[np.ix_(z1, x0)] + wz * wx * f[np.ix_(z1, x1)])
    m = np.repeat(np.repeat(mask, k, 0), k, 1)
    return np.where(m, out, np.nan)


_SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _log_ticks(lo, hi):
    """Decade labels 10ⁿ (and unlabeled 2–9 minor ticks) for an axis showing log10 values."""
    n0, n1 = int(np.floor(lo)), int(np.ceil(hi))
    step = max(1, (n1 - n0) // 6)
    major = [(float(n), "10" + str(n).translate(_SUP)) for n in range(n0, n1 + 1, step)]
    minor = []
    if n1 - n0 <= 3:
        minor = [(float(n + np.log10(k)), "") for n in range(n0, n1 + 1) for k in range(2, 10)]
    return [major, minor]


_TICK_FONT = QtGui.QFont()
_TICK_FONT.setPointSizeF(8.0)


def compact_axis(ax):
    """Small, readable tick labels that do not collide in small plots."""
    ax.setStyle(tickFont=_TICK_FONT, tickTextOffset=3, tickLength=-4, autoExpandTextSpace=True)
    if ax.label is not None and ax.labelText:
        ax.setLabel(ax.labelText, units=ax.labelUnits or None,
                    **{"font-size": "8pt", "color": theme.current["plot_fg"]})


def compact_axes(pi, names):
    for n in names:
        compact_axis(pi.getAxis(n))


class LatticeView(pg.GraphicsLayoutWidget):
    """One lattice field on the physical core cross-section (x across, z up)."""
    hovered = QtCore.pyqtSignal(int)
    clicked = QtCore.pyqtSignal(int)

    def __init__(self, field="T_fuel"):
        super().__init__()
        self.ci.layout.setContentsMargins(2, 2, 2, 2)
        self.pi = self.addPlot(row=0, col=0)
        self.pi.setMenuEnabled(False)
        self.pi.hideButtons()
        self.pi.setAspectLocked(False)        # the core fills the plot; zoom is limited to the core
        self.img = pg.ImageItem()
        self.pi.addItem(self.img)
        self.bar = pg.ColorBarItem(colorMap=theme.cmap("inferno"), width=14, interactive=False)
        self.bar.setImageItem(self.img)
        self.addItem(self.bar, row=0, col=1)
        self.state_leg = pg.LegendItem(labelTextSize="8.5pt", verSpacing=-4)
        self.state_leg.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.addItem(self.state_leg, row=0, col=2)
        self.ci.layout.setAlignment(self.state_leg, Qt.AlignVCenter)
        for c, n in zip(theme.STATE_COLORS, STATE_NAMES):
            self.state_leg.addItem(pg.ScatterPlotItem([], [], symbol="s", size=12, brush=c, pen=None), n)
        self.hl = QtWidgets.QGraphicsRectItem(); self.hl.setZValue(20); self.hl.hide()
        self.pi.addItem(self.hl)
        self.tip = pg.TextItem("", anchor=(0, 1)); self.tip.setZValue(30); self.tip.hide()
        self.pi.addItem(self.tip, ignoreBounds=True)
        self.pin_items = []
        self.f = None
        self.field = field
        self.log = False
        self.smooth = False
        self.geom = None        # (nx, nz, W, H)
        self.data = None        # current cell values (nz, nx), stored units
        self.levels = None
        self.scene().sigMouseMoved.connect(self._moved)
        self.scene().sigMouseClicked.connect(self._clicked)
        self.set_field(field)

    # ------------------------------------------------------------- appearance
    def restyle(self):
        t = theme.current
        W.style(self.pi, None, "x (m)", "z (m)")
        compact_axes(self.pi, ("left", "bottom"))
        compact_axis(self.bar.axis)
        self.hl.setPen(pg.mkPen(t["plot_fg"], width=2.5))
        self.tip.fill = pg.mkBrush(QtGui.QColor(t["panel"]).lighter(100))
        self.tip.border = pg.mkPen(t["border"])
        self.tip.setColor(t["text"])
        W.legend_color(self.state_leg, t["plot_fg"])
        ax = self.bar.axis
        ax.setPen(pg.mkPen(t["plot_fg"])); ax.setTextPen(pg.mkPen(t["plot_fg"]))

    def set_field(self, field, log=None):
        """field: a key of fields.FIELDS or a Field object (for derived maps)."""
        f = field if isinstance(field, Field) else BY_KEY[field]
        self.f = f
        self.field = f.key
        if log is not None:
            self.log = log
        is_state = field == "state"
        self.bar.setVisible(not is_state)
        self.state_leg.setVisible(is_state)
        if not is_state:
            self.bar.setColorMap(theme.cmap(f.cmap))
            self.bar.axis.enableAutoSIPrefix(False)
            self.bar.axis.setTicks(None)
        self.restyle()

    # ------------------------------------------------------------- data
    def show(self, data, geom, levels=None):
        """data: cell values (stored units), levels: (lo, hi) in displayed units or None."""
        nx, nz, Wd, Hd = geom
        if geom != self.geom:
            self.pi.vb.setLimits(xMin=-Wd / 2, xMax=Wd / 2, yMin=0.0, yMax=Hd, maxXRange=Wd, maxYRange=Hd,
                                 minXRange=Wd / nx, minYRange=Hd / nz)
            self.pi.setRange(xRange=(-Wd / 2, Wd / 2), yRange=(0, Hd), padding=0)
        self.geom = geom
        f = self.f
        g = np.asarray(data, float).reshape(nz, nx)
        self.data = g
        if self.field == "state":
            lut = np.array([QtGui.QColor(c).getRgb() for c in theme.STATE_COLORS], np.ubyte)
            img = _upsample(g, 4 if self.smooth else 1, nearest=True)
            self.img.setLookupTable(lut)
            self.img.setImage(img + 0.5, autoLevels=False, levels=(0, 5))
        else:
            v = self.display(g)
            if levels is None:
                fin = v[np.isfinite(v)]
                levels = (float(fin.min()), float(fin.max())) if fin.size else (0.0, 1.0)
            lo, hi = levels
            if not (hi > lo):
                hi = lo + max(abs(lo) * 1e-3, 1e-9)
            self.levels = (lo, hi)
            img = _upsample(v, 6 if self.smooth else 1)
            self.img.setLookupTable(None)
            self.img.setImage(img, autoLevels=False, levels=(lo, hi))
            self.bar.setColorMap(theme.cmap(f.cmap))
            self.bar.setLevels((lo, hi))
            self.bar.axis.setTicks(_log_ticks(lo, hi) if self.log else None)
        self.img.setRect(QtCore.QRectF(-Wd / 2, 0.0, Wd, Hd))

    def display(self, g):
        """Stored cell values -> displayed values (unit scale, optional log10)."""
        f = self.f
        v = np.asarray(g, float) * f.scale
        if self.log:
            pos = v[np.isfinite(v) & (v > 0)]
            floor = pos.max() * 1e-8 if pos.size else 1e-30
            v = np.where(np.isfinite(v), np.log10(np.maximum(v, floor)), np.nan)
        return v

    # ------------------------------------------------------------- geometry helpers
    def cell_rect(self, idx):
        nx, nz, Wd, Hd = self.geom
        ix, iz = idx % nx, idx // nx
        dx, dz = Wd / nx, Hd / nz
        return QtCore.QRectF(-Wd / 2 + ix * dx, iz * dz, dx, dz)

    def cell_at(self, scene_pos):
        if self.geom is None or not self.pi.vb.sceneBoundingRect().contains(scene_pos):
            return -1
        p = self.pi.vb.mapSceneToView(scene_pos)
        nx, nz, Wd, Hd = self.geom
        ix = int(np.floor((p.x() + Wd / 2) / Wd * nx)); iz = int(np.floor(p.y() / Hd * nz))
        if 0 <= ix < nx and 0 <= iz < nz:
            return iz * nx + ix
        return -1

    def highlight(self, idx, show_tip=False):
        if idx < 0 or self.geom is None:
            self.hl.hide(); self.tip.hide()
            return
        r = self.cell_rect(idx)
        self.hl.setRect(r); self.hl.show()
        if show_tip and self.data is not None:
            v = self.data.ravel()[idx]
            self.tip.setText(fmt(self.f, v))
            nx, nz, Wd, Hd = self.geom
            right = r.center().x() > Wd * 0.15
            self.tip.setAnchor((1, 1) if right else (0, 1))
            self.tip.setPos(r.left() if right else r.right(), r.bottom() + 0.02 * Hd)
            self.tip.show()
        else:
            self.tip.hide()

    def set_pins(self, pins):
        for it in self.pin_items:
            self.pi.removeItem(it)
        self.pin_items = []
        if self.geom is None:
            return
        for n, (idx, c) in enumerate(pins):
            r = self.cell_rect(idx)
            box = QtWidgets.QGraphicsRectItem(r)
            box.setPen(pg.mkPen(c, width=3)); box.setZValue(15)
            self.pi.addItem(box)
            lab = pg.TextItem(str(n + 1), color=c, anchor=(0.5, 0.5))
            f = lab.textItem.font(); f.setBold(True); f.setPointSizeF(8.5); lab.setFont(f)
            lab.setPos(r.center()); lab.setZValue(16)
            self.pi.addItem(lab, ignoreBounds=True)
            self.pin_items += [box, lab]

    # ------------------------------------------------------------- mouse
    def _moved(self, pos):
        self.hovered.emit(self.cell_at(pos))

    def _clicked(self, ev):
        if ev.button() == Qt.LeftButton and not ev.double():
            idx = self.cell_at(ev.scenePos())
            if idx >= 0:
                self.clicked.emit(idx)

    def leaveEvent(self, e):
        self.hovered.emit(-1)
        super().leaveEvent(e)


class MapPanel(QtWidgets.QWidget):
    """Header (field selector, log toggle) above a LatticeView."""
    changed = QtCore.pyqtSignal()

    def __init__(self, field):
        super().__init__()
        v = QtWidgets.QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(3)
        h = QtWidgets.QHBoxLayout(); h.setSpacing(6)
        self.box = QtWidgets.QComboBox()
        model = QtGui.QStandardItemModel()
        group = None
        for f in FIELDS:
            if f.group != group:
                group = f.group
                hdr = QtGui.QStandardItem(group.upper()); hdr.setEnabled(False)
                fnt = hdr.font(); fnt.setBold(True); fnt.setPointSizeF(7.5); hdr.setFont(fnt)
                model.appendRow(hdr)
            it = QtGui.QStandardItem("   " + f.label); it.setData(f.key, Qt.UserRole)
            model.appendRow(it)
        self.box.setModel(model)
        self.box.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.box.setMinimumContentsLength(14)
        self.chk_log = QtWidgets.QCheckBox("log scale")
        self.chk_log.setToolTip("logarithmic colour scale (log₁₀ of the value)")
        h.addWidget(self.box, 1); h.addWidget(self.chk_log)
        v.addLayout(h)
        self.view = LatticeView(field)
        v.addWidget(self.view, 1)
        self.set_field(field)
        self.box.currentIndexChanged.connect(self._box)
        self.chk_log.toggled.connect(self._log)

    def field(self):
        return self.view.field

    def set_field(self, key, log=None):
        f = BY_KEY[key]
        for i in range(self.box.count()):
            if self.box.itemData(i, Qt.UserRole) == key:
                self.box.blockSignals(True); self.box.setCurrentIndex(i); self.box.blockSignals(False)
        lg = f.log if log is None else log
        self.chk_log.blockSignals(True); self.chk_log.setChecked(lg); self.chk_log.blockSignals(False)
        self.chk_log.setEnabled(key != "state")
        self.view.set_field(key, lg and key != "state")

    def _box(self, i):
        key = self.box.itemData(i, Qt.UserRole)
        if key:
            self.set_field(key)
            self.changed.emit()

    def _log(self, on):
        self.view.set_field(self.view.field, on)
        self.changed.emit()


class Timeline(pg.GraphicsLayoutWidget):
    """Total power and the core damage stack versus time, event markers and a draggable cursor."""
    seek = QtCore.pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.setFixedHeight(240)
        self.ci.layout.setContentsMargins(2, 0, 2, 0)
        self.ci.layout.setVerticalSpacing(0)
        self.p1 = self.addPlot(row=0, col=0)
        self.p2 = self.addPlot(row=1, col=0)
        self.ci.layout.setRowStretchFactor(0, 3)
        self.ci.layout.setRowStretchFactor(1, 2)
        self.p2.setXLink(self.p1)
        for p in (self.p1, self.p2):
            p.setMenuEnabled(False); p.hideButtons(); p.setMouseEnabled(x=False, y=False)
            p.getAxis("left").setWidth(W.LEFT_W)
        self.p1.hideAxis("bottom")
        side = W.side_legend(self.p1)
        self.leg1 = side
        W.side_legend(self.p2)
        self.cursors = []
        for p in (self.p1, self.p2):
            c = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen(theme.current["accent"], width=2),
                                hoverPen=pg.mkPen(theme.current["accent"], width=4))
            c.setZValue(50)
            c.sigDragged.connect(self._dragged)
            p.addItem(c)
            self.cursors.append(c)
        self.scale = 1.0
        self.scene().sigMouseClicked.connect(self._clicked)
        self.ev = pg.ScatterPlotItem(hoverable=True, tip=lambda x, y, data: data, size=11,
                                     symbol="t", hoverSize=15)
        self.ev.setZValue(40)
        self.restyle()

    def restyle(self):
        W.style(self.p1, None, None, "power (MW)")
        W.style(self.p2, None, None, "fraction")
        compact_axes(self.p1, ("left",))
        compact_axes(self.p2, ("left", "bottom"))
        for c in self.cursors:
            c.setPen(pg.mkPen(theme.current["accent"], width=2))

    def set_result(self, res, scale, unit, log_time=False):
        self.scale = scale
        S = res.series
        t = np.asarray(res.t) / scale
        self.p1.clear(); self.p2.clear()
        self.p1.legend.clear(); self.p2.legend.clear()
        self.p1.setLogMode(y=True)
        self.p1.plot(t, np.maximum(S["P_total"], 1e-3), pen=theme.pen("power", 1.8), name="thermal power",
                     fillLevel=-3, brush=pg.mkBrush(QtGui.QColor(theme.color("power")).lighter(150)))
        self.p1.plot(t, np.maximum(S["P_decay"], 1e-3), pen=theme.pen("decay", 1.2, "dash"), name="decay heat")
        keys = ["frac_intact", "frac_failed", "frac_damaged", "frac_molten", "frac_relocated"]
        base = np.zeros_like(t)
        prev = pg.PlotCurveItem(t, base, pen=pg.mkPen(None))
        self.p2.addItem(prev)
        for k, c, nm in zip(keys, theme.STATE_COLORS, STATE_NAMES):
            top = base + S[k]
            c1 = pg.PlotCurveItem(t, top, pen=pg.mkPen(c, width=1))
            self.p2.addItem(c1)
            self.p2.addItem(pg.FillBetweenItem(prev, c1, brush=pg.mkBrush(c)))
            self.p2.legend.addItem(pg.ScatterPlotItem([], [], symbol="s", size=10, brush=c, pen=None), nm)
            prev, base = c1, top
        self.p2.setYRange(0, 1, padding=0)
        self.p2.setLabel("bottom", f"time ({unit})", **{"font-size": "8pt", "color": theme.current["plot_fg"]})
        self.p1.addItem(self.ev)
        ev = [(tt / scale, msg) for tt, msg in res.events]
        if ev:
            ymax = float(np.nanmax(S["P_total"])) if len(S["P_total"]) else 1.0
            y = np.log10(max(ymax, 1e-3)) + 0.3
            self.ev.setData([e[0] for e in ev], [y] * len(ev), data=[f"{_fmt(e[0] * scale)}: {e[1]}" for e in ev],
                            brush=theme.color("fg"), pen=None)
        for c in self.cursors:
            self.p1.addItem(c) if c is self.cursors[0] else self.p2.addItem(c)
        self.p1.setXRange(0, max(float(t[-1]), 1e-6), padding=0.01)
        P = np.asarray(S["P_total"], float)
        if P.size and np.isfinite(P).any():
            hi = float(np.nanmax(P)); lo = max(float(np.nanmin(P)), hi * 1e-4, 1e-3)
            self.p1.setYRange(np.log10(lo) - 0.1, np.log10(hi) + 0.45, padding=0)

    def set_cursor(self, t):
        for c in self.cursors:
            c.blockSignals(True); c.setValue(t / self.scale); c.blockSignals(False)

    def _dragged(self, line):
        v = line.value()
        for c in self.cursors:
            if c is not line:
                c.blockSignals(True); c.setValue(v); c.blockSignals(False)
        self.seek.emit(v * self.scale)

    def _clicked(self, ev):
        if ev.button() != Qt.LeftButton:
            return
        for p in (self.p1, self.p2):
            if p.vb.sceneBoundingRect().contains(ev.scenePos()):
                x = p.vb.mapSceneToView(ev.scenePos()).x()
                self.set_cursor(x * self.scale)
                self.seek.emit(x * self.scale)
                return


def _fmt(t):
    if t < 120:
        return f"{t:.1f} s"
    if t < 2 * 3600:
        return f"{t / 60:.1f} min"
    return f"{t / 3600:.2f} h"


class CellInspector(QtWidgets.QWidget):
    """Read-out of every field in the hovered (or last pinned) cell and the history of the
    pinned cells, with a cursor at the displayed time."""
    unpin_all = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        v = QtWidgets.QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(6)
        self.lab = W.muted("Hover over the core to read a cell; click a cell to pin it and follow its "
                           "history (up to six).")
        v.addWidget(self.lab)
        self.head = QtWidgets.QLabel(""); self.head.setTextFormat(Qt.RichText)
        v.addWidget(self.head)
        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["quantity", "value"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.table.verticalHeader().setDefaultSectionSize(20)
        self.table.setMinimumHeight(250)
        v.addWidget(self.table)
        h = QtWidgets.QHBoxLayout()
        h.addWidget(QtWidgets.QLabel("history of"))
        self.box = QtWidgets.QComboBox()
        for f in FIELDS:
            self.box.addItem(f.label, f.key)
        self.box.setCurrentIndex(self.box.findData("T_fuel"))
        self.box.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.box.setMinimumContentsLength(10)
        h.addWidget(self.box, 1)
        b = QtWidgets.QPushButton("Clear pins"); b.clicked.connect(self.unpin_all.emit)
        h.addWidget(b)
        v.addLayout(h)
        self.plot = pg.PlotWidget(); self.plot.setMinimumHeight(220)
        self.pi = self.plot.getPlotItem()
        self.pi.setMenuEnabled(False)
        W.style(self.pi)
        W.side_legend(self.pi)
        v.addWidget(self.plot)
        self.cursor = None

    def show_cell(self, res, k, idx, geom, pinned_no=None):
        if res is None or idx < 0 or not res.snapshots:
            return
        snap = res.snapshots[k]
        nx, nz, Wd, Hd = geom
        ix, iz = idx % nx, idx // nx
        x = -Wd / 2 + (ix + 0.5) * Wd / nx; z = (iz + 0.5) * Hd / nz
        tag = f"pin {pinned_no} · " if pinned_no else ""
        fuel = bool(res.fuel_mask[idx])
        self.head.setText(f"<b>{tag}cell ({ix + 1}, {iz + 1})</b> &nbsp; x = {x:+.2f} m, z = {z:.2f} m"
                          + ("" if fuel else " &nbsp; <i>(no fuel)</i>"))
        rows = [(f.label, fmt(f, float(snap.fields[f.key][idx]))) for f in FIELDS if f.key in snap.fields]
        if res.cell_times is not None:
            for s in range(1, 5):
                tt = res.cell_times[s][idx]
                if np.isfinite(tt) and tt <= snap.t + 1e-9:
                    rows.append((f"became {STATE_NAMES[s]} at", _fmt(float(tt))))
        self.table.setRowCount(len(rows))
        for i, (a, b) in enumerate(rows):
            self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(a))
            it = QtWidgets.QTableWidgetItem(b); it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(i, 1, it)

    def show_history(self, res, pins, t_now, scale, unit):
        self.pi.clear(); self.pi.legend.clear()
        key = self.box.currentData()
        f = BY_KEY[key]
        W.style(self.pi, None, f"time ({unit})", f.unit or f.name)
        if res is None or not res.snapshots or not pins:
            return
        ts = np.array([s.t for s in res.snapshots]) / scale
        stack = np.stack([s.fields[key] for s in res.snapshots])
        for n, (idx, c) in enumerate(pins):
            y = stack[:, idx].astype(float) * f.scale
            if key == "state":
                self.pi.plot(ts, y, pen=pg.mkPen(c, width=2), stepMode=None, name=f"pin {n + 1}")
            else:
                self.pi.plot(ts, y, pen=pg.mkPen(c, width=2), name=f"pin {n + 1}")
        if key == "state":
            self.pi.getAxis("left").setTicks([[(i, s) for i, s in enumerate(STATE_NAMES)]])
        else:
            self.pi.getAxis("left").setTicks(None)
        self.pi.setLogMode(y=f.log)
        self.cursor = pg.InfiniteLine(t_now / scale, angle=90, pen=theme.fg_pen(1.0, "dash"))
        self.pi.addItem(self.cursor)
