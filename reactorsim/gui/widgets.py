"""Shared GUI building blocks: background runner, parameter sliders, cards, plots, images."""
from __future__ import annotations

import math
import traceback
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

from . import theme


# =========================================================================== background jobs
class Cancelled(Exception):
    pass


class JobContext:
    def __init__(self, runner, gen):
        self.gen, self.cancelled, self._r = gen, False, runner

    def check(self):
        if self.cancelled:
            raise Cancelled

    def progress(self, i, n):
        self.check()
        self._r.progressed.emit(self.gen, int(i), int(n))


class _JobThread(QtCore.QThread):
    def __init__(self, fn, ctx, runner):
        super().__init__()
        self.fn, self.ctx, self.runner = fn, ctx, runner

    def run(self):
        try:
            out = self.fn(self.ctx)
            if not self.ctx.cancelled:
                self._emit("done", self.ctx.gen, out)
        except Cancelled:
            pass
        except Exception as e:
            if not self.ctx.cancelled:
                self._emit("failed", self.ctx.gen,
                           f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    def _emit(self, name, *args):
        try:
            getattr(self.runner, name).emit(*args)
        except RuntimeError:        # the window was closed while the job was running
            pass


class Runner(QtCore.QObject):
    """One job at a time in a QThread. A new submission cancels the previous one
    (cooperatively, at its next check/progress) and results carry a generation number, so
    stale results are never shown."""
    done = QtCore.pyqtSignal(int, object)
    failed = QtCore.pyqtSignal(int, str)
    progressed = QtCore.pyqtSignal(int, int, int)
    busy = QtCore.pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.gen = 0
        self.ctx = None
        self.threads = set()

    def submit(self, fn):
        if self.ctx is not None:
            self.ctx.cancelled = True
        self.gen += 1
        self.ctx = JobContext(self, self.gen)
        t = _JobThread(fn, self.ctx, self)
        self.threads.add(t)
        t.finished.connect(self._finished)
        t.start()
        self.busy.emit(True)
        return self.gen

    def cancel(self):
        if self.ctx is not None:
            self.ctx.cancelled = True

    def _finished(self):
        t = self.sender()
        self.threads.discard(t)
        if t is not None:
            t.deleteLater()
        if not self.running():
            self.busy.emit(False)

    def running(self):
        return any(t.isRunning() for t in self.threads)

    def is_current(self, gen):
        return gen == self.gen

    def wait_idle(self, ms=600_000):
        app = QtWidgets.QApplication.instance()
        timer = QtCore.QElapsedTimer(); timer.start()
        while (self.running() or self.threads) and timer.elapsed() < ms:
            app.processEvents()
            QtCore.QThread.msleep(5)
        app.processEvents()


# =========================================================================== small widgets
def card(title=None, subtitle=None):
    f = QtWidgets.QFrame(); f.setObjectName("card")
    v = QtWidgets.QVBoxLayout(f); v.setContentsMargins(12, 10, 12, 12); v.setSpacing(6)
    if title:
        lab = QtWidgets.QLabel(title); lab.setObjectName("cardTitle"); v.addWidget(lab)
    if subtitle:
        s = QtWidgets.QLabel(subtitle); s.setObjectName("muted"); s.setWordWrap(True)
        v.addWidget(s)
    return f, v


def muted(text):
    lab = QtWidgets.QLabel(text); lab.setObjectName("muted"); lab.setWordWrap(True)
    return lab


_PATH_CACHE = {}


def math_path(tex, size=13.0):
    """Typeset a mathtext string ($...$) with matplotlib into a QPainterPath (y down, origin at the
    top-left of the ink box). No LaTeX installation or images are involved."""
    key = (tex, size)
    if key in _PATH_CACHE:
        return _PATH_CACHE[key]
    import matplotlib
    from matplotlib.font_manager import FontProperties
    from matplotlib.path import Path
    from matplotlib.textpath import TextPath
    with matplotlib.rc_context({"mathtext.fontset": "stix"}):
        tp = TextPath((0, 0), tex, size=size, prop=FontProperties(family="STIXGeneral"))
    qp = QtGui.QPainterPath()
    for verts, code in tp.iter_segments(curves=True, simplify=False):
        v = verts.reshape(-1, 2)
        if code == Path.MOVETO:
            qp.moveTo(v[0, 0], -v[0, 1])
        elif code == Path.LINETO:
            qp.lineTo(v[0, 0], -v[0, 1])
        elif code == Path.CURVE3:
            qp.quadTo(v[0, 0], -v[0, 1], v[1, 0], -v[1, 1])
        elif code == Path.CURVE4:
            qp.cubicTo(v[0, 0], -v[0, 1], v[1, 0], -v[1, 1], v[2, 0], -v[2, 1])
        elif code == Path.CLOSEPOLY:
            qp.closeSubpath()
    qp.setFillRule(Qt.WindingFill)
    r = qp.boundingRect()
    qp.translate(-r.left(), -r.top())
    _PATH_CACHE[key] = qp
    return qp


class EquationView(QtWidgets.QWidget):
    """A column of typeset equations and short explanatory lines, painted as vector paths.

    Content is a list of items: a string in $...$ is mathtext (matplotlib's TeX subset), any other
    string is prose, and ("h", text) is a small heading. Wide equations shrink to fit."""

    def __init__(self, items=None):
        super().__init__()
        self.setObjectName("equation")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.items = []
        self._layout_w = -1
        self._ops = []
        self._h = 40
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.set_html(items or [])

    def set_html(self, items):          # name kept for the workspace interface
        if isinstance(items, str):
            items = [items] if items else []
        self.items = list(items)
        self._layout_w = -1
        self._relayout(max(self.width(), 300))
        self.updateGeometry()
        self.update()

    set_content = set_html

    def _relayout(self, w):
        pad = 10
        avail = max(w - 2 * pad, 50)
        y = pad
        ops = []
        f = self.font()
        fm = QtGui.QFontMetrics(f)
        hf = QtGui.QFont(f); hf.setBold(True)
        for it in self.items:
            if isinstance(it, tuple) and it and it[0] == "h":
                y += 4
                r = QtGui.QFontMetrics(hf).boundingRect(QtCore.QRect(0, 0, int(avail), 2000),
                                                        Qt.TextWordWrap, it[1])
                ops.append(("h", it[1], QtCore.QRectF(pad, y, avail, r.height())))
                y += r.height() + 4
            elif isinstance(it, str) and it.strip().startswith("$"):
                qp = math_path(it.strip(), 16.0)
                br = qp.boundingRect()
                sc = min(1.0, avail / max(br.width(), 1.0))
                ops.append(("m", qp, (pad + 6, y + 2), sc))
                y += br.height() * sc + 12
            else:
                r = fm.boundingRect(QtCore.QRect(0, 0, int(avail), 4000), Qt.TextWordWrap, str(it))
                ops.append(("t", str(it), QtCore.QRectF(pad, y, avail, r.height())))
                y += r.height() + 6
        self._ops = ops
        self._h = int(y + pad)
        self._layout_w = w
        self.setFixedHeight(self._h)

    def resizeEvent(self, e):
        if e.size().width() != self._layout_w:
            self._relayout(e.size().width())
        super().resizeEvent(e)

    def sizeHint(self):
        return QtCore.QSize(320, self._h)

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        t = theme.current
        opt = QtWidgets.QStyleOption(); opt.initFrom(self)
        self.style().drawPrimitive(QtWidgets.QStyle.PE_Widget, opt, p, self)
        ink = QtGui.QColor(t["eq"])
        muted = QtGui.QColor(t["muted"])
        for op in self._ops:
            if op[0] == "m":
                _, qp, (x, y), sc = op
                p.save(); p.translate(x, y); p.scale(sc, sc)
                p.fillPath(qp, ink)
                p.restore()
            elif op[0] == "h":
                f = QtGui.QFont(self.font()); f.setBold(True); p.setFont(f); p.setPen(ink)
                p.drawText(op[2], Qt.TextWordWrap, op[1])
            else:
                p.setFont(self.font()); p.setPen(muted)
                p.drawText(op[2], Qt.TextWordWrap, op[1])
        p.end()


@dataclass
class P:
    """Numeric parameter: shown in display units (value × scale = SI)."""
    key: str
    label: str
    lo: float
    hi: float
    default: float
    scale: float = 1.0
    log: bool | None = None
    integer: bool = False
    decimals: int = 3
    tip: str = ""
    heavy: bool = False          # changing it only triggers the heavy run, not live updates


@dataclass
class C:
    key: str
    label: str
    options: list               # [(text, value)]
    default: object = None
    tip: str = ""


class ParamRow(QtCore.QObject):
    changed = QtCore.pyqtSignal()
    N = 1000

    def __init__(self, p: P, parent=None):
        super().__init__(parent)
        self.p = p
        self.log = p.log if p.log is not None else (p.lo > 0 and p.hi / p.lo > 60)
        self.label = QtWidgets.QLabel(p.label)
        self.label.setWordWrap(True)
        if p.tip:
            self.label.setToolTip(p.tip)
        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider.setRange(0, self.N); self.slider.setMinimumWidth(80)
        if p.integer:
            self.spin = QtWidgets.QSpinBox(); self.spin.setRange(int(p.lo), int(p.hi))
        else:
            self.spin = QtWidgets.QDoubleSpinBox(); self.spin.setRange(p.lo, p.hi)
            self.spin.setDecimals(p.decimals)
            span = p.hi - p.lo
            self.spin.setSingleStep(10 ** math.floor(math.log10(span / 50)) if span > 0 else 0.1)
        self.spin.setKeyboardTracking(False)
        self.spin.setFixedWidth(84)
        self.spin.setValue(p.default)
        self._sync()
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)

    def _to_pos(self, v):
        p = self.p
        if self.log:
            u = (math.log(max(v, p.lo)) - math.log(p.lo)) / (math.log(p.hi) - math.log(p.lo))
        else:
            u = (v - p.lo) / (p.hi - p.lo) if p.hi > p.lo else 0
        return int(round(min(max(u, 0.0), 1.0) * self.N))

    def _from_pos(self, pos):
        p = self.p
        u = pos / self.N
        if self.log:
            v = math.exp(math.log(p.lo) + u * (math.log(p.hi) - math.log(p.lo)))
        else:
            v = p.lo + u * (p.hi - p.lo)
        if p.integer:
            return int(round(v))
        return float(f"{v:.4g}")

    def _sync(self):
        self.slider.blockSignals(True)
        self.slider.setValue(self._to_pos(self.spin.value()))
        self.slider.blockSignals(False)

    def _from_slider(self, pos):
        self.spin.blockSignals(True); self.spin.setValue(self._from_pos(pos))
        self.spin.blockSignals(False)
        self.changed.emit()

    def _from_spin(self, _):
        self._sync(); self.changed.emit()

    def value(self):
        return self.spin.value()

    def set_value(self, v):
        self.spin.blockSignals(True); self.spin.setValue(v); self.spin.blockSignals(False)
        self._sync()


class ParamPanel(QtWidgets.QWidget):
    """Grid of parameter rows and choice boxes; emits changed(key)."""
    changed = QtCore.pyqtSignal(str)

    def __init__(self, params, choices=(), parent=None):
        super().__init__(parent)
        g = QtWidgets.QGridLayout(self)
        g.setContentsMargins(0, 0, 0, 0); g.setHorizontalSpacing(8); g.setVerticalSpacing(5)
        self.rows, self.boxes, self.specs, self.box_labels = {}, {}, {}, {}
        self.choice_defaults = {c.key: c.default for c in choices if c.default is not None}
        r = 0
        for c in choices:
            lab = QtWidgets.QLabel(c.label)
            lab.setWordWrap(True)
            cb = QtWidgets.QComboBox()
            cb.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            cb.setMinimumContentsLength(8)
            for text, val in c.options:
                cb.addItem(text, val)
            if c.default is not None:
                cb.setCurrentIndex(max(0, cb.findData(c.default)))
            if c.tip:
                cb.setToolTip(c.tip)
            cb.currentIndexChanged.connect(lambda _, k=c.key: self.changed.emit(k))
            g.addWidget(lab, r, 0); g.addWidget(cb, r, 1, 1, 2)
            self.boxes[c.key] = cb
            self.box_labels[c.key] = lab
            r += 1
        for p in params:
            row = ParamRow(p, self)
            row.changed.connect(lambda k=p.key: self.changed.emit(k))
            g.addWidget(row.label, r, 0); g.addWidget(row.slider, r, 1); g.addWidget(row.spin, r, 2)
            self.rows[p.key] = row
            self.specs[p.key] = p
            r += 1
        g.setColumnStretch(1, 1)

    def values(self):
        out = {k: row.value() * row.p.scale for k, row in self.rows.items()}
        out.update({k: cb.currentData() for k, cb in self.boxes.items()})
        return out

    def display_values(self):
        out = {k: row.value() for k, row in self.rows.items()}
        out.update({k: cb.currentData() for k, cb in self.boxes.items()})
        return out

    def set_values(self, d):
        for k, v in d.items():
            if k in self.rows:
                self.rows[k].set_value(v)
            elif k in self.boxes:
                cb = self.boxes[k]
                cb.blockSignals(True); cb.setCurrentIndex(max(0, cb.findData(v)))
                cb.blockSignals(False)

    def set_hidden(self, keys):
        """Hide the controls that do not apply to the current configuration."""
        keys = set(keys or ())
        for k, row in self.rows.items():
            for w in (row.label, row.slider, row.spin):
                w.setVisible(k not in keys)
        for k, cb in self.boxes.items():
            cb.setVisible(k not in keys); self.box_labels[k].setVisible(k not in keys)

    def defaults(self):
        out = {k: r.p.default for k, r in self.rows.items()}
        out.update(self.choice_defaults)
        return out


class ResultTable(QtWidgets.QTableWidget):
    def __init__(self):
        super().__init__(0, 2)
        self.setHorizontalHeaderLabels(["quantity", "value"])
        hh = self.horizontalHeader()
        hh.setStretchLastSection(False)
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setTextElideMode(Qt.ElideMiddle)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setMinimumHeight(140)

    def set_rows(self, rows):
        self.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            a, b = QtWidgets.QTableWidgetItem(str(k)), QtWidgets.QTableWidgetItem(str(v))
            a.setToolTip(f"{k}: {v}"); b.setToolTip(f"{k}: {v}")
            self.setItem(i, 0, a)
            self.setItem(i, 1, b)

    def rows(self):
        return [(self.item(i, 0).text(), self.item(i, 1).text()) for i in range(self.rowCount())]


# =========================================================================== plots
_SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
AXIS_FONT = QtGui.QFont()
AXIS_FONT.setPointSizeF(8.0)
LEFT_W = 54        # fixed left/right axis widths so stacked plots line up


def _pow10(n):
    return "1" if n == 0 else "10" if n == 1 else "10" + str(int(n)).translate(_SUP)


def _nice_step(raw):
    mag = 10.0 ** math.floor(math.log10(raw))
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if m * mag >= raw * (1 - 1e-9):
            return m * mag
    return 10.0 * mag


def _sci(v, digits):
    if v == 0:
        return "0"
    e = int(math.floor(math.log10(abs(v)) + 1e-9))
    m = v / 10.0 ** e
    ms = f"{m:.{digits}f}".rstrip("0").rstrip(".")
    return ("" if ms == "1" else ("-" if ms == "-1" else ms + "×")) + "10" + str(e).translate(_SUP)


class NiceAxis(pg.AxisItem):
    """Axis whose tick count follows the space available (no crowding when a plot is small), with
    clean 1-2-2.5-5 steps, a consistent number of decimals, 10ⁿ labels on log axes, and plain
    1-2-5 labels on log axes that span less than a decade."""
    PX = {"left": 22, "right": 22, "top": 60, "bottom": 60}     # pixels per labelled tick

    def __init__(self, orientation, **kw):
        super().__init__(orientation, **kw)
        self.enableAutoSIPrefix(False)
        self.setStyle(tickFont=AXIS_FONT, tickTextOffset=3, tickLength=-4,
                      textFillLimits=[(0, 0.9), (2, 0.5)])
        self._loglin = False

    def boundingRect(self):
        # let end labels (e.g. 0 and 1 on a 0–1 axis) overhang the plot by half a line instead of
        # being dropped (pyqtgraph allows no overhang once grid lines are on)
        r = super().boundingRect()
        if self.orientation in ("left", "right"):
            return r | self.mapRectFromParent(self.geometry()).adjusted(0, -7, 0, 7)
        return r | self.mapRectFromParent(self.geometry()).adjusted(-12, 0, 12, 0)

    def tickValues(self, lo, hi, size):
        lo, hi = min(lo, hi), max(lo, hi)
        span = hi - lo
        if not (np.isfinite(span) and span > 0 and size > 0):
            return []
        nmax = max(2, int(size / self.PX[self.orientation]))
        self._loglin = False
        if self.logMode:
            n0, n1 = math.ceil(lo - 1e-9), math.floor(hi + 1e-9)
            step = max(1, math.ceil((n1 - n0) / max(1, nmax - 1)))
            major = [float(n) for n in range(n0, n1 + 1) if n % step == 0]
            if len(major) < 2:
                major = [float(n) for n in range(n0, n1 + 1, step)]
            if n1 > n0 or span >= 1.5:
                minor = []
                if step == 1 and size / span > 70:
                    minor = [n + math.log10(k) for n in range(n0 - 1, n1 + 1) for k in range(2, 10)
                             if lo <= n + math.log10(k) <= hi]
                return [(float(step), major), (None, minor)]
            # less than ~a decade: plain linear values placed on the log scale
            self._loglin = True
            a, b = 10.0 ** lo, 10.0 ** hi
            st = _nice_step((b - a) / (3 * nmax))
            while math.floor(b / st + 1e-9) - math.ceil(a / st - 1e-9) + 1 > nmax:
                st = _nice_step(st * 1.0001)
            vals = [v for v in np.arange(math.ceil(a / st) * st, b + st * 1e-6, st) if v > 0]
            return [(st, [math.log10(v) for v in vals])]
        st = _nice_step(span / (3 * nmax))
        while math.floor(hi / st + 1e-9) - math.ceil(lo / st - 1e-9) + 1 > nmax:
            st = _nice_step(st * 1.0001)
        major = np.arange(math.ceil(lo / st - 1e-9) * st, hi + st * 1e-6, st)
        major = [0.0 if abs(v) < st * 1e-6 else float(v) for v in major]
        sub = st / (4 if abs(st / 10 ** math.floor(math.log10(st)) - 2.0) < 1e-6 else 5)
        minor = []
        if size / span * sub >= 7:
            minor = [float(v) for v in np.arange(math.ceil(lo / sub) * sub, hi, sub)]
        return [(st, major), (None, minor)]

    def tickStrings(self, values, scale, spacing):
        if spacing is None:
            return [""] * len(values)
        if self.logMode:
            if self._loglin:
                lin = [10.0 ** v for v in values]
                return self._linear(lin, spacing)
            return [_pow10(round(v)) for v in values]
        return self._linear([v * scale for v in values], spacing * scale)

    @staticmethod
    def _linear(values, step):
        if not values:
            return []
        big = max(abs(v) for v in values)
        if big >= 1e5 or (0 < big < 1e-3):
            d = max(0, int(math.ceil(-math.log10(step / 10 ** math.floor(math.log10(big))) - 1e-9)))
            return [_sci(v, min(d, 2)) for v in values]
        d = max(0, int(math.ceil(-math.log10(step) - 1e-9)))
        if abs(round(step, d) - step) > 1e-9 * step:
            d += 1
        out = []
        for v in values:
            t = f"{v:.{d}f}"
            out.append("0" if t.lstrip("-").strip("0.") == "" else t)
        return out


def nice_axes(pi, names=("left", "bottom")):
    """Swap a PlotItem's axes for NiceAxis (keeps log mode, label, visibility and width)."""
    for n in names:
        old = pi.getAxis(n)
        if isinstance(old, NiceAxis):
            continue
        vis = old.isVisible()
        new = NiceAxis(n)
        pi.setAxisItems({n: new})
        new.setLogMode(old.logMode)
        if old.labelText:
            new.setLabel(old.labelText, units=old.labelUnits or None)
        if old.fixedWidth is not None:
            new.setWidth(old.fixedWidth)
        pi.showAxis(n, vis)


def axis_label(pi, name, text):
    pi.setLabel(name, text, **{"font-size": "8.5pt", "color": theme.current["plot_fg"]})


def top_legend(pi, ncol=3):
    """Horizontal legend between the title and the plot: costs a line of height, not plot width,
    so stacked plots keep identical widths."""
    old = getattr(pi, "legend", None)
    if old is not None and getattr(old, "_top", False):
        return old
    if old is not None and old.scene() is not None:
        old.scene().removeItem(old)
    leg = pg.LegendItem(labelTextSize="8pt", labelTextColor=theme.current["plot_fg"], verSpacing=-8,
                        horSpacing=10, colCount=ncol)
    leg._top = True
    leg.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
    top = pi.getAxis("top")                   # hidden; frees its layout cell for the legend
    pi.layout.removeItem(top)
    top.hide()
    pi.layout.addItem(leg, 1, 1)
    pi.layout.setAlignment(leg, Qt.AlignLeft | Qt.AlignBottom)
    leg.setMinimumWidth(0)                    # never widens the plot; columns adapt instead
    pi.legend = leg
    pi.vb.sigResized.connect(lambda *_: fit_legend(leg, pi.vb.width()))
    return leg


def fit_legend(leg, width):
    """Most legend columns that fit in `width` (wraps to more rows when the plot is narrow)."""
    if not leg.items:
        return
    w = max(s.boundingRect().width() + l.item.boundingRect().width() + leg.layout.horizontalSpacing() + 4
            for s, l in leg.items)
    n = int(max(1, min(len(leg.items), width // max(w, 1))))
    if n != leg.columnCount:
        leg.setColumnCount(n)


class Twin:
    """Second y axis on the right of a PlotItem, with its own ViewBox (x linked)."""

    def __init__(self, pi):
        self.pi = pi
        self.vb = pg.ViewBox()
        pi.scene().addItem(self.vb)
        ax = pi.getAxis("right")
        ax.linkToView(self.vb)
        self.vb.setXLink(pi)
        pi.vb.sigResized.connect(self._sync)
        self._sync()

    def _sync(self):
        self.vb.setGeometry(self.pi.vb.sceneBoundingRect())
        self.vb.linkedViewChanged(self.pi.vb, self.vb.XAxis)

    def show(self, on, label=""):
        ax = self.pi.getAxis("right")
        ax.setStyle(showValues=on)
        axis_label(self.pi, "right", label) if on else ax.showLabel(False)
        self.vb.setVisible(on)

    def clear(self):
        self.vb.clear()


def plot(title=None, x=None, y=None, invert_y=False, legend=False, aspect=False):
    w = pg.PlotWidget()
    pi = w.getPlotItem()
    style(pi, title, x, y)
    if invert_y:
        pi.invertY(True)
    if aspect:
        pi.setAspectLocked(True)
    if legend:
        side_legend(pi)
    return w


def legend_color(leg, c):
    """Recolour a legend's labels (pyqtgraph does not redraw existing labels by itself)."""
    leg.setLabelTextColor(c)
    for _, label in leg.items:
        label.setText(label.text, color=c)


def side_legend(pi, row=2, col=3):
    """Legend in its own column to the right of the plot area (never covers data). Curves added
    with name=... register in it like with addLegend()."""
    old = getattr(pi, "legend", None)
    if old is not None and getattr(old, "_side", False):
        return old
    if old is not None and old.scene() is not None:
        old.scene().removeItem(old)
    leg = pg.LegendItem(labelTextSize="8.5pt", labelTextColor=theme.current["plot_fg"], verSpacing=-6,
                        horSpacing=6)
    leg._side = True
    leg.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
    pi.layout.addItem(leg, row, col)
    pi.layout.setAlignment(leg, Qt.AlignTop | Qt.AlignLeft)
    pi.legend = leg
    return leg



def style(pi, title=None, x=None, y=None):
    t = theme.current
    if title:
        pi.setTitle(f"<span style='font-size:9pt;color:{t['plot_fg']}'>{title}</span>")
        pi.titleLabel.setMinimumWidth(0)      # long titles never force the plot wider
    nice_axes(pi)
    if x:
        axis_label(pi, "bottom", x)
    if y:
        axis_label(pi, "left", y)
    pi.showGrid(x=True, y=True, alpha=t["grid"] / 255)
    for ax in ("left", "bottom", "right"):
        pi.getAxis(ax).enableAutoSIPrefix(False)
        pi.getAxis(ax).setPen(pg.mkPen(t["plot_fg"]))
        pi.getAxis(ax).setTextPen(pg.mkPen(t["plot_fg"]))


class ImagePanel(pg.GraphicsLayoutWidget):
    """Image with physical axes and a colour bar."""

    def __init__(self, title=None, x="x (mm)", y="z (mm)", cmap="inferno", invert_y=True,
                 bipolar=False):
        super().__init__()
        self.pi = self.addPlot(row=0, col=0)
        style(self.pi, title, x, y)
        self.img = pg.ImageItem()
        self.pi.addItem(self.img)
        if invert_y:
            self.pi.invertY(True)
        self.pi.setAspectLocked(True)
        self.cmap_name = "CET-D1" if bipolar else cmap
        self.bipolar = bipolar
        self.bar = pg.ColorBarItem(colorMap=theme.cmap(self.cmap_name), width=12,
                                   interactive=False)
        self.bar.setImageItem(self.img, insert_in=self.pi)
        self.overlays = []

    def set(self, data, x, y, levels=None, title=None):
        data = np.asarray(data, float)
        if levels is None:
            if self.bipolar:
                m = np.nanmax(np.abs(data)) or 1.0
                levels = (-m, m)
            else:
                lo, hi = np.nanmin(data), np.nanmax(data)
                levels = (lo, hi if hi > lo else lo + 1)
        self.img.setImage(data, autoLevels=False, levels=levels)
        dx = (x[-1] - x[0]) / max(len(x) - 1, 1)
        dy = (y[-1] - y[0]) / max(len(y) - 1, 1)
        self.img.setRect(QtCore.QRectF(x[0] - dx / 2, y[0] - dy / 2, x[-1] - x[0] + dx,
                                       y[-1] - y[0] + dy))
        self.bar.setLevels(levels)
        if title:
            style(self.pi, title)

    def clear_overlays(self):
        for it in self.overlays:
            self.pi.removeItem(it)
        self.overlays = []

    def add(self, item):
        self.pi.addItem(item); self.overlays.append(item)
        return item

    def set_cmap(self, name):
        self.cmap_name = name
        self.bar.setColorMap(theme.cmap(name))


def curve(pi, x, y, c=0, width=1.6, name=None, style=None, symbol=None):
    kw = {}
    if symbol:
        kw = dict(symbol=symbol, symbolSize=6, symbolBrush=theme.color(c), symbolPen=None)
    return pi.plot(x, y, pen=None if style == "none" else theme.pen(c, width, style), name=name,
                   **kw)


def vline(pi, x, c="gate", style="dash", label=None, pos=0.92):
    l = pg.InfiniteLine(pos=x, angle=90, pen=theme.pen(c, 1.2, style), label=label,
                        labelOpts=dict(position=pos, color=theme.color(c)))
    pi.addItem(l)
    return l


def hline(pi, y, c="gate", style="dash", label=None):
    l = pg.InfiniteLine(pos=y, angle=0, pen=theme.pen(c, 1.2, style), label=label,
                        labelOpts=dict(position=0.15, color=theme.color(c), anchors=[(0, 1), (0, 1)]))
    pi.addItem(l)
    return l


def text(pi, x, y, s, c=None, anchor=(0, 1)):
    t = pg.TextItem(s, color=c or theme.current["plot_fg"], anchor=anchor)
    t.setPos(x, y)
    pi.addItem(t)
    return t


def grab_png(widget, path):
    pm = widget.grab()
    pm.save(path, "PNG")


def fmt(v, unit="", digits=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if isinstance(v, (int, np.integer)):
        return f"{v} {unit}".strip()
    return f"{v:.{digits}g} {unit}".strip()


def log_ticks(pi, axis, lo, hi):
    """Clean 1-2-5 tick labels on a log-mode axis spanning [lo, hi] (data units)."""
    vals = [m * 10.0 ** e for e in range(int(np.floor(np.log10(lo))) - 1, int(np.ceil(np.log10(hi))) + 1)
            for m in (1, 2, 5) if lo <= m * 10.0 ** e <= hi]
    major = [(np.log10(v), f"{v:g}") for v in vals if abs(np.log10(v) % 1) < 1e-9]
    minor = [(np.log10(v), f"{v:g}") for v in vals if abs(np.log10(v) % 1) >= 1e-9]
    if len(major) >= 4:
        minor = [(t, "") for t, _ in minor]
    pi.getAxis(axis).setTicks([major, minor])
