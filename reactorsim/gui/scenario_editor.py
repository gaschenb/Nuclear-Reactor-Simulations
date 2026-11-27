"""Scenario editor: duration, scenario parameters, the event schedule (times, settings, add /
remove), protection and controller switches, initial power and flow, lattice resolution.
Produces a modified copy of the built-in scenario; the built-ins are never changed."""
from __future__ import annotations

from dataclasses import replace

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

from . import widgets as W
from ..core.reactors import REACTORS
from ..core.scenarios import SCENARIOS, Event

EVENT_KINDS = {
    "scram": ("reactor trip (scram)", {}),
    "pump_trip": ("coolant pumps trip", {}),
    "pump_restart": ("coolant pumps restart", dict(frac=1.0)),
    "flow_ramp": ("pump flow change", dict(frac=0.5, tau=10.0)),
    "sbo": ("station blackout (loss of AC)", {}),
    "ac_restore": ("AC power restored", {}),
    "loss_of_feed": ("main feedwater lost", {}),
    "aux_feed": ("auxiliary feedwater", dict(on=True)),
    "break": ("primary break", dict(area=0.01, loc="liquid")),
    "depressurize": ("open relief valves", dict(area=0.005)),
    "eccs": ("injection system on/off", dict(name="HPI", on=True)),
    "inject": ("extra injection", dict(flow=20.0, p_max=1.0, T=300.0)),
    "turbine_trip": ("turbine trip", dict(bypass=0.0)),
    "msiv_close": ("steam isolation valves close", {}),
    "rod_bank": ("move rod bank", dict(name="", depth=0.5, speed=0.01)),
    "rod_eject": ("rod ejection", dict(time=0.1)),
    "setpoint": ("power setpoint", dict(power=0.5)),
    "rps": ("automatic protection on/off", dict(on=True)),
    "stop": ("end the run", {}),
}
UNITS = [("s", 1.0), ("min", 60.0), ("h", 3600.0)]


def _fmt_arg(v):
    if isinstance(v, bool):
        return "on" if v else "off"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _parse_args(text):
    out = {}
    for part in text.replace(";", ",").split(","):
        if "=" not in part:
            continue
        k, v = (s.strip() for s in part.split("=", 1))
        if not k:
            continue
        lv = v.lower()
        if lv in ("on", "true", "yes"):
            out[k] = True
        elif lv in ("off", "false", "no"):
            out[k] = False
        elif v.startswith("$"):
            out[k] = v
        else:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


class ScenarioEditor(QtWidgets.QWidget):
    changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sc = None
        v = QtWidgets.QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(8)
        # duration
        g = QtWidgets.QGridLayout(); g.setHorizontalSpacing(8); g.setVerticalSpacing(5)
        g.addWidget(QtWidgets.QLabel("duration"), 0, 0)
        self.sp_dur = QtWidgets.QDoubleSpinBox(); self.sp_dur.setDecimals(2); self.sp_dur.setRange(0.01, 1e6)
        self.sp_dur.setKeyboardTracking(False)
        self.cb_unit = QtWidgets.QComboBox()
        for u, _ in UNITS:
            self.cb_unit.addItem(u)
        self.cb_unit.currentIndexChanged.connect(self._unit_changed)
        g.addWidget(self.sp_dur, 0, 1); g.addWidget(self.cb_unit, 0, 2)
        g.addWidget(QtWidgets.QLabel("initial power"), 1, 0)
        self.sp_p0 = QtWidgets.QDoubleSpinBox(); self.sp_p0.setRange(0.0, 150.0); self.sp_p0.setSuffix(" % rated")
        self.sp_p0.setDecimals(6)
        self.sp_p0.setToolTip("hot zero power is ~10⁻⁴ % of rated")
        g.addWidget(self.sp_p0, 1, 1, 1, 2)
        g.addWidget(QtWidgets.QLabel("initial flow"), 2, 0)
        self.sp_f0 = QtWidgets.QDoubleSpinBox(); self.sp_f0.setRange(0.0, 150.0); self.sp_f0.setSuffix(" % rated")
        g.addWidget(self.sp_f0, 2, 1, 1, 2)
        g.addWidget(QtWidgets.QLabel("lattice (columns × levels)"), 3, 0)
        self.sp_nx = QtWidgets.QSpinBox(); self.sp_nx.setRange(4, 48)
        self.sp_nz = QtWidgets.QSpinBox(); self.sp_nz.setRange(4, 48)
        g.addWidget(self.sp_nx, 3, 1); g.addWidget(self.sp_nz, 3, 2)
        g.setColumnStretch(1, 1)
        v.addLayout(g)
        self.chk_rps = QtWidgets.QCheckBox("automatic reactor trip (protection system)")
        self.chk_ctrl = QtWidgets.QCheckBox("power controller holds the setpoint")
        v.addWidget(self.chk_rps); v.addWidget(self.chk_ctrl)
        # scenario parameters
        self.param_box = QtWidgets.QVBoxLayout(); v.addLayout(self.param_box)
        self.params = None
        # events
        lab = QtWidgets.QLabel("Event schedule"); lab.setObjectName("cardTitle")
        v.addWidget(lab)
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["time", "event", "settings"])
        self.table.verticalHeader().setVisible(False)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        hh.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.setMinimumHeight(150)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setToolTip("Double-click a time or the settings to edit them. Untick an event to skip it.")
        self.table.itemChanged.connect(lambda _: self.changed.emit())
        v.addWidget(self.table)
        row = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QToolButton(); self.btn_add.setText("Add event ▾")
        self.btn_add.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        menu = QtWidgets.QMenu(self.btn_add)
        for k, (name, _) in EVENT_KINDS.items():
            menu.addAction(name, lambda k=k: self.add_event(k))
        self.btn_add.setMenu(menu)
        b_rm = QtWidgets.QPushButton("Remove"); b_rm.clicked.connect(self.remove_selected)
        b_reset = QtWidgets.QPushButton("Reset"); b_reset.clicked.connect(lambda: self.set_scenario(self.sc.key))
        row.addWidget(self.btn_add); row.addWidget(b_rm); row.addStretch(1); row.addWidget(b_reset)
        v.addLayout(row)
        for w in (self.sp_dur, self.sp_p0, self.sp_f0, self.sp_nx, self.sp_nz):
            w.valueChanged.connect(lambda *_: self.changed.emit())
        for w in (self.chk_rps, self.chk_ctrl):
            w.toggled.connect(lambda *_: self.changed.emit())

    # ------------------------------------------------------------------ load
    def unit(self):
        return UNITS[self.cb_unit.currentIndex()][1]

    def set_scenario(self, key):
        sc = SCENARIOS[key]
        self.sc = sc
        rt = REACTORS[sc.reactor]
        p0 = {k: v[0] for k, v in sc.params.items()}
        t_end = float(sc.value(sc.t_end, p0))
        ui = 2 if t_end > 6 * 3600 else (1 if t_end > 600 else 0)
        self.cb_unit.blockSignals(True); self.cb_unit.setCurrentIndex(ui); self.cb_unit.blockSignals(False)
        self._prev_unit = self.unit()
        self.sp_dur.blockSignals(True); self.sp_dur.setValue(t_end / self.unit()); self.sp_dur.blockSignals(False)
        for w, val in ((self.sp_p0, sc.power0 * 100), (self.sp_f0, sc.flow0 * 100), (self.sp_nx, rt.nx),
                       (self.sp_nz, rt.nz)):
            w.blockSignals(True); w.setValue(val); w.blockSignals(False)
        for w, val in ((self.chk_rps, sc.rps), (self.chk_ctrl, sc.controller)):
            w.blockSignals(True); w.setChecked(val); w.blockSignals(False)
        # scenario parameters (duration handled above)
        if self.params is not None:
            self.param_box.removeWidget(self.params)
            self.params.hide(); self.params.setParent(None); self.params.deleteLater()
        specs = []
        for k, (d, lo, hi, *lab) in sc.params.items():
            if k == "t_end":
                continue
            label = lab[0] if lab else k
            scale, unit = (3600.0, "h") if "(s)" in label and hi >= 4 * 3600 else (1.0, "")
            disp = label.replace("(s)", "(h)") if unit == "h" else label
            specs.append(W.P(k, disp, lo / scale, hi / scale, d / scale, scale=scale, log=False, decimals=4))
        self.params = W.ParamPanel(specs)
        self.params.changed.connect(lambda _: (self._refresh_linked(), self.changed.emit()))
        self.param_box.addWidget(self.params)
        self.params.setVisible(bool(specs))
        # events
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for e in sc.events:
            self._add_row(e.t, e.kind, dict(e.args))
        self.table.blockSignals(False)
        self._refresh_linked()
        self.changed.emit()

    def _add_row(self, t, kind, args, enabled=True):
        r = self.table.rowCount()
        self.table.insertRow(r)
        linked = isinstance(t, str)
        it = QtWidgets.QTableWidgetItem()
        it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable
                    | (Qt.NoItemFlags if linked else Qt.ItemIsEditable))
        it.setCheckState(Qt.Checked if enabled else Qt.Unchecked)
        it.setData(Qt.UserRole, t)
        if linked:
            f = it.font(); f.setItalic(True); it.setFont(f)
            it.setToolTip(f"set by the scenario parameter '{t[1:]}' above")
        else:
            it.setText(f"{float(t) / self.unit():g}")
        self.table.setItem(r, 0, it)
        name = EVENT_KINDS.get(kind, (kind, {}))[0]
        ik = QtWidgets.QTableWidgetItem(name); ik.setData(Qt.UserRole, kind)
        ik.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        self.table.setItem(r, 1, ik)
        ia = QtWidgets.QTableWidgetItem(", ".join(f"{k}={_fmt_arg(v)}" for k, v in args.items()))
        ia.setToolTip("comma-separated settings, e.g. frac=0.5, tau=10; '$name' links to a scenario parameter")
        self.table.setItem(r, 2, ia)
        return r

    def _refresh_linked(self):
        if self.sc is None:
            return
        vals = self.param_values()
        self.table.blockSignals(True)
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            t = it.data(Qt.UserRole)
            if isinstance(t, str):
                v = vals.get(t[1:], 0.0)
                it.setText(f"{float(v) / self.unit():g}")
        self.table.blockSignals(False)

    def _unit_changed(self, _):
        old, new = getattr(self, "_prev_unit", 1.0), self.unit()
        self.sp_dur.blockSignals(True); self.sp_dur.setValue(self.sp_dur.value() * old / new)
        self.sp_dur.blockSignals(False)
        self.table.blockSignals(True)
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if not isinstance(it.data(Qt.UserRole), str):
                try:
                    it.setText(f"{float(it.text()) * old / new:g}")
                except ValueError:
                    pass
        self.table.blockSignals(False)
        self._prev_unit = new
        self._refresh_linked()
        self.table.setHorizontalHeaderLabels([f"time ({self.cb_unit.currentText()})", "event", "settings"])

    def add_event(self, kind):
        args = dict(EVENT_KINDS[kind][1])
        if kind == "rod_bank" and self.sc is not None and self.sc.banks:
            args["name"] = next(iter(self.sc.banks))
        self.table.blockSignals(True)
        r = self._add_row(0.0, kind, args)
        self.table.blockSignals(False)
        self.table.selectRow(r)
        self.table.editItem(self.table.item(r, 0))
        self.changed.emit()

    def remove_selected(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)
        self.changed.emit()

    # ------------------------------------------------------------------ output
    def param_values(self):
        out = self.params.values() if self.params is not None else {}
        out["t_end"] = self.sp_dur.value() * self.unit()
        return out

    def scenario(self):
        """The edited scenario (a copy) and its parameter values."""
        sc = self.sc
        evs = []
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if it.checkState() != Qt.Checked:
                continue
            t = it.data(Qt.UserRole)
            if not isinstance(t, str):
                try:
                    t = float(it.text()) * self.unit()
                except ValueError:
                    continue
            kind = self.table.item(r, 1).data(Qt.UserRole)
            evs.append(Event(t, kind, _parse_args(self.table.item(r, 2).text())))
        params = dict(sc.params)
        if "t_end" not in params:
            params["t_end"] = (float(sc.value(sc.t_end, {k: v[0] for k, v in sc.params.items()})), 1.0, 1e7,
                               "duration (s)")
        new = replace(sc, events=evs, params=params, t_end="$t_end", rps=self.chk_rps.isChecked(),
                      controller=self.chk_ctrl.isChecked(), power0=self.sp_p0.value() / 100,
                      flow0=self.sp_f0.value() / 100)
        return new, self.param_values()

    def lattice(self):
        rt = REACTORS[self.sc.reactor]
        nx, nz = self.sp_nx.value(), self.sp_nz.value()
        return (None if nx == rt.nx else nx), (None if nz == rt.nz else nz)

    def summary(self):
        """Short description of what differs from the built-in scenario."""
        if self.sc is None:
            return ""
        new, p = self.scenario()
        sc = self.sc
        d = []
        base_t = float(sc.value(sc.t_end, {k: v[0] for k, v in sc.params.items()}))
        if abs(p["t_end"] - base_t) > 1e-6 * base_t:
            d.append("duration")
        if new.rps != sc.rps or new.controller != sc.controller:
            d.append("protection / controller")
        if abs(new.power0 - sc.power0) > 1e-3 * sc.power0 + 1e-8 or abs(new.flow0 - sc.flow0) > 1e-4:
            d.append("initial state")
        if [(e.t, e.kind, e.args) for e in new.events] != [(e.t, e.kind, dict(e.args)) for e in sc.events]:
            d.append("events")
        if self.lattice() != (None, None):
            d.append("lattice")
        return ", ".join(d)
