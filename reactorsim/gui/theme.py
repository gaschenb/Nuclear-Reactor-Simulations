"""Light / dark themes: Qt palette, style sheet and pyqtgraph defaults."""
from __future__ import annotations

import pyqtgraph as pg
from PyQt5 import QtGui

THEMES = {
    "light": dict(
        bg="#f3f4f6", panel="#ffffff", border="#e2e5ea", text="#1f2937", muted="#6b7280",
        accent="#c2410c", accent_hover="#9a3412", accent_text="#ffffff", input="#ffffff",
        sel="#ffedd5", nav="#0f172a", nav_text="#cbd5e1", nav_sel="#1e293b",
        plot_bg="#ffffff", plot_fg="#374151", grid=60, eq="#0f172a", eq_bg="#f8fafc"),
    "dark": dict(
        bg="#14161b", panel="#1c1f26", border="#2c313a", text="#e5e7eb", muted="#9ca3af",
        accent="#f97316", accent_hover="#fb923c", accent_text="#140b05", input="#22262e",
        sel="#431407", nav="#0a0c10", nav_text="#9ca3af", nav_sel="#1f2937",
        plot_bg="#181b21", plot_fg="#d1d5db", grid=50, eq="#e2e8f0", eq_bg="#171a20"),
}

# series colours (readable on both backgrounds)
SERIES = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#76b7b2", "#edc948",
          "#9c755f", "#ff9da7", "#8c8c8c"]
NAMED = {"power": "#4e79a7", "decay": "#f28e2b", "limit": "#e15759", "ref": "#8c8c8c",
         "fuel": "#e15759", "clad": "#f28e2b", "cool": "#4e79a7", "mod": "#59a14f",
         "xenon": "#b07aa1", "iodine": "#76b7b2", "samarium": "#9c755f", "h2": "#e15759"}
# cell damage states: intact, failed, damaged, molten, relocated
STATE_COLORS = ["#4e79a7", "#edc948", "#f28e2b", "#e15759", "#7f3b08"]

current = dict(THEMES["light"], name="light")


def color(i_or_name):
    if isinstance(i_or_name, str):
        if i_or_name == "fg":
            return current["plot_fg"]
        return NAMED.get(i_or_name, i_or_name)
    return SERIES[i_or_name % len(SERIES)]


def pen(i_or_name, width=1.6, style=None):
    from PyQt5.QtCore import Qt
    p = pg.mkPen(color(i_or_name), width=width)
    if style == "dash":
        p.setStyle(Qt.DashLine)
    elif style == "dot":
        p.setStyle(Qt.DotLine)
    return p


def fg_pen(width=1.2, style=None):
    from PyQt5.QtCore import Qt
    p = pg.mkPen(current["plot_fg"], width=width)
    if style == "dash":
        p.setStyle(Qt.DashLine)
    return p


def cmap(name="inferno"):
    if name.endswith("_r"):
        cm = pg.colormap.get(name[:-2])
        cm.reverse()
        return cm
    return pg.colormap.get(name)


def style_sheet(t):
    return f"""
    QWidget {{ color: {t['text']}; font-size: 9pt; }}
    QMainWindow, QStackedWidget, #page {{ background: {t['bg']}; }}
    QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; border: none; }}
    #card {{ background: {t['panel']}; border: 1px solid {t['border']}; border-radius: 8px; }}
    #cardTitle {{ font-weight: 600; font-size: 9pt; color: {t['text']}; }}
    #muted {{ color: {t['muted']}; }}
    #equation {{ background: {t['eq_bg']}; border: 1px solid {t['border']}; border-radius: 6px;
                 padding: 8px 10px; color: {t['eq']}; }}
    #readout {{ color: {t['muted']}; font-family: Consolas, 'DejaVu Sans Mono', monospace; }}
    #nav {{ background: {t['nav']}; border: none; outline: none; padding-top: 6px; }}
    #nav::item {{ color: {t['nav_text']}; padding: 9px 14px; border-left: 3px solid transparent; }}
    #nav::item:selected {{ background: {t['nav_sel']}; color: #ffffff;
                           border-left: 3px solid {t['accent']}; }}
    #nav::item:hover:!selected {{ background: {t['nav_sel']}; }}
    #brand {{ background: {t['nav']}; color: #ffffff; font-weight: 700; font-size: 11pt;
              padding: 14px 14px 2px 14px; }}
    #brandSub {{ background: {t['nav']}; color: {t['nav_text']}; font-size: 8pt;
                 padding: 0 14px 10px 14px; }}
    QPushButton {{ background: {t['panel']}; border: 1px solid {t['border']}; border-radius: 6px;
                   padding: 5px 12px; }}
    QPushButton:hover {{ border-color: {t['accent']}; }}
    QPushButton:disabled {{ color: {t['muted']}; }}
    QPushButton#primary {{ background: {t['accent']}; color: {t['accent_text']}; border: none;
                           font-weight: 600; padding: 7px 14px; }}
    QPushButton#primary:hover {{ background: {t['accent_hover']}; }}
    QToolBar {{ background: {t['panel']}; border: none; border-bottom: 1px solid {t['border']};
                spacing: 4px; padding: 4px 8px; }}
    QToolButton {{ border: 1px solid transparent; border-radius: 6px; padding: 4px 9px; }}
    QToolButton:hover {{ background: {t['bg']}; border-color: {t['border']}; }}
    QToolButton:checked {{ background: {t['sel']}; border-color: {t['accent']}; }}
    QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{ background: {t['input']};
        border: 1px solid {t['border']}; border-radius: 5px; padding: 3px 6px; min-height: 20px; }}
    QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{
        border-color: {t['accent']}; }}
    QComboBox QAbstractItemView {{ background: {t['panel']}; selection-background-color: {t['sel']};
        selection-color: {t['text']}; border: 1px solid {t['border']}; }}
    QListWidget {{ background: {t['input']}; border: 1px solid {t['border']}; border-radius: 5px; }}
    QListWidget::item:selected {{ background: {t['sel']}; color: {t['text']}; }}
    QTabWidget::pane {{ border: 1px solid {t['border']}; border-radius: 6px; background: {t['panel']};
                        top: -1px; }}
    QTabBar::tab {{ background: transparent; padding: 6px 14px; border: none;
                    border-bottom: 2px solid transparent; color: {t['muted']}; }}
    QTabBar::tab:selected {{ color: {t['text']}; border-bottom: 2px solid {t['accent']}; }}
    QTabBar::tab:hover:!selected {{ color: {t['text']}; }}
    QTableWidget {{ background: {t['panel']}; gridline-color: {t['border']};
                    border: 1px solid {t['border']}; border-radius: 6px;
                    selection-background-color: {t['sel']}; selection-color: {t['text']}; }}
    QHeaderView::section {{ background: {t['bg']}; border: none; border-bottom: 1px solid {t['border']};
                            padding: 4px 6px; font-weight: 600; }}
    QSlider::groove:horizontal {{ height: 4px; background: {t['border']}; border-radius: 2px; }}
    QSlider::sub-page:horizontal {{ background: {t['accent']}; border-radius: 2px; }}
    QSlider::handle:horizontal {{ background: {t['panel']}; border: 2px solid {t['accent']};
                                  width: 12px; margin: -6px 0; border-radius: 7px; }}
    QProgressBar {{ border: 1px solid {t['border']}; border-radius: 4px; background: {t['bg']};
                    max-height: 8px; text-align: center; font-size: 1px; }}
    QProgressBar::chunk {{ background: {t['accent']}; border-radius: 3px; }}
    QStatusBar {{ background: {t['panel']}; border-top: 1px solid {t['border']}; color: {t['muted']}; }}
    QSplitter::handle {{ background: {t['bg']}; }}
    QCheckBox {{ spacing: 6px; }}
    QToolTip {{ background: {t['panel']}; color: {t['text']}; border: 1px solid {t['border']}; }}
    """


def apply(app, name="light"):
    t = THEMES[name]
    current.clear(); current.update(t); current["name"] = name
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    for role, key in [(QtGui.QPalette.Window, "bg"), (QtGui.QPalette.Base, "input"),
                      (QtGui.QPalette.AlternateBase, "bg"), (QtGui.QPalette.Button, "panel"),
                      (QtGui.QPalette.Text, "text"), (QtGui.QPalette.WindowText, "text"),
                      (QtGui.QPalette.ButtonText, "text"), (QtGui.QPalette.Highlight, "accent"),
                      (QtGui.QPalette.ToolTipBase, "panel"), (QtGui.QPalette.ToolTipText, "text")]:
        pal.setColor(role, QtGui.QColor(t[key]))
    pal.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#ffffff"))
    app.setPalette(pal)
    app.setStyleSheet(style_sheet(t))
    pg.setConfigOptions(antialias=True, background=t["plot_bg"], foreground=t["plot_fg"],
                        imageAxisOrder="row-major")
