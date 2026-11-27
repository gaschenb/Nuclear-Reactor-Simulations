"""Lattice fields shown in the maps: physical quantity, unit, display scaling and colour map."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Field:
    key: str
    name: str
    unit: str
    cmap: str | None = "inferno"
    scale: float = 1.0          # displayed value = stored value × scale
    log: bool = False           # default to a logarithmic colour scale
    group: str = ""
    digits: int = 4

    @property
    def label(self):
        return f"{self.name} ({self.unit})" if self.unit else self.name


FIELDS = [
    Field("q3", "power density", "W/cm³", group="Power"),
    Field("q1", "linear heat rate", "kW/m", group="Power"),
    Field("power", "relative power", "× core average", group="Power"),
    Field("flux", "neutron flux", "n/cm²·s", "viridis", log=True, group="Power", digits=3),
    Field("T_fuel", "fuel temperature", "K", group="Temperature"),
    Field("T_clad", "cladding temperature", "K", group="Temperature"),
    Field("T_cool", "coolant temperature", "K", "viridis", group="Temperature"),
    Field("T_mod", "moderator temperature", "K", "viridis", group="Temperature"),
    Field("enthalpy", "fuel enthalpy", "cal/g", group="Temperature", digits=3),
    Field("void", "void fraction", "%", "CET-L6", group="Coolant", digits=3),
    Field("Xe", "Xe-135", "10¹⁵ atoms/cm³", "magma", scale=1e-15, group="Poisons", digits=3),
    Field("I", "I-135", "10¹⁵ atoms/cm³", "magma", scale=1e-15, group="Poisons", digits=3),
    Field("Sm", "Sm-149", "10¹⁵ atoms/cm³", "magma", scale=1e-15, group="Poisons", digits=3),
    Field("xenon", "Xe-135 share of absorption", "%", "magma", group="Poisons", digits=3),
    Field("state", "damage state", "", None, group="Damage"),
    Field("hazard", "damage transition rate λ", "1/s", "plasma", log=True, group="Damage", digits=3),
    Field("ecr", "cladding oxidised", "% ECR", "plasma", group="Damage", digits=3),
    Field("fuel", "fuel in cell", "kg", "cividis", group="Damage"),
    Field("release", "volatile fission products released", "%", "plasma", group="Damage", digits=3),
]
BY_KEY = {f.key: f for f in FIELDS}
STATE_NAMES = ["intact", "clad failed", "damaged", "molten", "relocated"]


def fmt(f: Field, v):
    """Value with its unit, for read-outs."""
    import math
    if f.key == "state":
        return STATE_NAMES[int(v)] if v == v else "—"
    if v != v:
        return "—"
    v = v * f.scale
    if v != 0 and (abs(v) >= 1e5 or abs(v) < 1e-3):
        s = f"{v:.{f.digits - 1}e}"
        m, e = s.split("e")
        s = f"{m}×10" + f"{int(e)}".translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    else:
        d = max(0, f.digits - 1 - int(math.floor(math.log10(abs(v))))) if v != 0 else 0
        s = f"{v:.{min(d, 4)}f}"
    return f"{s} {f.unit}".strip()
