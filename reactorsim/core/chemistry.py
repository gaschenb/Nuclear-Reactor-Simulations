"""Zircaloy-steam oxidation, hydrogen and fission-product release.

  Oxidation     parabolic kinetics d(w^2)/dt = K(T) for consumed Zr per unit area w (kg/m2):
                Cathcart-Pawel (ORNL/NUREG-17, 1977) below 1853 K, Baker-Just (ANL-6548, 1962) above.
                Zr + 2 H2O -> ZrO2 + 2 H2,  586 kJ/mol Zr released (6.42 MJ/kg Zr),
                0.0442 kg H2 and 0.395 kg steam per kg Zr. Steam starvation limits the rate in
                uncovered channels to the steam rising from the boiling pool below.
  FP release    gap inventory released at clad failure (NUREG-1465 gap phase: 5 % of noble gases,
                halogens and alkali metals) and CORSOR-M diffusion release from hot fuel,
                k = 2.0e5 exp(-63.8 kcal/mol / RT) per minute (NUREG/CR-4173).
"""
from __future__ import annotations

import numpy as np

R_GAS = 8.314
Q_OX = 6.42e6           # J per kg Zr
H2_PER_ZR = 2 * 2.016 / 91.224
STEAM_PER_ZR = 2 * 18.015 / 91.224


def k_parabolic(T):
    """Parabolic rate constant for consumed Zr, (kg/m2)^2 / s."""
    T = np.asarray(T, float)
    cp = 0.3622 * np.exp(-1.6710e5 / (R_GAS * T)) * (91.224 / 32.0) ** 2 * 100.0
    bj = 33.3e6 * np.exp(-45500.0 / (1.987 * T)) * 1e-4
    return np.where(T < 1853.0, cp, bj) * (T > 900.0)


def oxidize(w, T, dt, w_max, area, steam_limit=None):
    """Advance consumed-Zr areal mass w (kg/m2) over dt. area (m2) per cell; w_max = available Zr.
    steam_limit: optional array of max Zr consumption (kg) allowed in each cell over the step.
    Returns new w and the Zr mass consumed per cell (kg)."""
    w_new = np.minimum(np.sqrt(w ** 2 + k_parabolic(T) * dt), w_max)
    dZr = np.maximum(w_new - w, 0.0) * area
    if steam_limit is not None:
        dZr = np.minimum(dZr, steam_limit)
        w_new = w + dZr / np.maximum(area, 1e-12)
    return w_new, dZr


def corsor_rate(T):
    """CORSOR-M fractional release rate (1/s) for Cs, I and noble gases."""
    T = np.asarray(T, float)
    return 2.0e5 / 60.0 * np.exp(-63.8e3 / (1.987 * np.maximum(T, 300.0)))


GAP_FRACTION = 0.05
