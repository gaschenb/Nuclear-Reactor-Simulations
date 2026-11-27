"""Fission-product poisons (I-135 -> Xe-135, Pm-149 -> Sm-149) and decay-heat precursors.

Each cell carries its own chain, advanced exactly over a step with the flux held constant:

    dI/dt  = gI Sf phi - lI I
    dX/dt  = gX Sf phi + lI I - (lX + sX phi) X
    dP/dt  = gP Sf phi - lP P
    dS/dt  = lP P - sS phi S

Decay heat follows ANS-5.1 as a 23-group linear filter of the local fission rate, plus the U-239 and
Np-239 capture products, so it moves with the fuel when molten material relocates.
"""
from __future__ import annotations

import numpy as np

from .nuclear_data import (ANS_ALPHA, ANS_LAMBDA, E_NP239, E_U239, LAMBDA_I, LAMBDA_NP239,
                           LAMBDA_PM, LAMBDA_U239, LAMBDA_XE, MEV)


def _chain2(y0, y_inf, a, x0, x_inf, b, src_coef, t):
    """Solution of dy/dt = s + c x - a y with dx/dt = ... - b x relaxing x0 -> x_inf.
    Returns y(t) given its equilibrium y_inf (for fixed x_inf), and coupling coefficient c."""
    ea, eb = np.exp(-a * t), np.exp(-b * t)
    d = a - b
    small = np.abs(d) < 1e-12
    coup = np.where(small, src_coef * (x0 - x_inf) * t * ea,
                    src_coef * (x0 - x_inf) / np.where(small, 1.0, d) * (eb - ea))
    return y_inf + (y0 - y_inf) * ea + coup


class Poisons:
    def __init__(self, N, gI, gX, gP, sigma_xe, sigma_sm):
        self.gI, self.gX, self.gP = gI, gX, gP
        self.sx, self.ss = sigma_xe, sigma_sm
        self.I = np.zeros(N)
        self.X = np.zeros(N)
        self.P = np.zeros(N)
        self.S = np.zeros(N)

    def equilibrium(self, F, phi, samarium=True):
        """F = fission rate density (1/cm3 s), phi = thermal flux."""
        self.I = self.gI * F / LAMBDA_I
        self.X = (self.gX + self.gI) * F / (LAMBDA_XE + self.sx * phi)
        self.P = self.gP * F / LAMBDA_PM
        if samarium:
            self.S = np.where(phi > 0, self.gP * F / np.maximum(self.ss * phi, 1e-300), 0.0)

    def step(self, F, phi, dt):
        I0, X0, P0, S0 = self.I, self.X, self.P, self.S
        I_inf = self.gI * F / LAMBDA_I
        a = LAMBDA_XE + self.sx * phi
        X_inf = (self.gX * F + LAMBDA_I * I_inf) / a
        self.I = I_inf + (I0 - I_inf) * np.exp(-LAMBDA_I * dt)
        self.X = _chain2(X0, X_inf, a, I0, I_inf, LAMBDA_I, LAMBDA_I, dt)
        P_inf = self.gP * F / LAMBDA_PM
        self.P = P_inf + (P0 - P_inf) * np.exp(-LAMBDA_PM * dt)
        b = self.ss * phi
        lin = b * dt < 1e-9
        S_inf = np.where(lin, 0.0, LAMBDA_PM * P_inf / np.where(lin, 1.0, b))
        S_exp = _chain2(S0, S_inf, np.where(lin, 1.0, b), P0, P_inf, LAMBDA_PM, LAMBDA_PM, dt)
        S_lin = S0 + LAMBDA_PM * P_inf * dt + (P0 - P_inf) * (1 - np.exp(-LAMBDA_PM * dt))
        self.S = np.maximum(np.where(lin, S_lin, S_exp), 0.0)
        self.X = np.maximum(self.X, 0.0)

    def absorption(self):
        return self.sx * self.X, self.ss * self.S


class DecayHeat:
    """Per-cell ANS-5.1 groups. y[g, i] has units of fissions (integral of the filtered rate)."""

    def __init__(self, N, R_capture=0.6):
        self.y = np.zeros((len(ANS_LAMBDA), N))
        self.U = np.zeros(N)
        self.Np = np.zeros(N)
        self.R = R_capture
        self._lam = ANS_LAMBDA[:, None]

    def set_history(self, rate, t_op):
        """Constant fission rate (1/s per cell) for t_op seconds."""
        f = 1 - np.exp(-self._lam * t_op)
        self.y = rate[None, :] / self._lam * f
        lu, ln = LAMBDA_U239, LAMBDA_NP239
        self.U = self.R * rate / lu * (1 - np.exp(-lu * t_op))
        self.Np = self.R * rate / ln * (1 - np.exp(-ln * t_op))   # long-operation approximation

    def step(self, rate, dt):
        e = np.exp(-self._lam * dt)
        self.y = self.y * e + rate[None, :] / self._lam * (1 - e)
        lu, ln = LAMBDA_U239, LAMBDA_NP239
        U0 = self.U
        U_inf = self.R * rate / lu
        self.U = U_inf + (U0 - U_inf) * np.exp(-lu * dt)
        Np_inf = self.R * rate / ln
        self.Np = _chain2(self.Np, Np_inf, ln, U0, U_inf, lu, lu, dt)

    def power(self):
        """Decay power per cell (W)."""
        p = (ANS_ALPHA[:, None] * self.y).sum(0)
        p += E_U239 * LAMBDA_U239 * self.U + E_NP239 * LAMBDA_NP239 * self.Np
        return p * MEV

    def move(self, src, dst):
        """Relocate the decay-heat inventory of cell src into dst (dst < 0: removed)."""
        if dst >= 0:
            self.y[:, dst] += self.y[:, src]
            self.U[dst] += self.U[src]
            self.Np[dst] += self.Np[src]
        self.y[:, src] = 0.0
        self.U[src] = 0.0
        self.Np[src] = 0.0
