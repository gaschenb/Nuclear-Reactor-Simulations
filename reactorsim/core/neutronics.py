"""Lattice neutronics: one-group nodal diffusion on the (x, z) slice and point kinetics.

Flux shape   -div(D grad phi) + (Sigma_a + D B_y^2) phi = (1/k) nu Sigma_f phi on an nx x nz mesh,
             zero-flux at the extrapolated boundary (reflector savings) or explicit reflector cells.
Amplitude    six-group point kinetics, advanced exactly over each step with a matrix exponential
             (quasi-static method). Between shape refreshes the reactivity is updated with
             first-order perturbation theory, rho = rho_0 + <phi,(lambda dF - dSigma_a) phi>/<phi,F phi>.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import expm


class Lattice:
    """Geometry of the slice. Cells are indexed i = iz * nx + ix, iz = 0 at the bottom."""

    def __init__(self, rt):
        self.rt = rt
        self.nx, self.nz = rt.nx, rt.nz
        self.N = self.nx * self.nz
        self.H = rt.height_m * 100.0
        self.W = rt.width_m * 100.0
        self.dx = self.W / self.nx
        self.dz = self.H / self.nz
        self.ix = np.tile(np.arange(self.nx), self.nz)
        self.iz = np.repeat(np.arange(self.nz), self.nx)
        self.xc = (self.ix + 0.5) * self.dx - self.W / 2
        self.zc = (self.iz + 0.5) * self.dz
        cols = rt.fuel_columns()
        self.fuel = cols[self.ix]
        self.n_fuel = int(self.fuel.sum())
        self.n_fuel_cols = int(cols.sum())
        # transverse (y) extent: equal-area square of the cylinder, CANDU: channel length
        self.Wy = (rt.width_m * 100.0) * np.sqrt(np.pi) / 2
        if rt.flow_direction == "cross":
            self.Wy = 594.0
        self.V_core = np.pi / 4 * rt.width_m ** 2 * rt.height_m if rt.flow_direction != "cross" \
            else np.pi / 4 * rt.width_m ** 2 * 5.94                      # m3
        # cells above / below
        self.below = np.where(self.iz > 0, np.arange(self.N) - self.nx, -1)
        self.above = np.where(self.iz < self.nz - 1, np.arange(self.N) + self.nx, -1)

    def grid(self, v):
        return np.asarray(v).reshape(self.nz, self.nx)


class Diffusion:
    def __init__(self, lat: Lattice, D: np.ndarray, delta_cm: float):
        self.lat = lat
        nx, nz, N = lat.nx, lat.nz, lat.N
        dx, dz = lat.dx, lat.dz
        By2 = (np.pi / (lat.Wy + 2 * delta_cm)) ** 2
        self.D = D
        rows, cols, vals = [], [], []
        diag = D * By2
        idx = np.arange(N).reshape(nz, nx)

        def couple(a, b, h):
            Dh = 2 * D[a] * D[b] / (D[a] + D[b]) / h ** 2
            rows.extend(a); cols.extend(b); vals.extend(-Dh)
            rows.extend(b); cols.extend(a); vals.extend(-Dh)
            np.add.at(diag, a, Dh); np.add.at(diag, b, Dh)

        couple(idx[:, :-1].ravel(), idx[:, 1:].ravel(), dx)
        couple(idx[:-1, :].ravel(), idx[1:, :].ravel(), dz)
        # extrapolated-boundary leakage: D / (h/2 + delta) per face
        for edge, h in ((idx[:, 0], dx), (idx[:, -1], dx), (idx[0, :], dz), (idx[-1, :], dz)):
            e = edge.ravel()
            diag[e] += D[e] / (h / 2 + max(delta_cm, 1e-3)) / h
        self.L = sp.csc_matrix((vals, (rows, cols)), shape=(N, N)) + sp.diags(diag)

    def eigen(self, nuSf, Sa, phi0=None, k0=1.0, tol=1e-10, maxit=200):
        """Fundamental mode by Wielandt-shifted inverse iteration, residual-controlled.

        Large or annular cores have nearly degenerate first harmonics, so the shift is tightened
        in stages around the Rayleigh-quotient estimate (each stage converges in a few iterations
        once the shift is much closer to the fundamental than the harmonic gap).
        Returns k and the max-normalised flux."""
        N = self.lat.N
        A = (self.L + sp.diags(Sa)).tocsc()
        F = nuSf
        phi = np.ones(N) if phi0 is None else np.maximum(phi0, 1e-12)
        lam = float(phi @ (A @ phi)) / max(float(phi @ (F * phi)), 1e-300)
        if not np.isfinite(lam) or lam <= 0:
            lam = 1.0 / k0
        best = None
        res = np.inf
        for shift in (2e-2, 2e-3, 1e-4, 3e-6):
            mu = lam - shift
            try:
                lu = spla.splu((A - sp.diags(mu * F)).tocsc())
            except RuntimeError:
                continue
            for it in range(maxit):
                psi = lu.solve(F * phi)
                if psi.sum() < 0:
                    psi = -psi
                nrm = np.abs(psi).max()
                if not np.isfinite(nrm) or nrm == 0:
                    break
                psi /= nrm
                Ap = A @ psi
                Fp = F * psi
                lam_new = float(psi @ Ap) / max(float(psi @ Fp), 1e-300)
                res = float(np.linalg.norm(Ap - lam_new * Fp)) / max(float(np.linalg.norm(Fp)), 1e-300)
                phi, lam = psi, lam_new
                if res < 1e-3 * shift + tol:
                    break
            if phi.min() < -1e-6:
                phi = np.abs(phi)
            best = (1.0 / lam, np.maximum(phi, 0.0), res)
            if res < tol * 10:
                break
        if best is None:
            return k0, (np.ones(N) if phi0 is None else phi0)
        return best[0], best[1]


class Shape:
    """Cached flux shape for first-order perturbation reactivity."""

    def __init__(self, k, phi, nuSf, Sa):
        self.k, self.phi = k, phi
        self.nuSf, self.Sa = nuSf.copy(), Sa.copy()
        self.w = phi ** 2
        self.den = float(self.w @ nuSf)
        self.rho0 = 1.0 - 1.0 / k

    def rho(self, nuSf, Sa):
        lam = 1.0 / self.k
        if not self.den > 0:          # no fissile material left in the flux
            return min(self.rho0, -0.5)
        return self.rho0 + float(self.w @ (lam * (nuSf - self.nuSf) - (Sa - self.Sa))) / self.den

    def drho_dSa_uniform(self, mask):
        return -float(self.w[mask].sum()) / self.den if self.den > 0 else 0.0


class PointKinetics:
    """n = P_fission / P_nominal, precursors c_i in the same units (sum lambda_i c_i = delayed source)."""

    def __init__(self, beta, lam, Lambda, source=1e-9):
        self.beta_i, self.lam = np.asarray(beta), np.asarray(lam)
        self.beta = float(self.beta_i.sum())
        self.Lambda = Lambda
        self.source = source
        self.n = 1.0
        self.c = self.beta_i / (self.lam * Lambda)

    def set_steady(self, n):
        self.n = n
        self.c = self.beta_i / (self.lam * self.Lambda) * n

    def step(self, rho, dt):
        """Exact solution over dt with constant rho; returns the time-average of n over the step."""
        m = len(self.lam)
        A = np.zeros((m + 3, m + 3))
        A[0, 0] = (rho - self.beta) / self.Lambda
        A[0, 1:m + 1] = self.lam
        A[1:m + 1, 0] = self.beta_i / self.Lambda
        A[np.arange(1, m + 1), np.arange(1, m + 1)] = -self.lam
        A[0, m + 2] = self.source / self.Lambda       # external source (constant state)
        A[m + 1, 0] = 1.0                              # integral of n
        y = np.concatenate([[self.n], self.c, [0.0, 1.0]])
        y1 = expm(A * dt) @ y
        self.n = max(float(y1[0]), 0.0)
        self.c = np.maximum(y1[1:m + 1], 0.0)
        return max(float(y1[m + 1]) / dt, 0.0)

    def period(self, rho):
        """Stable period from the inhour equation (s); inf for rho = 0, negative if subcritical."""
        return inhour_period(rho, self.beta_i, self.lam, self.Lambda)


def inhour_rho(omega, beta_i, lam, Lambda):
    """Reactivity giving the stable inverse period omega (1/s)."""
    omega = np.asarray(omega, float)
    return omega * Lambda + (beta_i[:, None] * omega.ravel()[None, :] /
                             (omega.ravel()[None, :] + lam[:, None])).sum(0).reshape(omega.shape)


def inhour_period(rho, beta_i, lam, Lambda):
    if abs(rho) < 1e-12:
        return np.inf
    from scipy.optimize import brentq
    f = lambda w: float(inhour_rho(w, beta_i, lam, Lambda)) - rho
    if rho > 0:
        hi = 1.0
        while f(hi) < 0:
            hi *= 10
        return 1.0 / brentq(f, 0.0, hi)
    lo = -lam.min() * (1 - 1e-12)
    return 1.0 / brentq(f, lo, 0.0 - 1e-15)
