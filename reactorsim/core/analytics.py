"""Closed-form and point-model results that complement the lattice simulator.

Everything here is fast (milliseconds) and is used by the GUI pages, the CLI and the tests:

* xenon and samarium after a shutdown or power change (point I/Xe/Pm/Sm chains, exact)
* the restart window: time after shutdown during which xenon worth exceeds the available excess
* inhour equation (stable period vs reactivity) and the prompt-jump approximation
* Nordheim-Fuchs prompt burst with adiabatic Doppler feedback (peak power, energy, width)
* ANS-5.1 decay heat and the Way-Wigner approximation
* boil-off and adiabatic heat-up times after loss of cooling
"""
from __future__ import annotations

import math

import numpy as np

from .neutronics import inhour_period, inhour_rho
from .nuclear_data import (LAMBDA_I, LAMBDA_PM, LAMBDA_XE, decay_heat_fraction, delayed_groups,
                           mix_yields, way_wigner)
from .reactors import REACTORS, ReactorType
from . import materials as mat

H = 3600.0


def core_masses(rt: ReactorType):
    """Fuel and cladding mass (kg) of the core from the rod geometry."""
    L = rt.n_rods * (5.94 if rt.flow_direction == "cross" else rt.height_m)
    fuel, clad = mat.FUELS[rt.fuel], mat.CLADS[rt.clad]
    m_f = fuel["rho"] * rt.smear * math.pi * rt.r_pellet ** 2 * L
    m_c = clad["rho"] * math.pi * (rt.r_clad_out ** 2 - rt.r_clad_in ** 2) * L * (1 + rt.zr_extra)
    return m_f, m_c


# ---------------------------------------------------------------------------------- poisons
def _poison_constants(rt: ReactorType):
    m = mix_yields(rt.fissions)
    return m["gI"], m["gX"], m["gP"], rt.sigma_xe, rt.sigma_sm


def xenon_equilibrium_worth(rt: ReactorType, power: float = 1.0, phi_nom: float | None = None):
    """Equilibrium Xe-135 worth (dk/k, negative) in the point model:
    rho_Xe = -(gI + gX) / nu * (phi/phi_X) / (1 + phi/phi_X),  phi_X = lambda_X / sigma_X."""
    gI, gX, _, sx, _ = _poison_constants(rt)
    phi = (phi_nom or rt.phi_nom) * power
    nu = _nu(rt)
    r = sx * phi / LAMBDA_XE
    return -(gI + gX) / nu * r / (1 + r)


def _nu(rt):
    return mix_yields(rt.fissions)["nu"]


def poison_transient(rt: ReactorType, t, p_before=1.0, p_after=0.0, equilibrium_worth=None):
    """Xe-135 and Sm-149 worth (dk/k) after a step in power at t = 0 from equilibrium.

    Returns dict(t, xenon, samarium, iodine, xe_rel) with worths scaled so the equilibrium
    full-power xenon worth equals ``equilibrium_worth`` (default: the point-model value)."""
    t = np.asarray(t, float)
    gI, gX, gP, sx, ss = _poison_constants(rt)
    phi0, phi1 = rt.phi_nom * p_before, rt.phi_nom * p_after
    F0, F1 = p_before, p_after                       # fission rate in units of the rated rate
    # equilibrium before the step
    I0 = gI * F0 / LAMBDA_I
    X0 = (gI + gX) * F0 / (LAMBDA_XE + sx * phi0)
    P0 = gP * F0 / LAMBDA_PM
    S0 = gP * F0 / (ss * phi0) if phi0 > 0 else 0.0
    a = LAMBDA_XE + sx * phi1
    I_inf = gI * F1 / LAMBDA_I
    I = I_inf + (I0 - I_inf) * np.exp(-LAMBDA_I * t)
    X_inf = (gX * F1 + LAMBDA_I * I_inf) / a
    d = a - LAMBDA_I
    coup = LAMBDA_I * (I0 - I_inf) / d * (np.exp(-LAMBDA_I * t) - np.exp(-a * t)) if abs(d) > 1e-14 \
        else LAMBDA_I * (I0 - I_inf) * t * np.exp(-a * t)
    X = X_inf + (X0 - X_inf) * np.exp(-a * t) + coup
    P_inf = gP * F1 / LAMBDA_PM
    b = ss * phi1
    if b > 0:
        S_inf = LAMBDA_PM * P_inf / b
        db = b - LAMBDA_PM
        S = S_inf + (S0 - S_inf) * np.exp(-b * t) + LAMBDA_PM * (P0 - P_inf) / db * (np.exp(-LAMBDA_PM * t) - np.exp(-b * t))
    else:
        S = S0 + P0 * (1 - np.exp(-LAMBDA_PM * t))
    # worth ~ sigma N relative to the full-power equilibrium value (fuel absorption fixed)
    Xref = (gI + gX) * 1.0 / (LAMBDA_XE + sx * rt.phi_nom)
    w_eq = xenon_equilibrium_worth(rt) if equilibrium_worth is None else equilibrium_worth
    Sref = gP / ss / rt.phi_nom * ss * rt.phi_nom   # = gP (units cancel below)
    nu = _nu(rt)
    xe = w_eq * X / Xref
    sm_eq = -gP / nu                                 # equilibrium Sm worth, flux independent
    sm = sm_eq * (S * ss * rt.phi_nom) / max(Sref, 1e-30)
    return dict(t=t, xenon=xe, samarium=sm, iodine=I / max(I0, 1e-30) if I0 > 0 else I, xe_rel=X / Xref)


def restart_window(rt: ReactorType, excess=None, p_before=1.0, t_max=60 * H, n=4000):
    """Interval after a trip from equilibrium during which the extra xenon worth exceeds the
    excess reactivity available to override it. Returns (t_peak, peak extra worth, t_lock, t_unlock)
    in seconds (t_lock/t_unlock None if the excess always suffices)."""
    excess = rt.excess if excess is None else excess
    t = np.linspace(0, t_max, n)
    r = poison_transient(rt, t, p_before, 0.0)
    extra = r["xenon"] - r["xenon"][0]               # additional negative worth
    i = int(np.argmin(extra))
    lock = np.nonzero(-extra > excess)[0]
    if lock.size == 0:
        return t[i], float(extra[i]), None, None
    return t[i], float(extra[i]), float(t[lock[0]]), float(t[lock[-1]])


# ---------------------------------------------------------------------------------- kinetics
def inhour_curve(rt: ReactorType, rho_dollars=None):
    """Stable period T(rho) for reactivities in dollars (positive and negative)."""
    beta, lam = delayed_groups(rt.fissions, rt.beta_eff)
    if rho_dollars is None:
        rho_dollars = np.concatenate([-np.geomspace(5, 0.01, 60), np.geomspace(0.01, 1.5, 80)])
    T = np.array([inhour_period(r * rt.beta_eff, beta, lam, rt.Lambda) for r in rho_dollars])
    return np.asarray(rho_dollars), T


def prompt_jump(rho_dollars):
    """Prompt-jump ratio n+/n- = beta / (beta - rho) for rho < beta."""
    r = np.asarray(rho_dollars, float)
    return np.where(r < 1, 1.0 / (1.0 - r), np.inf)


def nordheim_fuchs(rt: ReactorType, rho_dollars, P0=1e-6, gamma=None, heat_cap=None):
    """Prompt-supercritical burst with adiabatic feedback rho(t) = rho0 - gamma E(t).

    alpha0 = (rho0 - beta)/Lambda; peak power P_max = alpha0^2 / (2 gamma/Lambda) and burst energy
    E = 2 alpha0 Lambda / gamma; full width at half maximum = 3.52 / alpha0.
    gamma (dk per J) defaults to the Doppler coefficient over the core fuel heat capacity."""
    rho0 = rho_dollars * rt.beta_eff
    alpha0 = (rho0 - rt.beta_eff) / rt.Lambda
    if alpha0 <= 0:
        return None
    if heat_cap is None:
        fuel = mat.FUELS[rt.fuel]
        heat_cap = core_masses(rt)[0] * float(fuel["cp"](np.array(900.0)))
    if gamma is None:
        gamma = abs(rt.alpha_D + rt.alpha_fexp) / heat_cap
    E = 2 * alpha0 * rt.Lambda / gamma
    Pmax = alpha0 ** 2 * rt.Lambda / (2 * gamma)
    fwhm = 3.525 / alpha0
    dT = E / heat_cap
    return dict(alpha0=alpha0, period=1 / alpha0, energy_J=E, peak_W=Pmax, peak_rel=Pmax / (rt.power_MW * 1e6),
                fwhm_s=fwhm, dT_fuel=dT, gamma=gamma)


def nordheim_fuchs_curve(res, n=400):
    """P(t) of the burst: P = P_max sech^2(alpha0 t / 2), centred at t = 0."""
    a = res["alpha0"]
    t = np.linspace(-6 / a, 6 / a, n)
    return t, res["peak_W"] / np.cosh(a * t / 2) ** 2


# ---------------------------------------------------------------------------------- decay heat
def decay_heat_curves(t, t_op_days=(1.0, 30.0, 365.0, 1000.0)):
    """ANS-5.1 decay power fraction after shutdown for several operating histories, plus Way-Wigner."""
    t = np.asarray(t, float)
    out = {f"{d:g} d": decay_heat_fraction(t, d * 86400.0) for d in t_op_days}
    out["Way-Wigner (inf.)"] = way_wigner(t)
    return out


def boil_off(rt: ReactorType, t_shutdown=60.0, water_t=None, t_op_days=None):
    """Times to boil the coolant down to the top and bottom of the fuel with no injection,
    driven by ANS-5.1 decay heat from ``t_shutdown`` after the trip (saturated pool at p_nom)."""
    if rt.coolant != "water":
        return None
    t_op = (t_op_days or rt.t_op_days) * 86400.0
    hfg = float(mat.h_fg(rt.p_nom if rt.boiling_core else 1.0))
    M_top = (rt.M_nom - rt.M_top) * 1e3 if water_t is None else water_t * 1e3
    M_bot = (rt.M_top - rt.M_bot) * 1e3
    t = np.geomspace(max(t_shutdown, 1.0), 30 * 86400, 4000)
    P = decay_heat_fraction(t, t_op) * rt.power_MW * 1e6
    E = np.concatenate([[0.0], np.cumsum(0.5 * (P[1:] + P[:-1]) * np.diff(t))])
    m = E / hfg
    t_top = float(np.interp(M_top, m, t))
    t_bot = float(np.interp(M_top + M_bot, m, t))
    return dict(t_uncover=t_top, t_dry=t_bot, t=t, boiled=m)


def adiabatic_heatup_rate(rt: ReactorType, t_after, T=1000.0):
    """Core-average heat-up rate (K/s) of fuel + cladding with no cooling, t_after s after the trip."""
    fuel = mat.FUELS[rt.fuel]
    clad = mat.CLADS[rt.clad]
    m_f, m_c = core_masses(rt)
    C = m_f * float(fuel["cp"](np.array(T))) + m_c * float(clad["cp"](np.array(T)))
    P = float(decay_heat_fraction(np.array(t_after), rt.t_op_days * 86400.0)) * rt.power_MW * 1e6
    return P / C


def reactor_table():
    """Key design data of the built-in reactor types (for the GUI and the README)."""
    rows = []
    for k, rt in REACTORS.items():
        rows.append(dict(key=k, name=rt.name, power_MW=rt.power_MW, coolant=rt.coolant, moderator=rt.moderator,
                         fuel=rt.fuel, beta_eff=rt.beta_eff, Lambda=rt.Lambda, alpha_D=rt.alpha_D,
                         alpha_mod=rt.alpha_mod, alpha_void=rt.alpha_void, phi=rt.phi_nom,
                         xe_eq=xenon_equilibrium_worth(rt)))
    return rows


__all__ = ["core_masses", "xenon_equilibrium_worth", "poison_transient", "restart_window", "inhour_curve", "prompt_jump",
           "nordheim_fuchs", "nordheim_fuchs_curve", "decay_heat_curves", "boil_off", "adiabatic_heatup_rate",
           "reactor_table", "inhour_rho"]
