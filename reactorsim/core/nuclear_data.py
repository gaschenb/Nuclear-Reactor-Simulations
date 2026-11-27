"""Nuclear data: delayed neutrons, fission-product poisons, decay heat.

Sources
  Delayed neutrons      G.R. Keepin, Physics of Nuclear Kinetics (1965), six-group fits
  Poison chain yields   J.R. Lamarsh & A.J. Baratta, Introduction to Nuclear Engineering, 3rd ed.,
                        table 7.5 (I-135 cumulative, Xe-135 direct, Pm-149 cumulative)
  Half-lives            NNDC/ENSDF: I-135 6.57 h, Xe-135 9.14 h, Pm-149 53.08 h
  Decay heat            ANSI/ANS-5.1-1979, U-235 thermal fission, 23-term exponential fit
                        (also the basis of ANS-5.1-2005), plus U-239 / Np-239 capture terms
"""
from __future__ import annotations

import numpy as np

MEV = 1.602176634e-13            # J
E_FISSION = 200.0 * MEV          # recoverable energy per fission (J)
BARN = 1e-24                     # cm^2
LN2 = np.log(2.0)
HOUR = 3600.0

# ------------------------------------------------------------------ delayed neutrons (Keepin)
DELAYED = {
    "U235": dict(beta=0.0065,
                 a=np.array([0.033, 0.219, 0.196, 0.395, 0.115, 0.042]),
                 lam=np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])),
    "Pu239": dict(beta=0.0021,
                  a=np.array([0.035, 0.298, 0.211, 0.326, 0.086, 0.044]),
                  lam=np.array([0.0128, 0.0301, 0.124, 0.325, 1.12, 2.69])),
    "U238": dict(beta=0.0148,
                 a=np.array([0.013, 0.137, 0.162, 0.388, 0.225, 0.075]),
                 lam=np.array([0.0132, 0.0321, 0.139, 0.358, 1.41, 4.02])),
}


def delayed_groups(fission_fractions: dict, beta_eff: float):
    """Six-group (beta_i, lambda_i) for a fission mix, rescaled to the given beta_eff.
    Group decay constants are averaged weighted by each nuclide's delayed yield."""
    num_b = np.zeros(6)
    num_l = np.zeros(6)
    for nuc, w in fission_fractions.items():
        d = DELAYED[nuc]
        bi = w * d["beta"] * d["a"]
        num_b += bi
        num_l += bi * d["lam"]
    lam = num_l / num_b
    beta = num_b / num_b.sum() * beta_eff
    return beta, lam


# ------------------------------------------------------------------ poisons
LAMBDA_I = LN2 / (6.57 * HOUR)
LAMBDA_XE = LN2 / (9.14 * HOUR)
LAMBDA_PM = LN2 / (53.08 * HOUR)

YIELDS = {   # (gamma_I135, gamma_Xe135 direct, gamma_Pm149)
    "U235": (0.0639, 0.00237, 0.01130),
    "Pu239": (0.0604, 0.01050, 0.01900),
    "U238": (0.0697, 0.00027, 0.01600),
}
NU = {"U235": 2.43, "Pu239": 2.88, "U238": 2.60}

SIGMA_XE_2200 = 2.65e6 * BARN      # Xe-135 absorption at 0.0253 eV
SIGMA_SM_2200 = 4.01e4 * BARN      # Sm-149


def mix_yields(fission_fractions: dict):
    g = np.zeros(3)
    nu = 0.0
    for nuc, w in fission_fractions.items():
        g += w * np.array(YIELDS[nuc])
        nu += w * NU[nuc]
    return dict(gI=g[0], gX=g[1], gP=g[2], nu=nu)


# ------------------------------------------------------------------ decay heat (ANS-5.1-1979)
# f(t) = sum_i alpha_i exp(-lambda_i t)   [MeV / s per fission], t seconds after one fission
ANS_ALPHA = np.array([
    6.5057e-01, 5.1264e-01, 2.4384e-01, 1.3850e-01, 5.5440e-02, 2.2225e-02, 3.3088e-03,
    9.3015e-04, 8.0943e-04, 1.9567e-04, 3.2535e-05, 7.5595e-06, 2.5232e-06, 4.9948e-07,
    1.8531e-07, 2.6608e-08, 2.2398e-09, 8.1641e-12, 8.7797e-11, 2.5131e-14, 3.2176e-16,
    4.5038e-17, 7.4791e-17])
ANS_LAMBDA = np.array([
    2.2138e+01, 5.1587e-01, 1.9594e-01, 1.0314e-01, 3.3656e-02, 1.1681e-02, 3.5870e-03,
    1.3930e-03, 6.2630e-04, 1.8906e-04, 5.4988e-05, 2.0958e-05, 1.0010e-05, 2.5438e-06,
    6.6361e-07, 1.2290e-07, 2.7213e-08, 4.3714e-09, 7.5780e-10, 2.4786e-10, 2.2384e-13,
    2.4600e-14, 1.5699e-14])
E_U239, LAMBDA_U239 = 0.474, 4.91e-4         # MeV per decay, 1/s (23.5 min)
E_NP239, LAMBDA_NP239 = 0.419, 3.41e-6       # MeV per decay, 1/s (2.36 d)
Q_MEV = 200.0


def decay_heat_fraction(t, t_op=np.inf, R=0.6):
    """ANS-5.1-1979 decay power / operating power at time t (s) after shutdown, after
    constant-power operation for t_op seconds. R = U-239 atoms produced per fission."""
    t = np.asarray(t, float)
    lt = ANS_LAMBDA[:, None] * t.ravel()[None, :]
    if np.isinf(t_op):
        F = (ANS_ALPHA / ANS_LAMBDA)[:, None] * np.exp(-lt)
    else:
        F = (ANS_ALPHA / ANS_LAMBDA)[:, None] * (np.exp(-lt) - np.exp(-lt - ANS_LAMBDA[:, None] * t_op))
    fp = F.sum(0)
    tt = t.ravel()
    lu, ln = LAMBDA_U239, LAMBDA_NP239
    if np.isinf(t_op):
        nu_ = R * np.exp(-lu * tt)
        nn = R * (lu / (lu - ln) * (np.exp(-ln * tt) - np.exp(-lu * tt)) + np.exp(-ln * tt))
    else:
        nu_ = R * (1 - np.exp(-lu * t_op)) * np.exp(-lu * tt)
        nn = R * (lu / (lu - ln) * (1 - np.exp(-ln * t_op)) * (np.exp(-ln * tt) - np.exp(-lu * tt))
                  + (1 - np.exp(-ln * t_op)) * np.exp(-ln * tt))
    act = E_U239 * nu_ + E_NP239 * nn
    return ((fp + act) / Q_MEV).reshape(t.shape)


def way_wigner(t, t_op=np.inf):
    """Way-Wigner (1948) approximation P/P0 = 0.0622 [t^-0.2 - (t + t_op)^-0.2]."""
    t = np.maximum(np.asarray(t, float), 1e-3)
    if np.isinf(t_op):
        return 0.0622 * t ** -0.2
    return 0.0622 * (t ** -0.2 - (t + t_op) ** -0.2)
