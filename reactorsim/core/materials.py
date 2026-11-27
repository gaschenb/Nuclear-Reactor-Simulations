"""Material and coolant properties.

Sources
  UO2 heat capacity and conductivity   J.K. Fink, J. Nucl. Mater. 279 (2000) 1-18
  Zircaloy heat capacity               MATPRO (NUREG/CR-6150), linear fit below the alpha-beta transition
  Graphite heat capacity               Butland & Maddison, J. Nucl. Mater. 49 (1973) 45
  Saturated water / steam              IAPWS-IF97 values, tabulated 0.1-20 MPa
  Sodium                               J.K. Fink & L. Leibowitz, ANL/RE-95/2 (1995)
  Water pool-boiling CHF               N. Zuber (1959); nucleate boiling: Jens & Lottes (1951)
"""
from __future__ import annotations

import numpy as np

SIGMA_SB = 5.670374e-8


# ------------------------------------------------------------------ solids
def cp_uo2(T):
    """UO2 specific heat (J/kg K), Fink 2000."""
    T = np.clip(T, 280.0, 3100.0)
    th, C1, C2, C3, Ea = 548.68, 302.27, 8.463e-3, 8.741e7, 18531.7
    e = np.exp(th / T)
    return C1 * th ** 2 * e / (T ** 2 * (e - 1) ** 2) + 2 * C2 * T + C3 * Ea * np.exp(-Ea / T) / T ** 2


def k_uo2(T):
    """UO2 thermal conductivity, 95 % TD (W/m K), Fink 2000."""
    t = np.clip(T, 300.0, 3100.0) / 1000.0
    return 100.0 / (7.5408 + 17.692 * t + 3.6142 * t ** 2) + 6400.0 / t ** 2.5 * np.exp(-16.35 / t)


def cp_zr(T):
    T = np.clip(T, 280.0, 2200.0)
    return 255.7 + 0.1024 * T


def cp_steel(T):
    T = np.clip(T, 280.0, 1700.0)
    return 450.0 + 0.28 * (T - 300.0) * (T < 1100) + 224.0 * (T >= 1100)


def cp_graphite(T):
    """Graphite specific heat (J/kg K), Butland & Maddison 1973."""
    T = np.clip(T, 250.0, 3500.0)
    c = (0.54212 - 2.42667e-6 * T - 90.2725 / T - 43449.3 / T ** 2 + 1.59309e7 / T ** 3
         - 1.43688e9 / T ** 4)
    return 4184.0 * c


def cp_metal_fuel(T):          # U-10Zr
    T = np.clip(T, 280.0, 1800.0)
    return 160.0 + 0.04 * (T - 300.0)


FUELS = {
    #            density  cp-function  conductivity    melting(K)  latent(J/kg)
    "UO2": dict(rho=10400.0, cp=cp_uo2, k=k_uo2, T_melt=3120.0, L=2.77e5),
    "MOX": dict(rho=10700.0, cp=cp_uo2, k=lambda T: 0.92 * k_uo2(T), T_melt=3020.0, L=2.77e5),
    "metal": dict(rho=15800.0, cp=cp_metal_fuel, k=lambda T: 18.0 + 0.012 * (np.clip(T, 300, 1500) - 300),
                  T_melt=1430.0, L=3.8e4),
    "TRISO": dict(rho=1800.0, cp=cp_graphite, k=lambda T: 15.0 + 0 * T, T_melt=3600.0, L=1.0e6),
}
CLADS = {
    "zircaloy": dict(rho=6550.0, cp=cp_zr, T_melt=2125.0, oxidizes=True),
    "zr-nb": dict(rho=6550.0, cp=cp_zr, T_melt=2125.0, oxidizes=True),
    "steel": dict(rho=7900.0, cp=cp_steel, T_melt=1700.0, oxidizes=False),
    "graphite": dict(rho=1750.0, cp=cp_graphite, T_melt=3900.0, oxidizes=False),
}


# ------------------------------------------------------------------ saturated water
#             p(MPa) Tsat(K)  hf(kJ/kg) hfg     rho_l   rho_v   sigma(N/m)
_SAT = np.array([
    [0.1, 372.76, 417.4, 2257.5, 958.4, 0.590, 0.0589],
    [0.2, 393.36, 504.7, 2201.6, 942.9, 1.129, 0.0548],
    [0.5, 424.98, 640.1, 2108.0, 915.3, 2.668, 0.0485],
    [1.0, 453.03, 762.5, 2014.6, 887.0, 5.145, 0.0422],
    [2.0, 485.53, 908.5, 1889.8, 849.8, 10.04, 0.0350],
    [3.0, 506.99, 1008.3, 1794.9, 821.9, 15.00, 0.0300],
    [4.0, 523.51, 1087.4, 1713.4, 798.4, 20.09, 0.0261],
    [5.0, 537.09, 1154.5, 1639.7, 777.4, 25.35, 0.0228],
    [6.0, 548.74, 1213.7, 1570.8, 758.0, 30.82, 0.0199],
    [7.0, 558.98, 1267.4, 1505.1, 739.7, 36.53, 0.0175],
    [8.0, 568.16, 1317.1, 1441.3, 722.2, 42.51, 0.0152],
    [9.0, 576.48, 1363.7, 1379.3, 705.2, 48.79, 0.0132],
    [10.0, 584.15, 1408.0, 1319.0, 688.4, 55.43, 0.0119],
    [12.0, 597.83, 1491.5, 1197.4, 653.7, 68.10, 0.0089],
    [14.0, 609.82, 1571.6, 1066.5, 617.0, 87.10, 0.0062],
    [15.0, 615.31, 1610.5, 1000.0, 603.3, 96.71, 0.0050],
    [16.0, 620.51, 1650.5, 930.6, 586.4, 107.4, 0.0039],
    [18.0, 630.12, 1734.8, 779.0, 548.3, 132.8, 0.0021],
    [20.0, 638.90, 1826.6, 591.9, 490.2, 170.5, 0.00069],
])
_LOGP = np.log(_SAT[:, 0])


def _sat(col, p):
    return np.interp(np.log(np.clip(p, 0.1, 20.0)), _LOGP, _SAT[:, col])


def t_sat(p):
    return _sat(1, p)


def h_fg(p):
    return _sat(3, p) * 1e3


def h_f(p):
    return _sat(2, p) * 1e3


def rho_l(p):
    return _sat(4, p)


def rho_v(p):
    return _sat(5, p)


def surface_tension(p):
    return _sat(6, p)


def p_sat(T):
    """Inverse of t_sat (MPa)."""
    return float(np.exp(np.interp(T, _SAT[:, 1], _LOGP)))


def cp_water(T):
    """Compressed-liquid water specific heat (J/kg K), pressure-insensitive fit up to 620 K."""
    T = np.clip(T, 280.0, 625.0)
    return np.interp(T, [280, 350, 400, 450, 500, 530, 550, 570, 590, 610, 625],
                     [4200, 4190, 4260, 4400, 4660, 4900, 5100, 5450, 6000, 7000, 8500])


def chf_zuber(p):
    """Pool-boiling critical heat flux (W/m^2), Zuber 1959."""
    s, rl, rv, hfg = surface_tension(p), rho_l(p), rho_v(p), h_fg(p)
    return 0.131 * hfg * np.sqrt(rv) * (s * 9.81 * (rl - rv)) ** 0.25


def jens_lottes_dT(q, p):
    """Wall superheat in nucleate boiling (K), Jens-Lottes: 25 (q/1e6)^0.25 exp(-p/6.2)."""
    return 25.0 * (np.maximum(q, 1.0) / 1e6) ** 0.25 * np.exp(-p / 6.2)


COOLANTS = {
    # cp (J/kg K) used for the single-phase march; boiling handled with the tables above
    "water": dict(cp=cp_water, boils=True),
    "d2o": dict(cp=lambda T: 1.03 * cp_water(T), boils=True),
    "helium": dict(cp=lambda T: 5193.0 + 0 * np.asarray(T, float), boils=False),
    "sodium": dict(cp=lambda T: 1630.0 - 0.833 * np.clip(T, 371, 1500) + 4.62e-4 * np.clip(T, 371, 1500) ** 2,
                   boils=True, T_boil_1bar=1156.0, hfg=3.87e6),
}


def sodium_t_sat(p):
    """Sodium saturation temperature (K) from Clausius-Clapeyron around 1156 K at 0.1013 MPa."""
    p = np.maximum(p, 1e-3)
    return 1.0 / (1.0 / 1156.0 - 8.314 / (0.023 * 3.87e6) * np.log(p / 0.1013))


# --- homogeneous-equilibrium vessel state ------------------------------------------------
_PG = np.geomspace(0.1, 20.0, 240)
_VF = 1.0 / rho_l(_PG)
_VG = 1.0 / rho_v(_PG)
_UF = h_f(_PG) - _PG * 1e6 * _VF
_UG = h_f(_PG) + h_fg(_PG) - _PG * 1e6 * _VG


def hem_quality(rho, p):
    """Static quality of a saturated mixture of mean density rho (kg/m3) at p (MPa)."""
    vf, vg = 1.0 / rho_l(p), 1.0 / rho_v(p)
    return float(np.clip((1.0 / max(rho, 1e-6) - vf) / (vg - vf), 0.0, 1.0))


def hem_energy(rho, p):
    """Specific internal energy (J/kg) of a saturated mixture of density rho at p."""
    x = hem_quality(rho, p)
    i = np.interp(np.log(p), np.log(_PG), np.arange(_PG.size))
    uf, ug = np.interp(i, np.arange(_PG.size), _UF), np.interp(i, np.arange(_PG.size), _UG)
    return float(uf + x * (ug - uf))


def hem_pressure(rho, u):
    """Pressure (MPa) of a fixed-volume vessel holding water of mean density rho and internal energy u.

    Inverts u = u_f(p) + x(p, rho) (u_g - u_f) on the saturation line; liquid denser than saturated
    liquid is treated as saturated liquid (pressure set by its temperature)."""
    x = np.clip((1.0 / max(rho, 1e-6) - _VF) / (_VG - _VF), 0.0, 1.0)
    um = _UF + x * (_UG - _UF)
    if u <= um[0]:
        return float(_PG[0])
    if u >= um[-1]:
        return float(_PG[-1])
    k = int(np.searchsorted(um, u))
    k = min(max(k, 1), _PG.size - 1)
    w = (u - um[k - 1]) / max(um[k] - um[k - 1], 1e-9)
    return float(_PG[k - 1] * (_PG[k] / _PG[k - 1]) ** w)
