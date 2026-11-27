"""Coupled simulator: nodal neutronics + point kinetics, poisons, decay heat, fuel/clad/moderator heat
transfer, coolant system, oxidation and the stochastic damage process on the lattice.

    sim = Simulator(SCENARIOS["pwr-sbo"], method="gillespie", seed=1)
    res = sim.run()
    res.series["T_clad_max"], res.snapshots[-1].fields["state"]

Operator splitting per step dt (adaptive):
  events / protection -> coolant state (lagged heat) -> reactivity (perturbation, shape refresh)
  -> point kinetics (exact) -> oxidation -> implicit fuel/clad/moderator heat balance
  -> coolant-system mass and energy -> poisons and decay-heat groups -> jump process.
"""
from __future__ import annotations

import math
import time as _time
from dataclasses import dataclass, replace

import numpy as np

from . import chemistry as chem
from . import damage as dmg
from . import materials as mat
from .depletion import DecayHeat, Poisons
from .neutronics import Diffusion, Lattice, PointKinetics, Shape
from .nuclear_data import E_FISSION, delayed_groups, mix_yields
from .reactors import REACTORS
from .scenarios import SCENARIOS, Scenario

CAL_PER_G = 4186.8          # J/kg per cal/g
TAU_LAT = 20.0          # s, floor of the explicit lateral exchange time of a cell
SNAP_FIELDS = ("power", "T_fuel", "T_clad", "T_mod", "T_cool", "void", "xenon", "state", "ecr",
               "fuel", "release", "wet")


class Cancelled(Exception):
    pass


@dataclass
class Snapshot:
    t: float
    fields: dict


@dataclass
class Result:
    scenario: str
    reactor: str
    nx: int
    nz: int
    t: np.ndarray
    series: dict
    snapshots: list
    events: list
    summary: dict
    fuel_mask: np.ndarray
    runtime: float = 0.0
    cell_times: np.ndarray | None = None      # (5, N): time each cell first reached state s (inf: never)

    def at(self, t):
        """Snapshot nearest to time t."""
        ts = np.array([s.t for s in self.snapshots])
        return self.snapshots[int(np.argmin(np.abs(ts - t)))]


# ------------------------------------------------------------------ fuel enthalpy tables
_T_TAB = np.linspace(280.0, 3200.0, 600)


def _enthalpy_table(cp):
    c = cp(_T_TAB)
    h = np.concatenate([[0.0], np.cumsum(0.5 * (c[1:] + c[:-1]) * np.diff(_T_TAB))])
    return h - np.interp(298.0, _T_TAB, h)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


class Simulator:
    def __init__(self, scenario: Scenario | str, params=None, method="gillespie", seed=0,
                 nx=None, nz=None, overrides=None):
        if isinstance(scenario, str):
            scenario = SCENARIOS[scenario]
        self.sc = scenario
        self.events, self.t_end, self.params = scenario.resolved(params)
        rt = REACTORS[scenario.reactor]
        if scenario.core:
            rt = replace(rt, **{k: scenario.value(v, self.params) for k, v in scenario.core.items()})
        if overrides:
            rt = replace(rt, **overrides)
        self.rt = rt.with_lattice(nx, nz)
        self.method = method
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.log = []
        self._build()
        self._initialize()

    # ================================================================== construction
    def _build(self):
        rt = self.rt
        lat = self.lat = Lattice(rt)
        N, F = lat.N, lat.fuel
        self.N, self.F = N, F
        nF = lat.n_fuel
        self.fuel_props = mat.FUELS[rt.fuel]
        self.clad_props = mat.CLADS[rt.clad]
        self.h_tab = _enthalpy_table(self.fuel_props["cp"])
        # ---------------- neutronics constants
        V_core = rt.core_volume_m3 or (np.pi / 4 * rt.width_m ** 2 *
                                       (5.94 if rt.flow_direction == "cross" else rt.height_m))
        self.V_cell = V_core * 1e6 / nF                     # cm3 represented by one fuel cell
        mix = mix_yields(rt.fissions)
        self.nu = mix["nu"]
        Pd = rt.power_MW * 1e6 / (V_core * 1e6)
        self.Sf0 = np.where(F, Pd / (E_FISSION * rt.phi_nom), 0.0)
        from .nuclear_data import decay_heat_fraction
        self.dec_frac = float(decay_heat_fraction(0.0, rt.t_op_days * 86400.0, rt.R_capture))
        self.P_fis_nom = rt.power_MW * 1e6 * (1 - self.dec_frac)   # fission power at rated thermal power
        xr = np.abs(lat.xc) / (lat.W / 2)
        self.kinf_shape = np.where(F, 1.0 + rt.k_inf_edge_boost * xr ** 2, 0.0)
        self.kinf = rt.k_inf
        self._set_kinf(rt.k_inf)
        self.beta_i, self.lam_i = delayed_groups(rt.fissions, rt.beta_eff)
        self.beta = rt.beta_eff
        self.poisons = Poisons(N, mix["gI"], mix["gX"], mix["gP"], rt.sigma_xe, rt.sigma_sm)
        self.decay = DecayHeat(N, rt.R_capture)
        # ---------------- geometry per fuel cell (thermal, SI)
        L_total = rt.n_rods * (5.94 if rt.flow_direction == "cross" else rt.height_m)
        L = L_total / nF
        rp, ri, ro = rt.r_pellet, rt.r_clad_in, rt.r_clad_out
        self.m_fuel0 = np.where(F, self.fuel_props["rho"] * rt.smear * np.pi * rp ** 2 * L, 0.0)
        m_cl = self.clad_props["rho"] * np.pi * (ro ** 2 - ri ** 2) * L
        self.m_clad = np.where(F, m_cl * (1 + rt.zr_extra), 0.0)
        oxid = self.clad_props["oxidizes"]
        self.m_zr0 = self.m_clad.copy() if oxid else np.zeros(N)
        self.m_zr = self.m_zr0.copy()
        n_struct = N if rt.fuel_pattern else nF
        self.m_struct = (np.full(N, rt.struct_mass_t * 1e3 / n_struct) if rt.fuel_pattern
                         else np.where(F, rt.struct_mass_t * 1e3 / nF, 0.0))
        A_cl = 2 * np.pi * ro * L
        self.A_clad = np.where(F, A_cl, 0.0)
        A_conv = (rt.A_conv_m2 / nF) if rt.A_conv_m2 else A_cl
        self.A_conv = np.where(F, A_conv, 0.1 * A_conv)
        self.A_ox = np.where(F, A_cl * (1 + rt.zr_extra), 0.0) if oxid else np.zeros(N)
        self.w_max = self.clad_props["rho"] * (ro - ri)
        self.L_cell = L
        if rt.G_fc_override:
            self.G_fc_fixed = rt.G_fc_override / nF
        else:
            self.G_fc_fixed = None
        # moderator node
        self.has_mod = rt.moderator is not None
        if self.has_mod:
            self.m_mod0 = np.where(F, rt.mod_mass_t * 1e3 / nF, 0.0)
            self.cp_mod = mat.cp_graphite if rt.moderator == "graphite" else (lambda T: 4200.0 + 0 * T)
        else:
            self.m_mod0 = np.zeros(N)
            self.cp_mod = lambda T: 1000.0 + 0 * T
        self.G_mc = np.where(F, rt.G_mod_cool / nF, 0.0)
        self.G_cm0 = np.where(F, rt.G_clad_mod / nF, 0.0)
        self.A_rad = np.where(F, rt.rad_area_mod / nF, 0.0)
        if rt.moderator == "d2o":
            self.UA_modhx = np.where(F, rt.mod_power_frac * rt.power_MW * 1e6 / (rt.mod_T_nom - 310.0) / nF, 0.0)
        else:
            self.UA_modhx = np.zeros(N)
        # lateral conduction (outer node), boundary
        dxm, dzm, Wym = lat.dx / 100, lat.dz / 100, lat.Wy / 100
        self.Gx0 = dzm * Wym / dxm
        self.Gz0 = dxm * Wym / dzm
        self.UA_b = np.zeros(N)
        edge = (lat.ix == 0) | (lat.ix == lat.nx - 1)
        self.UA_b[edge] = rt.UA_boundary / edge.sum()
        # coolant
        self.n_cols = lat.n_fuel_cols
        self.coolant = mat.COOLANTS[rt.coolant]
        self.water = rt.coolant in ("water", "d2o")
        self.A_flow_col = rt.flow_area_m2 / max(self.n_cols, 1)
        # lower head
        self.LH_decay = DecayHeat(1, rt.R_capture)

    def _set_kinf(self, k):
        self.kinf = k
        self.nuSf0 = self.nu * self.Sf0
        self.Sa0 = np.where(self.F, self.nuSf0 / np.maximum(k * self.kinf_shape, 1e-9), self.rt.refl_Sa)
        Dc = self.rt.M2 * float(self.Sa0[self.F].mean())
        self.Dcoef = np.where(self.F, Dc, self.rt.refl_D)
        self.diff = Diffusion(self.lat, self.Dcoef, self.rt.reflector_savings_cm if not self.rt.fuel_pattern else 5.0)

    # ================================================================== initial state
    def _initialize(self):
        rt, N, F = self.rt, self.N, self.F
        self.t = 0.0
        self.m = F.astype(float)                       # fuel mass relative to a nominal cell
        self.state = np.where(F, 0, dmg.RELOCATED).astype(np.int8)
        self.state[~F] = 0
        self.state_fuel = F.copy()
        self.Tf = np.where(F, rt.T_f_ref, rt.T_in)
        self.Tc = np.full(N, rt.T_in + 20.0)
        self.Tm = np.full(N, rt.mod_T_nom if self.has_mod else rt.T_in)
        self.Tcool = np.full(N, rt.T_in)
        self.Tsink = self.Tcool.copy()
        self.void = np.zeros(N)
        self.wet = np.ones(N)
        self.film = np.zeros(N, bool)
        self.melt = np.zeros(N)
        self.w_ox = np.zeros(N)
        self.absorber_ok = np.ones(N)
        self.inv = np.where(F, 1.0 / max(self.lat.n_fuel, 1), 0.0)   # unreleased volatile FP inventory
        self.inv_gap = F.copy()
        self.Q_cool = np.zeros(N)
        self.Q_wet = np.zeros(N)
        self.qflux = np.zeros(N)
        self.H2 = 0.0
        self.Zr_ox = 0.0
        self.m_mod = self.m_mod0.copy()
        self.M_mod = float(self.m_mod0.sum())
        # coolant system
        self.p = rt.p_nom
        self.M = rt.M_nom * 1e3
        self.M_sec0 = rt.sg_mass_t * 1e3
        self.M_sec = self.M_sec0
        self.saturated = rt.boiling_core
        self.T_in = rt.T_in
        self.ac = True
        self.dc = True
        self.t_dc_lost = None
        self.main_feed = True
        self.afw = True
        self.feed_frac = self.sc.feed0
        self.turbine = 1.0
        self.break_area = 0.0
        self.break_loc = "liquid"
        self.vent_area = 0.0
        self.eccs = {e.name: dict(sys=e, on=e.name not in self.sc.eccs_off, used=0.0) for e in rt.eccs}
        self.extra_inj = []
        self.pump = dict(mode="run", f=self.sc.flow0, f0=self.sc.flow0, t0=0.0, target=self.sc.flow0, tau=1.0)
        self.f = self.sc.flow0
        self.seal_leak = 0.0
        self.mod_cooling = True
        self.rps = self.sc.rps
        self.scram_t = None
        self.controller = self.sc.controller
        self.P_set = self.sc.power0
        self.Sc = 0.0
        self.Sc_min = 0.0
        self.ctrl_limited = False
        self.LH = dict(m=0.0, T=600.0, failed=False, C=0.0)
        self.boron0 = self.boron = rt.boron_ppm
        self.Sigma_B = 0.0
        self.flags = set()
        # rods
        self._init_banks()
        # nominal reference values
        W = rt.flow
        cp = float(self.coolant["cp"](np.array(rt.T_in)))
        self.dT_core_nom = rt.power_MW * 1e6 / (W * cp) if not rt.boiling_core else 0.0
        self.T_avg_nom = rt.T_in + 0.5 * self.dT_core_nom
        if self.water:
            T_ref = self.T_avg_nom if not rt.boiling_core else float(mat.t_sat(rt.p_nom))
            self.V_prim = rt.M_nom * 1e3 / float(mat.rho_l(mat.p_sat(min(T_ref, 630.0))))
        self.U_prim = None
        if rt.has_sg:
            if self.water:
                self.T_sec = float(mat.t_sat(rt.p_sec))
            else:
                self.T_sec = rt.T_in - 40.0
            self.UA_sg = rt.power_MW * 1e6 / max(self.T_avg_nom - self.T_sec, 5.0)
        else:
            self.T_sec, self.UA_sg = 0.0, 0.0
        self.W_steam_nom = rt.power_MW * 1e6 / (float(mat.h_fg(rt.p_nom)) + 4500 * (mat.t_sat(rt.p_nom) - rt.T_feed)) \
            if rt.boiling_core else 0.0
        # ---- calibrate k_inf for the design's excess reactivity at hot full power, equilibrium poisons
        self._calibrate_excess()
        # ---- scenario initial state
        hist = list(self.sc.history)
        P0 = self.sc.power0
        t_op = rt.t_op_days * 86400.0
        if hist:
            self._steady(hist[0][1], equilibrium=True)
            self.decay.set_history(self._fission_rate(hist[0][1]), t_op)
            for dur, Pk in hist:
                self._steady(Pk, equilibrium=False)
                self._evolve_poisons(Pk, dur)
            self._steady(P0, equilibrium=False)
        else:
            self._steady(P0, equilibrium=True)
            self.decay.set_history(self._fission_rate(max(P0, 1e-6)) if P0 > 1e-3 else
                                   self._fission_rate(1.0) * 0.0, t_op)
        if self.sc.eject:
            self._calibrate_eject()
            self._steady(P0, equilibrium=not hist)
        self._calibrate_scram()
        if rt.boron_ppm > 0:
            # soluble boron worth per ppm as a uniform absorber
            self.Sigma_B = rt.boron_worth_pcm * 1e-5 / self.shape.drho_dSa_uniform(self.F)
        self.pk = PointKinetics(self.beta_i, self.lam_i, rt.Lambda, source=1e-10)
        self.pk.set_steady(P0)
        self.n = P0
        self.n_avg = P0
        self.Sc_min = min(0.0, self.Sc)
        self._refresh_shape()
        self._log(0.0, f"initial state: {P0 * 100:.3g} % power, k∞ = {self.kinf:.4f}, "
                       f"control absorber worth {self._rho_ctrl() / self.beta:.2f} $")

    # ------------------------------------------------------------------ rods
    def _cols(self, spec):
        lat = self.lat
        cols = self.rt.fuel_columns()
        if spec == "all":
            m = cols.copy()
        elif spec == "center":
            m = np.zeros(lat.nx, bool)
            m[lat.nx // 2 - (1 if lat.nx % 2 == 0 else 0): lat.nx // 2 + 1] = True
            m &= cols
            if not m.any():
                m = cols.copy()
        elif spec == "left":
            m = cols & (np.arange(lat.nx) < lat.nx // 2)
        else:
            m = np.asarray(spec, bool)
        return m

    def _init_banks(self):
        rt = self.rt
        self.banks = {}
        scram_cols = np.ones(self.lat.nx, bool) if rt.fuel_pattern else self.rt.fuel_columns()
        self.banks["scram"] = dict(cols=scram_cols, depth=0.0, target=0.0, speed=1.0 / rt.scram_time,
                                   frm=rt.rods_from, S=0.0, worth=rt.scram_worth, disp=rt.displacer, t0=None)
        for name, b in self.sc.banks.items():
            w = self.sc.value(b.get("worth", 1.0), self.params)
            d0 = self.sc.value(b.get("depth", 0.0), self.params)
            self.banks[name] = dict(cols=self._cols(b.get("cols", "all")), depth=d0, target=d0, speed=0.0,
                                    frm=b.get("frm", "top"), S=0.0, worth=w, disp=False, t0=None)
        if self.sc.eject:
            w = self.sc.value(self.sc.eject["worth"], self.params)
            self.banks["eject"] = dict(cols=self._cols(self.sc.eject.get("col", "center")), depth=1.0,
                                       target=1.0, speed=0.0, frm="top", S=0.0, worth=w, disp=False, t0=None)

    def _coverage(self, frm, depth):
        lat = self.lat
        z0 = lat.zc - lat.dz / 2
        z1 = lat.zc + lat.dz / 2
        if frm == "top":
            a, b = lat.H * (1 - depth), lat.H
        else:
            a, b = 0.0, lat.H * depth
        return np.clip((np.minimum(z1, b) - np.maximum(z0, a)) / lat.dz, 0.0, 1.0)

    def _water_cov(self, depth):
        """RBMK control channel: fraction of each cell's channel filled by water (vs graphite displacer
        or absorber), absorber tip inserted to depth * H from the top."""
        lat = self.lat
        H = lat.H
        top = lat.H - (lat.zc + lat.dz / 2)         # depth of cell top below core top
        bot = lat.H - (lat.zc - lat.dz / 2)
        D = depth * H
        g = 1.25 / 7.0 * H
        Ld = 4.5 / 7.0 * H

        def ov(a, b):
            return np.clip((np.minimum(bot, b) - np.maximum(top, a)) / lat.dz, 0.0, 1.0)
        return ov(D, D + g) + ov(D + g + Ld, 2 * H)

    def _rod_absorption(self):
        Sa = np.zeros(self.N)
        dk = np.zeros(self.N)
        for b in self.banks.values():
            if b["S"] == 0.0 and not b["disp"]:
                continue
            colmask = b["cols"][self.lat.ix]
            cov = self._coverage(b["frm"], b["depth"]) * colmask
            Sa += b["S"] * cov * self.absorber_ok
            if b["disp"]:
                wc = (self._water_cov(b["depth"]) - self._water_cov(0.0)) * colmask
                dk -= self.rt.displacer_dk * wc          # water in the channel absorbs: graphite in its place adds k
        return Sa, dk

    # ------------------------------------------------------------------ neutronics state
    def _feedback(self):
        rt = self.rt
        Tf = np.maximum(self.Tf, 250.0)
        rD = 2 * rt.alpha_D * np.sqrt(rt.T_f_ref) * (np.sqrt(Tf) - np.sqrt(rt.T_f_ref))
        rE = rt.alpha_fexp * (Tf - rt.T_f_ref)
        node = {"coolant": self.Tcool, "moderator": self.Tm, "clad": self.Tc}[rt.feedback_mod_node]
        if rt.coolant == "water" and rt.feedback_mod_node == "coolant":
            # light-water moderator coefficient grows with temperature (thermal expansion of water):
            # alpha(T) = alpha_ref (T - 300) / (T_ref - 300), integrated
            a = rt.alpha_mod / (2 * max(rt.T_mod_ref - 300.0, 1.0))
            rM = a * ((np.maximum(node, 280.0) - 300.0) ** 2 - (rt.T_mod_ref - 300.0) ** 2)
        else:
            rM = rt.alpha_mod * (node - rt.T_mod_ref)
        rV = rt.alpha_void * (self.void - rt.void_ref)
        if rt.coolant == "water" and rt.moderator is None:
            # light-water lattice: beyond ~85 % void the moderation collapses (no criticality when dry)
            rV = rV - 0.6 * np.clip((self.void - 0.85) / 0.15, 0.0, 1.0)
        if rt.moderator == "d2o":
            # channels left above a boiled-down moderator are unmoderated (natural uranium: k << 1)
            present = np.clip(self.m_mod / np.maximum(self.m_mod0, 1e-12), 0.0, 1.0)
            rV = rV - 0.6 * (1.0 - present)
        # core radial expansion: grid plate follows the inlet, the above-core load pads and the
        # control-rod drivelines follow the outlet temperature
        rR = rt.alpha_radial * (0.3 * (self.T_in - rt.T_in) + 0.7 * (self._T_out() - getattr(self, "T_out_ref", self._T_out())))
        return rD, rE, rM, rV, rR

    def _T_out(self):
        return float(np.mean(self.Tcool.reshape(self.lat.nz, self.lat.nx)[-1]))

    def _xs(self):
        """Current nu Sigma_f and Sigma_a (and the feedback split for diagnostics)."""
        rD, rE, rM, rV, rR = self._feedback()
        Sa_rod, dk_disp = self._rod_absorption()
        fb = rD + rE + rM + rV + rR + dk_disp
        nuSf = self.nuSf0 * self.m * (1.0 + fb)
        if self.rt.debris_k != 1.0:
            # molten pools and relocated debris have lost the optimised fuel-moderator lattice
            nuSf = nuSf * np.where(self.state >= dmg.MOLTEN, self.rt.debris_k, 1.0)
        SX, SS = self.poisons.absorption()
        # fuel absorption follows the fuel; relocated (concentrated, m > 1) debris keeps the cell's k_inf
        fa = np.where(self.m <= 1.0, 0.5 + 0.5 * self.m, self.m)
        Sa = (np.where(self.F, self.Sa0 * fa, self.Sa0) + SX + SS + Sa_rod
              + np.where(self.F, self.Sc + self.Sigma_B * (self.boron - self.boron0), 0.0))
        self._parts = dict(doppler=rD + rE, moderator=rM, void=rV, radial=rR + 0 * rD, displacer=dk_disp,
                           xenon=SX, samarium=SS, rods=Sa_rod)
        return nuSf, Sa

    def _mirror(self, v):
        g = v.reshape(self.lat.nz, self.lat.nx)
        return (0.5 * (g + g[:, ::-1])).ravel()

    def _refresh_shape(self):
        self._Sc_shape = getattr(self, "Sc", 0.0)
        for b in getattr(self, "banks", {}).values():
            b["depth_shape"] = b["depth"]
        nuSf, Sa = self._xs()
        phi0 = self.shape.phi if getattr(self, "shape", None) is not None else None
        k0 = self.shape.k if getattr(self, "shape", None) is not None else 1.0
        k, phi = self.diff.eigen(nuSf, Sa, phi0=phi0, k0=k0)
        if getattr(self, "_symmetric", False):
            phi = self._mirror(phi)
        self.shape = Shape(k, phi, nuSf, Sa)
        Sf = self.Sf0 * self.m
        fr = Sf * phi
        tot = fr.sum()
        self.pshape = fr / tot if tot > 0 else np.zeros(self.N)
        self.phi_per_W = phi / (E_FISSION * tot * self.V_cell) if tot > 0 else 0 * phi
        self.t_shape = self.t
        self.rho_shape = self.shape.rho0
        return k

    def _rho(self):
        nuSf, Sa = self._xs()
        return self.shape.rho(nuSf, Sa)

    def _rho_ctrl(self):
        return -self.Sc * self.shape.drho_dSa_uniform(self.F) * -1.0 if getattr(self, "shape", None) else 0.0

    def _critical(self, iters=6):
        """Uniform absorber (boron / regulating rods) that makes k = 1."""
        for _ in range(iters):
            k = self._refresh_shape()
            rho = 1 - 1 / k
            if abs(rho) < 2e-6:
                break
            d = self.shape.drho_dSa_uniform(self.F)
            self.Sc += -rho / d
        return self.Sc

    def _fission_rate(self, P):
        return P * self.P_fis_nom * self.pshape / E_FISSION

    def _flux(self, P):
        return P * self.P_fis_nom * self.phi_per_W

    # ------------------------------------------------------------------ steady states
    def _steady(self, P, equilibrium=True, iters=40):
        """Fixed point of power shape, temperatures, void and (optionally) equilibrium poisons. The
        fission-source shape is under-relaxed: large, loosely coupled cores with strong feedback
        otherwise flip between axial modes from one iteration to the next."""
        self.n_avg = P
        w = 1.0
        p_old = None
        self._symmetric = True          # the designs are left-right symmetric at steady state
        for it in range(iters):
            self._critical(iters=3)
            if p_old is not None:
                self.pshape = w * self.pshape + (1 - w) * p_old[0]
                self.phi_per_W = w * self.phi_per_W + (1 - w) * p_old[1]
            dp = float(np.max(np.abs(self.pshape - p_old[0]))) / max(float(self.pshape.max()), 1e-30) \
                if p_old is not None else 1.0
            p_old = (self.pshape.copy(), self.phi_per_W.copy())
            if equilibrium:
                phi = self._flux(P)
                Fd = self._fission_rate(P) / self.V_cell
                self.poisons.equilibrium(Fd, phi)
            self._steady_thermal(P)
            for a in (self.Tf, self.Tc, self.Tm, self.Tcool, self.void, self.poisons.X, self.poisons.I,
                      self.poisons.S, self.poisons.P):
                a[:] = self._mirror(a)
            if it > 2 and dp < 2e-4:
                break
            w = 0.4
        self._critical(iters=3)
        self._symmetric = bool(self.rt.fuel_pattern)     # annular core: the two halves are one ring
        self.steady_iters = it + 1

    def _steady_thermal(self, P):
        rt = self.rt
        q = P * self.P_fis_nom * self.pshape
        q_dec = self.decay.power() if self.decay.y.any() else self.dec_frac * P * rt.power_MW * 1e6 * self.pshape
        q = q + q_dec
        qf = q * (1 - rt.mod_power_frac)
        qm = q * rt.mod_power_frac
        if self.has_mod and rt.moderator == "d2o":
            qf = q * (1 - rt.mod_power_frac)
        Qc = float(qf.sum()) + (float(qm.sum()) if (self.has_mod and rt.moderator == "graphite") else 0.0)
        if rt.has_sg:
            cp = float(self.coolant["cp"](np.array(rt.T_in)))
            dT = Qc / (rt.flow * self.sc.flow0 * cp)
            if getattr(self, "_ua_fixed", False):
                self.T_in = self.T_sec + Qc / self.UA_sg - 0.5 * dT
            else:
                self.T_in = rt.T_in
                self.UA_sg = Qc / max(rt.T_in + 0.5 * dT - self.T_sec, 3.0)
                self.T_avg_nom = rt.T_in + 0.5 * dT
        elif rt.boiling_core:
            self.T_in = self._boiling_inlet(P)
        self.Q_cool = qf + (qm if (self.has_mod and rt.moderator == "graphite") else 0.0)
        for _ in range(3):
            self._coolant(dt=None)
            Hc, Ts = self._htc()
            Tc = Ts + qf / np.maximum(Hc, 1e-9)
            Gfc = self._G_fc()
            self.Tc = np.where(self.F, Tc, Ts)
            self.Tf = np.where(self.F, self.Tc + qf / np.maximum(Gfc, 1e-9), self.Tc)
            if self.has_mod and rt.moderator == "graphite":
                self.Tm = np.where(self.F, self.Tcool + qm / np.maximum(self.G_mc, 1e-9), self.Tm)
            self.qflux = qf / np.maximum(self.A_conv, 1e-9)
        self.Q_cool = qf + (qm if (self.has_mod and rt.moderator == "graphite") else 0.0)

    def _evolve_poisons(self, P, dur):
        n = max(int(math.ceil(dur / 1800.0)), 1)
        dt = dur / n
        for _ in range(n):
            self._critical(iters=2)
            phi = self._flux(P)
            R = self._fission_rate(P)
            self.poisons.step(R / self.V_cell, phi, dt)
            self.decay.step(R, dt)
            self._steady_thermal(P)

    def _calibrate_excess(self):
        """Choose k_inf so that, at hot full power with equilibrium Xe/Sm and all control absorbers out,
        rho = excess."""
        rt = self.rt
        k = rt.k_inf
        for it in range(4):
            self._set_kinf(k)
            self.Sc = 0.0
            self._steady(1.0, equilibrium=True, iters=12 if it == 0 else 6)
            self.T_out_ref = self._T_out()
            if it == 0:
                # feedback reference = hot-full-power core averages (power weighted), so the
                # reactivity components read zero at rated conditions
                w = self.pshape
                node = {"coolant": self.Tcool, "moderator": self.Tm, "clad": self.Tc}[rt.feedback_mod_node]
                self.rt = rt = replace(rt, T_f_ref=float(w @ self.Tf), T_mod_ref=float(w @ node),
                                       void_ref=float(w @ self.void))
                if abs(rt.alpha_void) * float(np.ptp(self.void[self.F])) > 0.005:
                    # fuel-loading design (axial enrichment / burnable-absorber zoning) offsets most of
                    # the rated-power void distribution, as in real boiling-channel cores
                    rV = rt.alpha_void * (self.void - rt.void_ref)
                    self.kinf_shape = np.where(self.F, self.kinf_shape / (1 + 0.85 * rV), 0.0)
                    self._set_kinf(k)
                    self._steady(1.0, equilibrium=True, iters=12)
            # rho with control out
            Sc = self.Sc
            d = self.shape.drho_dSa_uniform(self.F)
            rho_out = d * (0.0 - Sc)           # first order
            err = rho_out - rt.excess
            k = k / (1 + err)
            if abs(err) < 2e-4:
                break
        self._set_kinf(k)
        self._ua_fixed = True

    def _bank_worth(self, name, S):
        b = self.banks[name]
        saveS, saved = b["S"], b["depth"]
        b["S"] = S
        b["depth"] = 1.0
        nuSf, Sa = self._xs()
        k_in, _ = self.diff.eigen(nuSf, Sa, phi0=self.shape.phi, k0=self.shape.k)
        b["depth"] = 0.0
        nuSf, Sa = self._xs()
        k_out, _ = self.diff.eigen(nuSf, Sa, phi0=self.shape.phi, k0=self.shape.k)
        b["S"], b["depth"] = saveS, saved
        return (1 - 1 / k_out) - (1 - 1 / k_in)

    def _calibrate_bank(self, name):
        b = self.banks[name]
        target = b["worth"] * self.beta
        S = 1e-3 * float(self.Sa0[self.F].mean()) * 10
        w = self._bank_worth(name, S)
        for _ in range(8):
            if w <= 0:
                S *= 3
                w = self._bank_worth(name, S)
                continue
            S_new = S * target / w if w < 0.8 * target or w > 1.25 * target else S * (1 + (target - w) / w * 1.2)
            S = S_new
            w = self._bank_worth(name, S)
            if abs(w - target) < 0.01 * target:
                break
        b["S"] = S

    def _calibrate_eject(self):
        self._calibrate_bank("eject")

    def _calibrate_scram(self):
        for name in self.banks:
            if name != "eject":
                self._calibrate_bank(name)
        self._critical(iters=3)

    # ================================================================== thermal hydraulics
    def _G_fc(self):
        rt = self.rt
        if self.G_fc_fixed is not None:
            return np.where(self.F, self.G_fc_fixed, 1.0)
        kf = self.fuel_props["k"](np.maximum(self.Tf, 300.0))
        Rp = 1.0 / (8 * np.pi * kf) + 1.0 / (2 * np.pi * rt.r_pellet * rt.h_gap) \
            + (rt.r_clad_out - rt.r_clad_in) / (2 * np.pi * rt.r_clad_in * 15.0)
        G = self.L_cell / Rp
        # gap closes / opens with damage: debris and melt couple fuel and clad tightly
        G = G * np.where(self.state >= dmg.DAMAGED, 3.0, 1.0) * np.maximum(self.m, 1e-3)
        return np.where(self.F, G, 1.0)

    def _flow_fraction(self, dt=None):
        rt = self.rt
        pm = self.pump
        t = self.t
        if pm["mode"] == "run":
            fp = pm["f"]
        elif pm["mode"] == "coast":
            fp = pm["f0"] / (1 + (t - pm["t0"]) / rt.pump_tau)
        elif pm["mode"] == "ramp":
            fp = pm["target"] + (pm["f0"] - pm["target"]) * math.exp(-(t - pm["t0"]) / max(pm["tau"], 1e-3))
        else:
            fp = 0.0
        if self.water:
            M_nc = (rt.M_top + rt.nc_level * (rt.M_nom - rt.M_top)) * 1e3
            cav = float(np.clip((self.M - rt.M_top * 1e3) / max(M_nc - rt.M_top * 1e3, 1.0), 0.0, 1.0))
            nc_ok = self.M >= M_nc
        else:
            cav, nc_ok = 1.0, True
        Q = max(float(self.Q_cool.sum()), 0.0)
        # the natural-circulation head builds up with the loop's heat-up (lag ~10 s)
        Qf = getattr(self, "_Q_nc", Q)
        if dt is not None:
            Qf += (Q - Qf) * (1 - math.exp(-dt / 10.0))
            self._Q_nc = Qf
        f_nc = rt.nat_circ * (max(Qf, 0.0) / (rt.power_MW * 1e6)) ** (1 / 3) if nc_ok else 0.0
        if rt.coolant == "helium":
            f_nc = max(f_nc, 1e-4 * self.p / rt.p_nom)
        if self.water and self.break_area > 0 and self.break_loc == "liquid":
            # a large cold-leg break short-circuits the pumped flow out of the break
            cav *= 1.0 / (1.0 + self.break_area / 0.02)
        self.f_pump = fp * cav
        was = getattr(self, "pool", False)
        thr = 0.03 if was else 0.015          # hysteresis between forced flow and a boiling pool
        self.pool = self.water and (self.M < rt.M_top * 1e3 * (1.02 if was else 1.0)
                                    or (self.f_pump < thr and not nc_ok))
        return max(self.f_pump, f_nc)

    def _boiling_inlet(self, P):
        """Core-inlet temperature of a direct-cycle plant from feedwater/injection mixing."""
        rt = self.rt
        Ts = float(mat.t_sat(self.p))
        W_core = max(rt.flow * max(getattr(self, "f", self.sc.flow0), 0.02), 1.0)
        W_fw = getattr(self, "W_fw", self.W_steam_nom * P * self.feed_frac) if self.main_feed else 0.0
        inj = getattr(self, "W_inj_list", [])
        num = W_fw * (Ts - rt.T_feed) + sum(w * max(Ts - T, 0) for w, T in inj)
        return Ts - min(num / W_core, 80.0)

    def _coolant(self, dt):
        """Coolant temperature, void and wetted fraction per cell from the lagged heat input."""
        rt, lat = self.rt, self.lat
        nx, nz = lat.nx, lat.nz
        self.f = self._flow_fraction(dt) if dt is not None else self.sc.flow0
        if dt is None:
            self.pool = False
        f = self.f
        Ts = float(mat.t_sat(self.p)) if self.water else 0.0
        Q = self.Q_cool.reshape(nz, nx)
        Tcool = np.zeros((nz, nx))
        void = np.zeros((nz, nx))
        wet = np.ones((nz, nx))
        T_dry = np.zeros((nz, nx))
        self.W_steam_col = np.zeros(nx)
        colF = self.rt.fuel_columns()
        if self.water and self.pool:
            # boiling pool with a two-phase mixture level; dry cells above are steam cooled
            rl, rv, hfg = float(mat.rho_l(self.p)), float(mat.rho_v(self.p)), float(mat.h_fg(self.p))
            Mtop, Mbot = rt.M_top * 1e3, rt.M_bot * 1e3
            zc = lat.H / 100 * float(np.clip((self.M - Mbot) / max(Mtop - Mbot, 1.0), 0.0, 1.0))
            W_boil = max(float(self.Q_wet.sum()), 0.0) / hfg
            jg = W_boil / (rv * rt.flow_area_m2)
            Vgj = 1.41 * (float(mat.surface_tension(self.p)) * 9.81 * (rl - rv) / rl ** 2) ** 0.25
            a_pool = float(np.clip(jg / (1.13 * jg + Vgj + 1e-12), 0.0, 0.7))
            zmix = min(zc / (1 - a_pool), lat.H / 100) if zc > 0 else 0.0
            self.level = zmix
            z0 = (lat.zc - lat.dz / 2) / 100
            wet_cell = np.clip((zmix - z0) / (lat.dz / 100), 0.0, 1.0).reshape(nz, nx)
            wet = wet_cell
            Tcool[:] = Ts
            void[:] = wet * a_pool + (1 - wet)
            Qw = self.Q_wet.reshape(nz, nx)
            # steam from the lower plenum (debris and hot structures boiling the remaining water)
            # steam from below the core: water boiled in the lower plenum by debris, hot structures
            # and radiation from the core bottom (the vessel balance's steam generation less the
            # boiling of wetted fuel cells)
            W_below = max(getattr(self, "W_gen_pool", 0.0) - max(float(self.Q_wet.sum()), 0.0) / hfg,
                          max(getattr(self, "Q_LH_water", 0.0), 0.0) / hfg, 0.0)
            Wst = np.full(nx, W_below / nx)
            for iz in range(nz):
                Wst = Wst + np.maximum(Qw[iz], 0.0) / hfg
                T_dry[iz] = Ts
            self.W_steam_col = Wst
            # superheated steam march through dry cells
            Tst = np.full(nx, Ts)
            Tc = self.Tc.reshape(nz, nx)
            Hd = self._h_dry(Wst)[None, :] * self.A_conv.reshape(nz, nx)
            for iz in range(nz):
                Wc = np.maximum(Wst, 1e-4)
                ntu = (1 - wet[iz]) * Hd[iz] / (Wc * 2300.0)
                Tout = Tc[iz] - (Tc[iz] - Tst) * np.exp(-ntu)
                T_dry[iz] = np.where(wet[iz] >= 1, Ts, 0.5 * (Tst + Tout))
                Tst = np.where(wet[iz] >= 1, Ts, Tout)
            self.T_steam_out = Tst
        else:
            self.level = lat.H / 100 if self.water else 0.0
            W_tot = rt.flow * f
            if rt.coolant == "helium":
                W_tot *= self.p / rt.p_nom      # circulators move a volume: mass flow follows the density
            if rt.flow_direction == "cross":
                W = np.full(nx, W_tot / max(lat.n_fuel, 1))
                per_cell = True
            else:
                n_r = nx - self.n_cols
                byp = 0.1 if n_r > 0 else 0.0
                W = np.where(colF, (1 - byp) * W_tot / max(self.n_cols, 1), byp * W_tot / max(n_r, 1))
                per_cell = False
            W = np.maximum(W, 1e-3)
            self.W_cols = W
            if self.water:
                hfg = float(mat.h_fg(self.p))
                rl, rv = float(mat.rho_l(self.p)), float(mat.rho_v(self.p))
                cp_in = float(mat.cp_water(min(self.T_in, Ts)))
                e_in = np.full(nx, cp_in * (self.T_in - Ts))
                e = e_in.copy()
                Tcl = self.Tc.reshape(nz, nx)
                for iz in range(nz):
                    if per_cell:
                        e = e_in.copy()
                    e_out = e + Q[iz] / W
                    # a colder wall can at most cool the water down to the wall temperature
                    e_out = np.maximum(e_out, np.minimum(e, cp_in * (np.minimum(Tcl[iz], Ts) - Ts)))
                    em = 0.5 * (e + e_out)
                    cpl = mat.cp_water(np.minimum(Ts + np.minimum(em, 0) / cp_in, Ts))
                    x = em / hfg
                    Tcool[iz] = np.where(em < 0, Ts + em / cpl, np.where(x < 1, Ts, Ts + (em - hfg) / 2300.0))
                    # superheated steam cannot get hotter than the wall heating it
                    Tcool[iz] = np.minimum(Tcool[iz], np.maximum(Tcl[iz], Ts))
                    e_out = np.minimum(e_out, hfg + 2300.0 * np.maximum(Tcl[iz] - Ts, 0.0))
                    xx = np.clip(x, 1e-9, 1.0)
                    a = xx / (xx + rt.slip * (1 - xx) * rv / rl)
                    void[iz] = np.where(x <= 0, 0.0, a)
                    wet[iz] = np.where(x > 0.9, np.clip((1.0 - x) / 0.1, 0.0, 1.0), 1.0)
                    T_dry[iz] = Tcool[iz]
                    e = e_out
                self.W_steam_col = np.maximum(e, 0) / hfg * W
            elif rt.coolant == "sodium":
                # each cell is a heat exchanger against the clad wall (bounded and stable for any
                # step); heat above saturation goes into vapour quality
                cpn = float(self.coolant["cp"](np.array(rt.T_in)))
                Tsat = float(mat.sodium_t_sat(self.p + 0.05))
                Tc = self.Tc.reshape(nz, nx)
                Hc, _ = self._htc()
                Hc = Hc.reshape(nz, nx)
                T = np.full(nx, self.T_in)
                xq = np.zeros(nx)
                for iz in range(nz):
                    if per_cell:
                        T, xq = np.full(nx, self.T_in), np.zeros(nx)
                    ntu = Hc[iz] / (W * cpn)
                    Tout = Tc[iz] - (Tc[iz] - T) * np.exp(-ntu)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        lm = np.where(ntu > 1e-6, (1 - np.exp(-ntu)) / ntu, 1.0)
                    Tm_ = Tc[iz] - (Tc[iz] - T) * lm
                    # vapour: heat beyond saturation (and condensation when the wall is colder)
                    xq = np.clip(xq + np.maximum(Tout - Tsat, 0) * cpn / 3.87e6
                                 - np.maximum(Tsat - Tc[iz], 0) * cpn / 3.87e6 * (xq > 0), 0.0, 1.0)
                    xx = np.clip(xq, 1e-12, 1.0)
                    a = np.where(xq > 0, xx / (xx + (1 - xx) * 4e-4), 0.0)
                    void[iz] = a
                    Tcool[iz] = np.minimum(Tm_, Tsat)
                    wet[iz] = np.where(a > 0.95, 0.2, 1.0)
                    T_dry[iz] = Tcool[iz]
                    T = np.minimum(Tout, Tsat)
                self.T_out = T
            else:   # helium: exponential approach (bounded for any flow)
                T = np.full(nx, self.T_in)
                Tc = self.Tc.reshape(nz, nx)
                Hc = (self._h_gas() * self.A_conv).reshape(nz, nx)
                for iz in range(nz):
                    ntu = Hc[iz] / (W * 5193.0)
                    Tout = Tc[iz] - (Tc[iz] - T) * np.exp(-ntu)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        lm = np.where(ntu > 1e-6, (1 - np.exp(-ntu)) / ntu, 1.0)
                    Tcool[iz] = Tc[iz] - (Tc[iz] - T) * lm
                    T_dry[iz] = Tcool[iz]
                    T = Tout
                self.T_out = T
        Tcool, void, wet, T_dry = Tcool.ravel(), np.where(self.F, void.ravel(), 0.0), wet.ravel(), T_dry.ravel()
        if dt is not None and (self.water or rt.coolant == "sodium") and getattr(self, "wet", None) is not None:
            # the coolant responds over its transit time; this lag also stops wet/dry chatter
            rho_c = float(mat.rho_l(self.p)) if self.water else 850.0
            m_cool = rho_c * rt.flow_area_m2 * lat.H / 100
            tau = float(np.clip(0.25 * m_cool / max(rt.flow * max(self.f, 1e-3), 1.0), 0.05, 2.0))
            a = 1.0 - math.exp(-dt / tau)
            Tcool = self.Tcool + a * (Tcool - self.Tcool)
            void = self.void + a * (void - self.void)
            wet = self.wet + a * (wet - self.wet)
            T_dry = self.T_dry + a * (T_dry - self.T_dry)
        self.Tcool, self.void, self.wet, self.T_dry = Tcool, void, wet, T_dry

    def _h_dry(self, Wst):
        G = Wst / max(self.A_flow_col, 1e-6)
        return 25.0 + 20.0 * np.maximum(G, 0.0) ** 0.8

    def _h_gas(self):
        rt = self.rt
        f = max(self.f, 1e-5)
        return rt.h_forced * f ** 0.8 * (self.p / rt.p_nom) ** 0.2 + 6.0 * (self.p / rt.p_nom) ** 0.5

    def _htc(self):
        """Linearised clad-to-coolant conductance (W/K) and sink temperature per cell."""
        rt = self.rt
        A = self.A_conv
        Tc = self.Tc
        if rt.coolant == "helium":
            H = self._h_gas() * A
            return H, self.Tcool
        if rt.coolant == "sodium":
            h = rt.h_forced * (0.3 + 0.7 * self.f ** 0.8) * (1 - self.void) + 600.0 * self.void
            return h * A, self.Tcool
        Ts = float(mat.t_sat(self.p))
        q = np.maximum(self.qflux, 1e3)
        f = self.f
        if self.pool:
            h_wet = np.maximum(q / np.maximum(mat.jens_lottes_dT(q, self.p), 0.5), 3000.0)
            T_wet = np.full(self.N, Ts)
        else:
            h_fc = rt.h_forced * max(f, 0.005) ** 0.8 + 800.0
            dTJL = mat.jens_lottes_dT(q, self.p)
            h_nb = q / (dTJL + np.maximum(Ts - self.Tcool, 0.0))
            h_wet = np.where(Tc > Ts, np.maximum(h_fc, h_nb), h_fc)
            T_wet = self.Tcool
        # boiling crisis: critical heat flux, transition and film boiling with hysteresis
        vd = np.clip(self.void, 0, 1)
        if self.pool:
            chf = float(mat.chf_zuber(self.p)) * 0.9 * (1 - vd)
        else:
            fm = min(f, 1.0)
            # high mass flux keeps a liquid film to high quality; at low flow dryout comes early
            chf = float(mat.chf_zuber(self.p)) * (0.5 + 0.7 * min(f, 1.2) ** 0.5) \
                * (1 - 0.7 * vd) ** fm * (1 - vd) ** (2 * (1 - fm))
        chf = np.maximum(chf, 1e3)
        self.dnbr = np.where(self.F & (self.qflux > 1e3), chf / np.maximum(self.qflux, 1.0), np.inf)
        # wall superheat at CHF (Jens-Lottes) and the minimum film-boiling temperature (Groeneveld-Stewart)
        T_chf = Ts + np.clip(mat.jens_lottes_dT(chf, self.p), 5.0, 40.0)
        pm = min(self.p, 9.0)
        T_min = max(557.85 + 44.1 * pm - 3.72 * pm ** 2 + 273.15 - 25.0 * max(self.p - 9.0, 0.0), Ts + 60.0)
        self.T_min_fb = T_min
        enter = (self.qflux > chf) & self.F & (self.wet > 0.5)
        leave = Tc < T_chf
        self.film = (self.film | enter) & ~leave
        rad = 0.6 * mat.SIGMA_SB * (Tc ** 2 + Ts ** 2) * (Tc + Ts)
        tb = np.clip((T_min - Tc) / np.maximum(T_min - T_chf, 1.0), 0.0, 1.0) ** 2
        h_tb = chf * tb / np.maximum(Tc - Ts, 1.0)
        h_wet = np.where(self.film, 250.0 + rad + h_tb, h_wet)
        h_dry = self._h_dry(self.W_steam_col)[self.lat.ix] + 0.1 * rad
        w = self.wet
        Hw, Hd = w * h_wet * A, (1 - w) * h_dry * A
        H = Hw + Hd
        Ts_eff = (Hw * T_wet + Hd * self.T_dry) / np.maximum(H, 1e-12)
        self._Hw, self._Twet = Hw, T_wet
        return H, Ts_eff

    # ================================================================== the time step
    def _apply_event(self, e):
        rt = self.rt
        k, a = e.kind, e.args
        t = self.t
        if k == "scram":
            self._scram("manual scram")
        elif k == "pump_trip":
            self.pump.update(mode="coast", f0=self._flow_fraction_pump(), t0=t)
            self._log(t, "coolant pumps tripped")
        elif k == "pump_restart":
            self.pump.update(mode="ramp", f0=self._flow_fraction_pump(), target=a.get("frac", 1.0), t0=t, tau=8.0)
            self._log(t, "coolant pumps restarted")
        elif k == "flow_ramp":
            self.pump.update(mode="ramp", f0=self._flow_fraction_pump(), target=a["frac"], t0=t, tau=a.get("tau", 10.0))
            self._log(t, f"pump flow to {a['frac'] * 100:.0f} %")
        elif k == "sbo":
            self.ac = False
            self.main_feed = False
            self.mod_cooling = False
            if not rt.afw_turbine:
                self.afw = False
            self.t_dc_lost = t + rt.battery_h * 3600.0
            self.pump.update(mode="coast", f0=self._flow_fraction_pump(), t0=t)
            self.seal_leak = rt.rcp_seal_leak
            if rt.boiling_core:
                self.turbine = 0.0
            self._log(t, "station blackout: AC power lost")
            if self.rps:
                self._scram("loss of AC power")
        elif k == "ac_restore":
            self.ac = True
            self.mod_cooling = True
            self._log(t, "AC power restored")
        elif k == "loss_of_feed":
            self.main_feed = False
            self._log(t, "main feedwater lost")
        elif k == "aux_feed":
            self.afw = bool(a.get("on", True))
            self._log(t, "auxiliary feedwater " + ("available" if self.afw else "unavailable"))
        elif k == "break":
            self.break_area = float(a["area"])
            self.break_loc = a.get("loc", "liquid")
            self._log(t, f"primary break, {self.break_area * 1e4:.0f} cm² ({self.break_loc})")
        elif k == "depressurize":
            self.vent_area = float(a["area"])
            self._log(t, "relief valves opened" if self.vent_area > 0 else "relief path closed")
        elif k == "eccs":
            if a["name"] in self.eccs:
                self.eccs[a["name"]]["on"] = bool(a.get("on", True))
                self._log(t, f"{a['name']} " + ("enabled" if a.get("on", True) else "disabled"))
        elif k == "inject":
            from .reactors import ECCSystem
            name = a.get("name", "injection")
            self.eccs[name] = dict(sys=ECCSystem(name, p_max=a.get("p_max", 20.0), flow=a["flow"],
                                                 volume=a.get("volume", np.inf), T=a.get("T", 300.0),
                                                 level_control=False, needs_ac=False), on=True, used=0.0)
            self._log(t, f"{name} injection available ({a['flow']:.0f} kg/s below {a.get('p_max', 20.0):.1f} MPa)")
        elif k == "turbine_trip":
            self.turbine = float(a.get("bypass", 0.0))
            self._log(t, "turbine trip" + (" (no bypass)" if self.turbine == 0 else ""))
            if self.rps:
                self._scram("turbine stop-valve closure", delay=0.08)
        elif k == "msiv_close":
            self.turbine = 0.0
            self._log(t, "main steam isolation valves closed")
        elif k == "rod_bank":
            b = self.banks[a["name"]]
            b["target"] = float(a["depth"])
            b["speed"] = float(a.get("speed", 0.01))
            self._log(t, f"bank {a['name']} → {b['target'] * 100:.0f} % inserted")
        elif k == "rod_eject":
            b = self.banks["eject"]
            b["target"] = 0.0
            b["speed"] = 1.0 / float(a.get("time", 0.1))
            self._log(t, f"rod ejection ({b['worth']:.2f} $)")
        elif k == "setpoint":
            if "on" in a:
                self.controller = bool(a["on"])
            if "power" in a:
                P = float(a["power"])
                if P <= 0:
                    self._scram("operator shutdown")
                    self.controller = False
                else:
                    if self.scram_t is not None:
                        b = self.banks["scram"]
                        b["target"] = 0.0
                        b["speed"] = 1.0 / 120.0
                        self.scram_t = None
                        self.flags.discard("scram")
                    self.controller = True
                    self.P_set = P
                    self.ctrl_limited = False
                    self._log(t, f"power setpoint {P * 100:.0f} %")
        elif k == "rps":
            self.rps = bool(a.get("on", True))
        elif k == "stop":
            self.t_end = t

    def _flow_fraction_pump(self):
        pm = self.pump
        if pm["mode"] == "run":
            return pm["f"]
        if pm["mode"] == "coast":
            return pm["f0"] / (1 + (self.t - pm["t0"]) / self.rt.pump_tau)
        if pm["mode"] == "ramp":
            return pm["target"] + (pm["f0"] - pm["target"]) * math.exp(-(self.t - pm["t0"]) / max(pm["tau"], 1e-3))
        return 0.0

    def _scram(self, why, delay=None):
        if self.scram_t is not None:
            return
        rt = self.rt
        self.scram_t = self.t + (rt.scram_delay if delay is None else delay)
        b = self.banks["scram"]
        b["t0"] = self.scram_t
        b["target"] = 1.0
        b["speed"] = 1.0 / rt.scram_time
        self.controller = False
        self.flags.add("scram")
        self._log(self.t, f"scram: {why}")

    def _protection(self):
        if not self.rps or self.scram_t is not None:
            return
        rt = self.rt
        if self.n > 1.18:
            self._scram("high neutron flux")
        elif self.sc.power0 > 0.1 and self.f < 0.87 * self.sc.flow0 and self.pump["mode"] != "run":
            self._scram("low coolant flow")
        elif self.water and rt.has_sg and self.p < 0.85 * rt.p_nom:
            self._scram("low primary pressure")
        elif self.water and self.p > 1.07 * rt.p_nom:
            self._scram("high primary pressure")
        elif self.water and self.M < 0.9 * rt.M_nom * 1e3:
            self._scram("low coolant level")
        elif rt.coolant == "helium" and self.p < 0.8 * rt.p_nom:
            self._scram("low helium pressure")

    def _move_rods(self, dt):
        for b in self.banks.values():
            if b["depth"] == b["target"]:
                continue
            if b.get("t0") is not None and self.t < b["t0"]:
                continue
            step = b["speed"] * dt
            if b["target"] > b["depth"]:
                b["depth"] = min(b["target"], b["depth"] + step)
            else:
                b["depth"] = max(b["target"], b["depth"] - step)
            # absorber moves distort the flux: re-solve the shape every few percent of travel
            if abs(b["depth"] - b.get("depth_shape", b["depth"])) > 0.04 or b["depth"] == b["target"]:
                self._shape_stale = True

    def _controller(self, rho, dt):
        if not self.controller:
            return rho
        beta = self.beta
        n = max(self.n, 1e-14)
        # one-group inhour: reach the setpoint over ~max(dt, 40 s) (rho = beta L / (|L| + lambda T))
        if self.P_set > 0:
            L = math.log(self.P_set / n)
            desired = beta * L / (abs(L) + 0.08 * max(dt, 40.0))
        else:
            desired = -beta
        desired = float(np.clip(desired, -0.5 * beta, 0.12 * beta))
        d = self.shape.drho_dSa_uniform(self.F)
        dS = (desired - rho) / d
        rate = 0.05 * beta / abs(d) * dt          # absorber speed limit ~5 cents/s
        dS = float(np.clip(dS, -rate, rate))
        newS = max(self.Sc + dS, self.Sc_min)
        if newS <= self.Sc_min + 1e-15 and desired > rho + d * (newS - self.Sc) + 0.01 * beta:
            if not self.ctrl_limited:
                self.ctrl_limited = True
                self._log(self.t, "control absorbers fully withdrawn: excess reactivity cannot override "
                                  "the poison (xenon)")
        elif self.ctrl_limited and newS > self.Sc_min + 1e-12:
            self.ctrl_limited = False
            self._log(self.t, "criticality regained")
        rho = rho + d * (newS - self.Sc)
        self.Sc = newS
        # large absorber moves make the first-order reactivity estimate stale: re-solve the shape
        if abs(d * (self.Sc - getattr(self, "_Sc_shape", self.Sc))) > 0.03 * beta:
            self._shape_stale = True
        return rho

    def _held_power_ok(self, dt):
        return (self.controller and dt >= 60.0 and self.scram_t is None and self.P_set > 0.2
                and self.n_avg > 0.2 and abs(math.log(max(self.n, 1e-9) / self.P_set)) < 0.03
                and not self.pool and self.break_area == 0 and self.vent_area == 0
                and (self.rt.boiling_core or not getattr(self, "saturated", False))
                and (self.state == dmg.INTACT).all())

    def _held_power_step(self, dt):
        """Step on the xenon time scale with the power held at the setpoint.

        Over minutes the kinetics, fuel and coolant are in equilibrium with the slowly changing poison
        field, so the step is a relaxed static solve: critical absorber position, power shape and
        temperatures, then the analytic I/Xe/Pm/Sm and decay-heat update over dt."""
        P = self.P_set
        Sc_prev = self.Sc
        self._move_rods(dt)
        for _ in range(4):
            old = (self.pshape.copy(), self.phi_per_W.copy())
            self._critical(iters=2)
            self.pshape = 0.4 * self.pshape + 0.6 * old[0]
            self.phi_per_W = 0.4 * self.phi_per_W + 0.6 * old[1]
            self._steady_thermal(P)
        if self.Sc < self.Sc_min:
            self.Sc = Sc_prev
            self._refresh_shape()
            return False
        self._refresh_shape()
        self.rho = self._rho() * 0.0
        self.pk.set_steady(P)
        self.n = self.n_avg = P
        self._heat_flows(*self._htc())
        # the plant is in balance at constant power (inventory, pressure, secondary held)
        self.Q_sink = float(self.Q_cool.sum())
        R = P * self.P_fis_nom * self.pshape / E_FISSION
        self.poisons.step(R / self.V_cell, P * self.P_fis_nom * self.phi_per_W, dt)
        self.decay.step(R, dt)
        self._wet_prev = self.wet.copy()
        self._damage(dt, np.zeros(self.N, bool))
        self.t = self.t + dt
        return True

    def step(self, dt):
        rt, N = self.rt, self.N
        t = self.t
        if self._held_power_ok(dt) and self._held_power_step(dt):
            return
        # ---------------------------------------------------------- coolant state, reactivity
        self._coolant(dt)
        Hc, Tsink = self._htc()
        self._move_rods(dt)
        refresh = (t - self.t_shape > self._shape_interval()) or self._shape_stale
        if refresh:
            self._refresh_shape()
            self._shape_stale = False
        rho = self._rho()
        if not refresh and abs(rho - self.rho_shape) > 3 * self.beta:
            # far from the shape's state the first-order estimate is unreliable
            self._refresh_shape()
            self._shape_stale = False
            rho = self._rho()
        if rho > self.beta:
            # prompt supercritical: never take more than a few prompt periods in one step
            a = (rho - self.beta) / rt.Lambda
            if a * dt > 3.0:
                dt = max(3.0 / a, 1e-8)
        rho = self._controller(rho, dt)
        self.rho = rho
        n_avg = self.pk.step(rho, dt)
        self.n = self.pk.n
        self.n_avg = n_avg
        # ---------------------------------------------------------- powers
        P_fis = n_avg * self.P_fis_nom * self.pshape
        P_dec = self.decay.power()
        q = P_fis + P_dec
        qm = q * rt.mod_power_frac if self.has_mod else 0.0 * q
        qf = q - qm
        # ---------------------------------------------------------- oxidation (steam-limited when dry)
        qox = np.zeros(N)
        if self.clad_props["oxidizes"]:
            Tox = np.maximum(self.Tc, np.where(self.state >= dmg.DAMAGED, self.Tf, 0.0))
            geom = np.select([self.state <= dmg.FAILED, self.state == dmg.DAMAGED, self.state == dmg.MOLTEN],
                             [1.0, 0.6, 0.15], 0.0)
            w_avail = self.m_zr / np.maximum(self.A_ox, 1e-12)
            w_cap = self.w_ox + w_avail
            w_new, dZr = chem.oxidize(self.w_ox, Tox, dt, w_cap, self.A_ox * geom)
            if self.pool:
                dZr = self._steam_limit(dZr, dt)
            dZr = np.minimum(dZr, self.m_zr)
            self.m_zr = self.m_zr - dZr
            self.w_ox = (self.m_zr0 - self.m_zr) / np.maximum(self.A_ox, 1e-12)
            qox = dZr * chem.Q_OX / dt
            self.H2 += float(dZr.sum()) * chem.H2_PER_ZR
            self.Zr_ox += float(dZr.sum())
            self.q_ox_tot = float(qox.sum())
        else:
            self.q_ox_tot = 0.0
        # ---------------------------------------------------------- implicit heat balance
        Tc_old = self.Tc.copy()
        wet_old = getattr(self, "_wet_prev", self.wet)
        self._heat(qf, qm, qox, Hc, Tsink, dt)
        self._heat_flows(Hc, Tsink)
        # ---------------------------------------------------------- coolant system
        self._plant(dt)
        # ---------------------------------------------------------- poisons and decay heat
        R = P_fis / E_FISSION
        phi = n_avg * self.P_fis_nom * self.phi_per_W
        self.poisons.step(R / self.V_cell, phi, dt)
        self.decay.step(R, dt)
        # ---------------------------------------------------------- fission-product release
        hot = self.Tf
        kr = chem.corsor_rate(hot) * (self.state >= dmg.FAILED)
        self.inv = self.inv * np.exp(-kr * dt)
        # ---------------------------------------------------------- jump process
        quench = (self.wet > 0.5) & (wet_old < 0.5) & (Tc_old > 1200.0)
        self._wet_prev = self.wet.copy()
        self._damage(dt, quench)
        self._lower_head(dt)
        self.t = t + dt

    def _heat_flows(self, Hc, Tsink):
        F = self.F
        self.qflux = np.where(F, Hc * np.maximum(self.Tc - Tsink, 0) / np.maximum(self.A_conv, 1e-9), 0.0)
        self.Q_cool = Hc * (self.Tc - Tsink) + (self.G_mc * (self.Tm - self.Tcool) if self.has_mod else 0.0)
        Hw = self._Hw if self.water else Hc
        self.Q_wet = Hw * (self.Tc - getattr(self, "_Twet", self.Tcool)) if self.water else self.Q_cool

    def _shape_interval(self):
        if self.scram_t is not None and self.t < self.scram_t + self.rt.scram_time + 5:
            return 0.02
        return 2.0 if self.n > 0.05 else 60.0

    def _steam_limit(self, dZr, dt):
        """Zr consumption in uncovered cells limited by the steam rising through the column."""
        nx, nz = self.lat.nx, self.lat.nz
        d = dZr.reshape(nz, nx).copy()
        wet = self.wet.reshape(nz, nx)
        avail = self.W_steam_col * dt / chem.STEAM_PER_ZR + 0.0     # kg Zr that the steam can oxidise
        for iz in range(nz):
            dry = wet[iz] < 0.99
            use = np.where(dry, np.minimum(d[iz], avail), d[iz])
            avail = np.where(dry, np.maximum(avail - use, 0.0), avail)
            d[iz] = use
        return d.ravel()

    def _heat(self, qf, qm, qox, Hc, Tsink, dt):
        rt, N, F = self.rt, self.N, self.F
        cpf = self.fuel_props["cp"](self.Tf)
        Cf = np.maximum(self.m_fuel0 * self.m * cpf, 1.0)
        Cc = np.maximum(self.m_clad_th() * self.clad_props["cp"](self.Tc) + self.m_struct * rt.struct_cp, 50.0)
        Gfc = self._G_fc()
        if self.has_mod:
            Cm = np.maximum(self.m_mod * self.cp_mod(self.Tm), 1.0)
            Gcm = self.G_cm0 + self.A_rad * 0.3 * mat.SIGMA_SB * (self.Tc ** 2 + self.Tm ** 2) * (self.Tc + self.Tm)
            if rt.moderator == "d2o":
                present = np.clip(self.m_mod / np.maximum(self.m_mod0, 1e-12), 0.0, 1.0)
                Gcm = Gcm * present
                # gamma heating of a boiled-off moderator region goes into the calandria tubes
                qox = qox + qm * (1 - present)
                qm = qm * present
            Hm = self.G_mc.copy()
            Q_hx = self.UA_modhx * np.maximum(self.Tm - 310.0, 0.0) * (1.0 if self.mod_cooling else 0.0)
        else:
            Cm = np.ones(N)
            Gcm = np.zeros(N)
            Hm = np.zeros(N)
            Q_hx = np.zeros(N)
        # explicit lateral conduction on the outer node
        outer_is_mod = self.has_mod and rt.moderator == "graphite"
        To = self.Tm if outer_is_mod else self.Tc
        Q_lat = self._lateral(To)
        Ub = self.UA_b
        # 3x3 per cell
        A = np.zeros((N, 3, 3))
        b = np.zeros((N, 3))
        A[:, 0, 0] = Cf / dt + Gfc
        A[:, 0, 1] = -Gfc
        b[:, 0] = Cf / dt * self.Tf + qf
        A[:, 1, 0] = -Gfc
        A[:, 1, 1] = Cc / dt + Gfc + Hc + Gcm + (0 if outer_is_mod else Ub)
        A[:, 1, 2] = -Gcm
        b[:, 1] = Cc / dt * self.Tc + qox + Hc * Tsink + (0 if outer_is_mod else Q_lat + Ub * rt.T_boundary)
        A[:, 2, 1] = -Gcm
        A[:, 2, 2] = Cm / dt + Gcm + Hm + (Ub if outer_is_mod else 0)
        b[:, 2] = Cm / dt * self.Tm + qm + Hm * self.Tcool - Q_hx + (Q_lat + Ub * rt.T_boundary if outer_is_mod else 0)
        T = np.linalg.solve(A, b[..., None])[..., 0]
        Tf, Tc, Tm = T[:, 0], T[:, 1], T[:, 2]
        # fuel melting with latent heat
        Tm_f = self.fuel_props["T_melt"]
        over = (Tf > Tm_f) & F
        if over.any():
            E = Cf[over] * (Tf[over] - Tm_f)
            Lm = self.fuel_props["L"] * self.m_fuel0[over] * self.m[over]
            dmelt = E / np.maximum(Lm, 1e-9)
            room = 1.0 - self.melt[over]
            use = np.minimum(dmelt, room)
            self.melt[over] += use
            Tf[over] = Tm_f + (dmelt - use) * Lm / np.maximum(Cf[over], 1e-9)
        # liquid moderator saturates and boils off from the top
        if self.has_mod and rt.moderator == "d2o":
            Tsat_m = 383.0
            hot = (Tm > Tsat_m) & (self.m_mod > 0)
            if hot.any():
                E = Cm[hot] * (Tm[hot] - Tsat_m)
                boil = float(E.sum()) / 2.26e6
                Tm[hot] = Tsat_m
                self.M_mod = max(self.M_mod - boil, 0.0)
                frac = self.M_mod / max(self.m_mod0.sum(), 1.0)
                zl = frac * self.lat.H
                cov = np.clip((zl - (self.lat.zc - self.lat.dz / 2)) / self.lat.dz, 0.0, 1.0)
                self.m_mod = self.m_mod0 * cov
            # emptied regions hold only vapour at saturation
            Tm = np.where(self.m_mod > 1e-6 * np.maximum(self.m_mod0, 1.0), Tm, Tsat_m)
        # vaporization limits of the lumped nodes (energy above them leaves with the vapour / debris)
        Tc = np.minimum(Tc, 3200.0)
        # fuel at its boiling point: the vapour pressure expels fuel from the cell (core disassembly),
        # the shutdown mechanism of a superprompt-critical excursion
        boil = F & (Tf >= 3850.0) & (self.m > 0) & (self.n > 0.5)
        if boil.any() and dt is not None:
            lost = self.m[boil] * -math.expm1(-dt * 20.0)
            self.m[boil] -= lost
            self.dispersed = getattr(self, "dispersed", 0.0) + float(lost.sum())
            self._shape_stale = True
            if "disassembly" not in self.flags:
                self.flags.add("disassembly")
                self._log(self.t, "fuel boiling: fuel vapour expels fuel from the core (disassembly)")
        self.Tf = np.minimum(np.where(F, Tf, Tc), 3900.0)
        self.Tc, self.Tm = Tc, Tm

    def m_clad_th(self):
        return self.m_clad

    def _lateral(self, T):
        lat = self.lat
        nx, nz = lat.nx, lat.nz
        k = self.rt.k_lateral + (4 * mat.SIGMA_SB * T ** 3 * self.rt.pitch if self.rt.lateral_radiation else 0.0)
        k = np.where(self.state == dmg.RELOCATED, 0.0, k) if not self.has_mod else k
        if self.rt.fuel_pattern == "" and not self.has_mod:
            k = np.where(self.F, k, 0.0)
        K = k.reshape(nz, nx)
        Tg = T.reshape(nz, nx)
        Q = np.zeros((nz, nx))
        kx = 2 * K[:, 1:] * K[:, :-1] / np.maximum(K[:, 1:] + K[:, :-1], 1e-12) * self.Gx0
        kz = 2 * K[1:, :] * K[:-1, :] / np.maximum(K[1:, :] + K[:-1, :], 1e-12) * self.Gz0
        # explicit exchange: cells with little heat capacity left (debris, emptied channels) would
        # force tiny steps; their exchange time is floored at TAU_LAT (they still equilibrate)
        G = np.zeros((nz, nx))
        G[:, :-1] += kx; G[:, 1:] += kx; G[:-1, :] += kz; G[1:, :] += kz
        C = self._C_outer().reshape(nz, nx)
        sc = np.clip(C / np.maximum(TAU_LAT * G, 1e-12), 0.0, 1.0)
        kx = kx * np.minimum(sc[:, 1:], sc[:, :-1])
        kz = kz * np.minimum(sc[1:, :], sc[:-1, :])
        fx = kx * (Tg[:, 1:] - Tg[:, :-1])
        Q[:, :-1] += fx
        Q[:, 1:] -= fx
        fz = kz * (Tg[1:, :] - Tg[:-1, :])
        Q[:-1, :] += fz
        Q[1:, :] -= fz
        self._G_lat_sum = np.zeros((nz, nx))
        self._G_lat_sum[:, :-1] += kx
        self._G_lat_sum[:, 1:] += kx
        self._G_lat_sum[:-1, :] += kz
        self._G_lat_sum[1:, :] += kz
        return Q.ravel()

    # ------------------------------------------------------------------ plant (coolant system)
    def _plant(self, dt):
        rt = self.rt
        t = self.t
        if self.t_dc_lost is not None and t >= self.t_dc_lost and self.dc:
            self.dc = False
            self._log(t, "batteries depleted: DC power lost")
            if rt.afw_turbine:
                self.afw = False
        Q = float(self.Q_cool.sum()) + getattr(self, "Q_LH_water", 0.0)
        if self.water and self.pool:
            # boiling pool: wetted cells boil water; heat taken up by steam in dry cells leaves as
            # superheat, except a part radiated / conducted down to the pool by the hot core bottom
            Qw = max(float(self.Q_wet.sum()), 0.0)
            Q = Qw + 0.15 * max(float(self.Q_cool.sum()) - Qw, 0.0) + getattr(self, "Q_LH_water", 0.0)
        p = self.p
        if not self.water:
            self._plant_single_phase(Q, dt)
            return
        Ts = float(mat.t_sat(p))
        hfg = float(mat.h_fg(p))
        cpl = float(mat.cp_water(min(self.T_in, Ts)))
        C = self.M * cpl + rt.C_struct_MJK * 1e6
        # injection
        W_inj, Q_inj_sub, inj_list = 0.0, 0.0, []
        for name, e in self.eccs.items():
            s = e["sys"]
            ok = e["on"] and (self.ac or not s.needs_ac) and (self.dc or not s.needs_dc)
            ok = ok and p < s.p_max and p >= s.p_min and e["used"] < s.volume * 1e3
            if not ok:
                if e.get("active"):
                    e["active"] = False
                continue
            if not e.get("armed"):
                # safety-injection signal: low level, low pressure or a detected break
                if self.M < 0.92 * rt.M_nom * 1e3 or p < 0.8 * rt.p_nom or self.break_area > 0 or not s.level_control:
                    e["armed"] = True
                else:
                    continue
            if s.level_control:
                w = s.flow * float(np.clip((rt.M_nom * 1e3 - self.M) / (0.03 * rt.M_nom * 1e3), 0.0, 1.0))
            else:
                w = s.flow * float(np.clip((s.p_max - p) / (0.5 * s.p_max), 0.05, 1.0)) ** 0.5
            w = min(w, (s.volume * 1e3 - e["used"]) / dt)
            if w > 0 and not e.get("active"):
                e["active"] = True
                if not e.get("logged"):
                    e["logged"] = True
                    self._log(t, f"{name} injecting")
            e["used"] += w * dt
            if self.break_area > 0.05 and self.break_loc == "liquid" and rt.has_sg:
                # ECC bypass: during blowdown the upflow in the downcomer sweeps the injected water
                # straight out of a large cold-leg break (end of bypass near ~0.5 MPa)
                w *= float(np.clip((1.2 - p) / 0.7, 0.0, 1.0))
            W_inj += w
            Q_inj_sub += w * cpl * max(min(self.T_in, Ts) - s.T, 0.0)
            inj_list.append((w, s.T))
        self.W_inj_list = inj_list
        self.W_inj = W_inj
        if rt.boron_ppm > 0 and W_inj > 0:
            self.boron += W_inj * dt * (rt.eccs_boron_ppm - self.boron) / max(self.M + W_inj * dt, 1.0)
        # secondary side / turbine
        Q_sink = 0.0
        W_feed = 0.0
        W_turb = 0.0
        T_avg = self.T_in
        if rt.has_sg:
            g = 0.0 if self.pool else min(self.f, 1.0) ** 0.6
            a = float(np.clip(self.M_sec / (0.2 * self.M_sec0), 0.0, 1.0))
            T_avg = self.T_in + 0.5 * Q / max(rt.flow * max(self.f, 0.01) * cpl, 1.0)
            T_avg = min(T_avg, Ts) if self.saturated else T_avg
            Q_sink = self.UA_sg * g * a * (T_avg - self.T_sec)
            h_sec = float(mat.h_fg(rt.p_sec))
            if self.main_feed:
                W_sf = max(Q_sink, 0.0) / h_sec
            elif self.afw and (self.ac or rt.afw_turbine) and (self.dc or not rt.afw_turbine):
                W_sf = rt.afw_flow * float(np.clip((self.M_sec0 - self.M_sec) / (0.1 * self.M_sec0), 0.0, 1.0))
            else:
                W_sf = 0.0
            self.M_sec = float(np.clip(self.M_sec + (W_sf - max(Q_sink, 0.0) / h_sec) * dt, 0.0, 1.5 * self.M_sec0))
            if self.M_sec <= 0.01 * self.M_sec0 and "sg_dry" not in self.flags:
                self.flags.add("sg_dry")
                self._log(t, "steam generators dry")
        elif rt.boiling_core:
            W_turb_cap = 1.1 * self.W_steam_nom * self.turbine
            W_turb = W_turb_cap   # limited below to steam generated
        Q_aux = rt.UA_aux * max(Ts - rt.T_aux, 0.0) if rt.UA_aux else 0.0
        # breaks
        W_liq = 0.0
        A_steam = self.vent_area
        if self.break_area > 0:
            if self.break_loc == "liquid" and self.M > rt.M_top * 1e3:
                rl = float(mat.rho_l(p))
                G = min(0.61 * math.sqrt(2 * rl * max(p - rt.p_back, 0.0) * 1e6), 3.0e4 * math.sqrt(p / 15.5))
                W_liq = G * self.break_area
            else:
                A_steam += self.break_area
        W_seal = self.seal_leak if self.M > rt.M_top * 1e3 else 0.0
        if self.break_area > 0 and (W_liq > 0 or A_steam > 0 or self.U_prim is not None):
            self.saturated = True
            self._plant_blowdown(dt, Q, Q_sink, Q_aux, inj_list, W_inj, W_seal, A_steam, T_avg)
            return
        # energy balance
        if not self.saturated:
            # implicit in the steam-generator coupling (its time constant is a few seconds)
            K = Q_sink / max(T_avg - self.T_sec, 1e-6) if rt.has_sg and abs(T_avg - self.T_sec) > 1e-6 else 0.0
            delta = T_avg - self.T_in
            T_new = (C / dt * self.T_in + Q - Q_aux - Q_inj_sub - K * (delta - self.T_sec)) / (C / dt + K)
            Q_sink = K * (T_new + delta - self.T_sec)
            self.T_in = T_new
            T_out = self.T_in + Q / max(rt.flow * max(self.f, 0.01) * cpl, 1.0)
            dT_avg = (self.T_in + 0.5 * (T_out - self.T_in)) - self.T_avg_nom
            T_hot = float(np.max(self.Tcool[self.F]))
            # pressurizer: heaters (AC) make up the shrink on cooldown; insurge compresses the steam space
            dT_p = max(dT_avg, 0.0) if self.ac else dT_avg
            p_tgt = float(np.clip(rt.p_nom + 0.09 * dT_p, float(mat.p_sat(min(T_hot, 640.0))) + 0.05, rt.p_relief))
            W_vent = 0.0
            if W_liq > 0 or A_steam > 0:
                p_tgt = float(mat.p_sat(min(T_out, 640.0)))
            if self.T_in >= float(mat.t_sat(self.p)) - 0.5 or T_out >= float(mat.t_sat(self.p)) + 8.0:
                self.saturated = True
                self._log(t, f"primary coolant saturated at {self.p:.1f} MPa")
            tau = 15.0 if (W_liq == 0 and A_steam == 0) else max(0.02 / max(self.break_area + A_steam, 1e-6), 0.5)
            self.W_vent = 0.0
        elif self.break_area > 0:
            self._plant_blowdown(dt, Q, Q_sink, Q_aux, inj_list, W_inj, W_seal, A_steam, T_avg)
            return
        else:
            Q_fw = 0.0
            if rt.boiling_core:
                W_feed = getattr(self, "W_fw", 0.0) if self.main_feed else 0.0
                Q_fw = W_feed * cpl * (Ts - rt.T_feed)
            Q_net = Q - Q_sink - Q_aux - Q_inj_sub - Q_fw
            W_gen = max(Q_net, 0.0) / hfg
            self.W_gen_pool = W_gen
            if rt.boiling_core:
                W_turb = min(W_turb, W_gen)
                W_gen -= W_turb
            # where does the remaining steam go
            if A_steam > 0:
                p_eq = W_gen / (1.456e-3 * A_steam) / 1e6
                p_tgt = float(np.clip(p_eq, rt.p_back, rt.p_relief))
                W_cap = 1.456e-3 * p * 1e6 * A_steam
                W_vent = min(W_gen, W_cap) if p <= p_tgt + 0.05 else W_cap
            elif W_liq > 0:
                p_tgt = rt.p_back + (float(mat.p_sat(min(Ts, 640))) - rt.p_back) * 0.3
                W_vent = W_gen
            else:
                if Q_net > 0:
                    p_tgt = rt.p_relief if (rt.boiling_core and self.turbine * 1.1 * self.W_steam_nom < W_gen + W_turb) \
                        or not rt.boiling_core else rt.p_nom
                    if rt.boiling_core and self.turbine > 0.9:
                        p_tgt = rt.p_nom
                    W_vent = W_gen if p >= rt.p_relief - 0.05 else 0.0
                else:
                    p_tgt = max(rt.p_back, p - 0.5) if not (rt.boiling_core and self.turbine > 0.9) else rt.p_nom
                    W_vent = 0.0
                if rt.boiling_core and self.turbine > 0.9:
                    W_vent = 0.0
            tau = 20.0 if A_steam == 0 and W_liq == 0 else max(0.03 / max(self.break_area + A_steam, 1e-6), 1.0)
            if self.T_in < Ts - 15.0:
                pass
            if rt.boiling_core:
                # feedwater follows the steam flow through the level controller (~10 s); the
                # downcomer mixes it into the core inlet with a transport lag (~4 s)
                W_target = max(W_turb + (rt.M_nom * 1e3 - self.M) / 60.0, 0.0) * self.feed_frac \
                    if self.main_feed else 0.0
                self.W_fw = getattr(self, "W_fw", W_target)
                self.W_fw += (W_target - self.W_fw) * (1 - math.exp(-dt / 10.0))
                T_tgt = self._boiling_inlet(max(self.n_avg, 0.0))
                self.T_in += (T_tgt - self.T_in) * (1 - math.exp(-dt / 4.0))
            else:
                self.T_in = max(Ts - (Q_inj_sub / max(rt.flow * max(self.f, 0.02) * cpl, 1.0)), 280.0)
            # flashing when pressure falls
            self.W_vent = W_vent
        W_mass = W_inj + W_feed - W_liq - W_seal
        if self.saturated:
            W_mass -= self.W_vent
            if rt.boiling_core:
                W_mass -= W_turb
        p_new = self.p + (p_tgt - self.p) * (1 - math.exp(-dt / tau))
        if self.saturated:
            # flashing: a pressure drop boils some of the inventory
            dTs = float(mat.t_sat(p_new)) - Ts
            if dTs < 0:
                W_mass -= C * (-dTs) / hfg / dt * 0.5
        self.p = float(np.clip(p_new, rt.p_back, 21.0))
        self.M = max(self.M + W_mass * dt, 0.0)
        self.W_break = W_liq + (self.W_vent if self.saturated and (A_steam > 0) else 0.0)
        self.Q_sink = Q_sink
        self._level_flags()

    def _plant_blowdown(self, dt, Q, Q_sink, Q_aux, inj_list, W_inj, W_seal, A_steam, T_avg):
        """Saturated primary with an open break: fixed-volume homogeneous-equilibrium vessel.

        State (M, U); the pressure follows from rho = M/V and u = U/M on the saturation line, so
        flashing, condensation by cold injection and the end of blowdown come out of the balance."""
        rt = self.rt
        V = self.V_prim
        p = self.p
        if self.U_prim is None:
            # internal energy of the (possibly subcooled) inventory at its mean temperature
            T_m = float(np.mean(self.Tcool[self.F])) if self.F.any() else self.T_in
            T_m = min(0.5 * (self.T_in + T_m) + 0.5 * (T_m - self.T_in), float(mat.t_sat(p)))
            p_m = float(mat.p_sat(T_m))
            self.U_prim = self.M * mat.hem_energy(max(self.M / V, float(mat.rho_l(p_m))), p_m)
            p = self.p = min(p, max(p_m, rt.p_back))
        rho = self.M / V
        x = mat.hem_quality(rho, p)
        hf, hfg = float(mat.h_f(p)), float(mat.h_fg(p))
        Ts = float(mat.t_sat(p))
        dp = max(p - rt.p_back, 0.0) * 1e6
        # break discharge: liquid-side breaks see the mixture below the level, steam-side breaks steam
        W_liq, W_st = 0.0, 0.0
        if self.break_loc == "liquid" and self.M > rt.M_top * 1e3:
            x_b = min(x, 0.6)
            rho_b = 1.0 / (x_b / float(mat.rho_v(p)) + (1 - x_b) / float(mat.rho_l(p)))
            G = min(0.61 * math.sqrt(2 * rho_b * dp), 3.0e4 * math.sqrt(p / 15.5) * (1 - x_b) + 1.456e-3 * p * 1e6 * x_b)
            W_liq = G * self.break_area
            h_b = hf + x_b * hfg
        else:
            A_steam = A_steam + 0.0
            h_b = hf
        if A_steam > 0:
            W_st = 1.456e-3 * p * 1e6 * A_steam * (1.0 if p > rt.p_back + 0.01 else 0.0)
        h_g = hf + hfg
        W_turb = 0.0
        W_feed = 0.0
        if rt.boiling_core:
            # turbine control valves close as the pressure regulator loses pressure
            W_turb = 1.1 * self.W_steam_nom * self.turbine * float(np.clip((p - 0.7 * rt.p_nom) / (0.2 * rt.p_nom), 0, 1))
            W_feed = getattr(self, "W_fw", 0.0) if self.main_feed else 0.0
        h_inj = sum(w * 4186.0 * (T - 273.15) for w, T in inj_list)
        h_fw = W_feed * 4186.0 * (rt.T_feed - 273.15)
        dU = Q - Q_sink - Q_aux + h_inj + h_fw - W_liq * h_b - (W_st + W_turb) * h_g - W_seal * hf
        dM = W_inj + W_feed - W_liq - W_st - W_turb - W_seal
        # keep the step from draining more than the inventory
        if self.M + dM * dt < 0:
            s_ = self.M / max(-dM * dt, 1e-9)
            dM *= s_
            dU *= s_
        self.M = max(self.M + dM * dt, 0.0)
        self.U_prim = self.U_prim + dU * dt
        if self.M > 1.0:
            p_new = mat.hem_pressure(self.M / V, self.U_prim / self.M)
        else:
            p_new = rt.p_back
        self.p = float(np.clip(p_new, rt.p_back, 21.0))
        if self.p <= rt.p_back + 1e-9 and self.M > 1.0:
            # containment back-pressure holds: the excess energy leaves as steam
            self.U_prim = min(self.U_prim, self.M * mat.hem_energy(self.M / V, rt.p_back))
        Ts = float(mat.t_sat(self.p))
        cpl = float(mat.cp_water(Ts))
        Q_inj_sub = sum(w * cpl * max(Ts - T, 0.0) for w, T in inj_list)
        self.T_in = max(Ts - Q_inj_sub / max(rt.flow * max(self.f, 0.02) * cpl, 1.0), 280.0)
        self.W_vent = W_st
        self.W_break = W_liq + W_st
        self.Q_sink = Q_sink
        self._level_flags()

    def _level_flags(self):
        rt, t = self.rt, self.t
        lvl_flags = [("uncovered", self.M < rt.M_top * 1e3, "core uncovery begins (level at top of fuel)"),
                     ("dry", self.M < rt.M_bot * 1e3 + 0.1 * (rt.M_top - rt.M_bot) * 1e3,
                      "core nearly dry (level near bottom of fuel)")]
        for key, cond, msg in lvl_flags:
            if cond and key not in self.flags:
                self.flags.add(key)
                self._log(t, msg)
            if not cond and key in self.flags and self.M > rt.M_top * 1e3 + 2e3:
                self.flags.discard(key)
                if key == "uncovered":
                    self._log(t, "core recovered by water")

    def _plant_single_phase(self, Q, dt):
        rt = self.rt
        cp = float(self.coolant["cp"](np.array(self.T_in)))
        C = rt.M_nom * 1e3 * cp + rt.C_struct_MJK * 1e6
        g = min(self.f, 1.0) ** (0.8 if rt.coolant == "helium" else 0.6)
        T_out = getattr(self, "T_out", np.array([self.T_in]))
        Wc = getattr(self, "W_cols", None)
        T_hot = float(np.average(T_out, weights=Wc)) if Wc is not None and np.size(T_out) == np.size(Wc) \
            else float(np.mean(T_out))
        T_avg = 0.5 * (self.T_in + T_hot)
        Q_sink = self.UA_sg * g * (T_avg - self.T_sec) if rt.has_sg else 0.0
        Q_aux = rt.UA_aux * max(T_hot - rt.T_aux, 0.0)
        K = self.UA_sg * g if rt.has_sg else 0.0
        delta = T_avg - self.T_in
        T_new = (C / dt * self.T_in + Q - Q_aux - K * (delta - self.T_sec)) / (C / dt + K)
        Q_sink = K * (T_new + delta - self.T_sec)
        self.T_in = float(np.clip(T_new, 300.0, 1400.0))
        if rt.coolant == "helium" and self.break_area > 0:
            tau = max(0.3 / self.break_area, 5.0)
            self.p = rt.p_back + (self.p - rt.p_back) * math.exp(-dt / tau)
        self.Q_sink = Q_sink + Q_aux
        self.W_break = 0.0
        self.W_inj = 0.0

    # ------------------------------------------------------------------ damage process
    def _damage(self, dt, quench):
        rt = self.rt
        F = self.F
        H = np.interp(self.Tf, _T_TAB, self.h_tab) / CAL_PER_G
        H = np.where(F, H, 0.0) + self.melt * self.fuel_props["L"] / CAL_PER_G
        self.H_fuel = H
        ecr = np.where(self.m_zr0 > 0, 1 - self.m_zr / np.maximum(self.m_zr0, 1e-12), 0.0)
        self.ecr = ecr
        has = self.state_fuel & (self.m > 1e-6)

        def rates():
            below_ = np.where(self.lat.below >= 0, self.state[np.maximum(self.lat.below, 0)], -1)
            lam, tgt = dmg.hazards(rt, self.state, self.Tc, self.Tf, ecr, H, self.melt, below_, quench, H - self.H0)
            lam = np.where(has | (self.state == dmg.MOLTEN), lam, 0.0)
            lam[~F] = 0.0
            return lam, tgt

        lam, tgt = rates()
        if self.method == "tau-leap":
            evs = dmg.tau_leap(lam, tgt, dt, self.rng)
            for i, s, _ in sorted(evs, key=lambda e: self.lat.iz[e[0]]):
                self._transition(i, s)
        else:
            def recompute(_):
                return rates()
            evs = dmg.gillespie(lambda ev: (self._transition(ev[-1][0], ev[-1][1]) or recompute(ev)),
                                lam, tgt, dt, self.rng)
        self.lam_max = float(lam.max()) if lam.size else 0.0
        self.lam = lam

    def _transition(self, i, s):
        rt = self.rt
        old = int(self.state[i])
        if s == old:
            return
        tnow = self.t
        if s == dmg.FAILED and self.inv_gap[i]:
            self.inv[i] *= (1 - chem.GAP_FRACTION)
            self.inv_gap[i] = False
        if s == dmg.RELOCATED:
            self._relocate(i)
        else:
            self.state[i] = s
        if s == dmg.DAMAGED and self.H_fuel[i] > rt.H_fragment and self.m[i] > 0 and self.n > 0.5:
            # energetic fragmentation in a power excursion: fine fuel particles are swept out of the
            # lattice with the coolant / steam (fuel-coolant interaction), taking their fissile content
            lost = 0.7 * self.m[i]
            self.m[i] -= lost
            self.dispersed = getattr(self, "dispersed", 0.0) + lost
            if "dispersal" not in self.flags:
                self.flags.add("dispersal")
                self._log(tnow, "fuel fragmentation: fragmented fuel dispersed from the lattice")
        self._shape_stale = True
        counts = np.bincount(self.state[self.F], minlength=5)
        first = {dmg.FAILED: ("first_fail", "first cladding failure"),
                 dmg.DAMAGED: ("first_damage", "first loss of fuel-rod geometry"),
                 dmg.MOLTEN: ("first_melt", "first molten material"),
                 dmg.RELOCATED: ("first_reloc", "first relocation of molten material")}
        if s in first and first[s][0] not in self.flags:
            self.flags.add(first[s][0])
            ix, iz = self.lat.ix[i], self.lat.iz[i]
            self._log(tnow, f"{first[s][1]} (column {ix + 1}, level {iz + 1})")
        del counts

    def _relocate(self, i):
        lat = self.lat
        j = lat.below[i]
        while j >= 0 and self.state[j] == dmg.RELOCATED:
            j = lat.below[j]
        cpf = self.fuel_props["cp"]
        if j >= 0:
            Ci = self.m_fuel0[i] * self.m[i] * cpf(self.Tf[i])
            Cj = self.m_fuel0[j] * self.m[j] * cpf(self.Tf[j])
            if Ci + Cj > 0:
                self.Tf[j] = (Ci * self.Tf[i] + Cj * self.Tf[j]) / (Ci + Cj)
                self.melt[j] = (self.melt[i] * self.m[i] + self.melt[j] * self.m[j]) / max(self.m[i] + self.m[j], 1e-12)
            self.m[j] += self.m[i]
            ccl = self.m_clad[i]
            self.Tc[j] = (ccl * self.Tc[i] + self.m_clad[j] * self.Tc[j]) / max(ccl + self.m_clad[j], 1e-9)
            self.m_clad[j] += self.m_clad[i]
            self.m_zr[j] += self.m_zr[i]
            self.m_zr0[j] += self.m_zr0[i]
            self.inv[j] += self.inv[i]
            self.decay.move(i, j)
            if self.state[j] < dmg.DAMAGED:
                self.state[j] = dmg.DAMAGED
        else:
            frac = self.m[i] / max(self.lat.n_fuel, 1)
            Ci = self.m_fuel0[i] * self.m[i] * cpf(self.Tf[i])
            LH = self.LH
            if LH["C"] + Ci > 0:
                LH["T"] = (LH["C"] * LH["T"] + Ci * self.Tf[i]) / (LH["C"] + Ci)
            LH["C"] += Ci
            LH["m"] += frac
            self.LH_decay.y[:, 0] += self.decay.y[:, i]
            self.LH_decay.U[0] += self.decay.U[i]
            self.LH_decay.Np[0] += self.decay.Np[i]
            self.decay.move(i, -1)
            self.inv_LH = getattr(self, "inv_LH", 0.0) + self.inv[i]
            if "lower_head" not in self.flags:
                self.flags.add("lower_head")
                self._log(self.t, "molten material reaches the lower head")
        self.m[i] = 0.0
        self.m_clad[i] = 0.0
        self.m_zr[i] = 0.0
        self.m_zr0[i] = 0.0
        self.inv[i] = 0.0
        self.melt[i] = 0.0
        self.absorber_ok[i] = 0.0
        self.state[i] = dmg.RELOCATED

    def _lower_head(self, dt):
        LH = self.LH
        self.Q_LH_water = 0.0
        if LH["m"] <= 0:
            return
        rt = self.rt
        P = float(self.LH_decay.power()[0])
        self.LH_decay.step(np.zeros(1), dt)
        wet = self.water and self.M > rt.M_bot * 1e3 * 0.6
        T = LH["T"]
        if wet:
            Ts = float(mat.t_sat(self.p))
            Q_out = 2.0e3 * 20.0 * LH["m"] ** 0.5 * max(T - Ts, 0.0) + P * 0.0
            self.Q_LH_water = min(Q_out, LH["C"] * max(T - Ts, 0) / dt + P)
        else:
            self.Q_LH_water = 0.0
        Q_rad = 0.5 * mat.SIGMA_SB * 25.0 * (T ** 4 - 600.0 ** 4)
        LH["T"] = T + (P - self.Q_LH_water - max(Q_rad, 0)) / max(LH["C"], 1.0) * dt
        if not LH["failed"]:
            lam = _sig((LH["T"] - 1600.0) / 50.0) * _sig((LH["m"] - 0.05) / 0.02) / (rt.lower_head_tau_h * 3600.0)
            if self.rng.random() < -math.expm1(-lam * dt):
                LH["failed"] = True
                self._log(self.t, "lower-head failure: molten core breaches the vessel")

    # ================================================================== recording and run loop
    def _log(self, t, msg):
        self.log.append((float(t), msg))

    def _components(self):
        """Reactivity split ($) by first-order perturbation on the current shape."""
        w, den = self.shape.w, self.shape.den
        lam = 1.0 / self.shape.k
        pr = self._parts
        nuSf = self.nuSf0 * self.m
        c = {}
        if not den > 0:
            return {k: 0.0 for k in ("doppler", "moderator", "void", "radial", "displacer", "xenon",
                                     "samarium", "rods", "control")}
        for key in ("doppler", "moderator", "void", "radial", "displacer"):
            c[key] = float(w @ (lam * nuSf * pr[key])) / den / self.beta
        for key in ("xenon", "samarium", "rods"):
            c[key] = -float(w @ pr[key]) / den / self.beta
        c["control"] = -float(w @ np.where(self.F, self.Sc, 0.0)) / den / self.beta
        return c

    def _snapshot(self):
        F = self.F
        Pf = self.n_avg * self.P_fis_nom * self.pshape                  # fission power per cell (W)
        P = Pf + self.decay.power()
        nominal = self.rt.power_MW * 1e6 / max(self.lat.n_fuel, 1)
        Sa_tot = self.Sa0 * (0.5 + 0.5 * self.m) + self.Sc
        nan = np.nan
        lam = getattr(self, "lam", None)
        H = getattr(self, "H_fuel", None)
        fields = dict(
            power=np.where(F, P / nominal, nan),
            q3=np.where(F, P / self.V_cell, nan),                                   # W/cm3 of core
            q1=np.where(F, P / max(self.L_cell, 1e-9) / 1e3, nan),                  # kW/m of rod
            flux=np.where(F, self._flux(self.n_avg), nan),                          # n/cm2 s
            T_fuel=np.where(F, self.Tf, nan),
            T_clad=np.where(F, self.Tc, nan) if not self.rt.fuel_pattern else self.Tc.copy(),
            T_mod=self.Tm.copy() if self.has_mod else np.where(F, self.Tcool, nan),
            T_cool=np.where(F, self.Tcool, nan), void=np.where(F, self.void * 100, nan),
            xenon=np.where(F, self.rt.sigma_xe * self.poisons.X / np.maximum(Sa_tot, 1e-12) * 100, nan),
            Xe=np.where(F, self.poisons.X, nan), I=np.where(F, self.poisons.I, nan),
            Sm=np.where(F, self.poisons.S, nan),
            state=np.where(F, self.state, nan).astype(float), ecr=np.where(F, self.ecr * 100, nan),
            hazard=np.where(F, lam if lam is not None else 0.0, nan),
            enthalpy=np.where(F, H if H is not None else 0.0, nan),
            fuel=np.where(F, self.m * self.m_fuel0, nan),
            release=np.where(F, np.maximum(1 - self.inv * self.lat.n_fuel, 0.0) * 100, nan),
            wet=np.where(F, self.wet, nan))
        return Snapshot(self.t, {k: np.asarray(v, float).astype(np.float32) for k, v in fields.items()})

    def _record(self, S):
        F = self.F
        Pf = self.n_avg * self.P_fis_nom / 1e6
        Pd = float(self.decay.power().sum()) / 1e6 + float(self.LH_decay.power()[0]) / 1e6
        comp = self._components()
        counts = np.bincount(self.state[F].astype(int), minlength=5) / max(self.lat.n_fuel, 1)
        hasf = F & (self.m > 1e-6)
        Tfmax = float(self.Tf[hasf].max()) if hasf.any() else float("nan")
        p = self.pshape.reshape(self.lat.nz, self.lat.nx).sum(1)
        half = self.lat.nz // 2
        AO = (p[half:].sum() - p[:half].sum()) / max(p.sum(), 1e-12)
        dn = getattr(self, "dnbr", np.array([np.inf]))
        rel = 1 - float(self.inv.sum()) - 0.0
        vals = dict(
            n=self.n_avg, P_fission=Pf, P_decay=Pd, P_total=Pf + Pd,
            rho=self.rho / self.beta, k_eff=self.shape.k, **{"rho_" + k: v for k, v in comp.items()},
            T_fuel_max=Tfmax, T_fuel_avg=float(np.average(self.Tf[hasf], weights=self.m[hasf])) if hasf.any() else np.nan,
            T_clad_max=float(self.Tc[hasf].max()) if hasf.any() else np.nan,
            T_mod_avg=float(self.Tm[F].mean()), T_cool_max=float(self.Tcool[F].max()), T_in=self.T_in,
            void_avg=float(self.void[F].mean()), H_fuel_max=float(self.H_fuel[F].max()) if hasattr(self, "H_fuel") else 0.0,
            p=self.p, M=self.M / 1e3, level=getattr(self, "level", 0.0), M_sec=self.M_sec / 1e3, flow=self.f,
            W_inj=getattr(self, "W_inj", 0.0), Q_sink=getattr(self, "Q_sink", 0.0) / 1e6,
            H2=self.H2, Zr_ox=self.Zr_ox / max(float(self.m_zr0_total), 1e-9) * 100 if self.m_zr0_total else 0.0,
            ECR_max=float(self.ecr[F].max()) * 100 if hasattr(self, "ecr") else 0.0,
            Q_ox=self.q_ox_tot / 1e6, release=max(rel, 0.0) * 100, LH=self.LH["m"] * 100, T_LH=self.LH["T"] if self.LH["m"] > 0 else np.nan,
            DNBR=float(np.min(dn)) if np.isfinite(np.min(dn)) else np.nan, AO=AO * 100, ctrl=self.Sc,
            M_mod=self.M_mod / 1e3 if self.has_mod else 0.0,
            frac_intact=counts[0], frac_failed=counts[1], frac_damaged=counts[2], frac_molten=counts[3],
            frac_relocated=counts[4])
        for k, v in vals.items():
            S.setdefault(k, []).append(float(v))

    def run(self, progress=None, cancelled=None, live=None, max_wall=None):
        """Integrate to the end time. progress(frac), cancelled() -> bool, live(snapshot, partial)."""
        rt = self.rt
        wall0 = _time.time()
        self.m_zr0_total = float(self.m_zr0.sum())
        self._shape_stale = False
        self.ecr = np.zeros(self.N)
        self.H_fuel = np.where(self.F, np.interp(self.Tf, _T_TAB, self.h_tab) / CAL_PER_G, 0.0)
        self.H0 = self.H_fuel.copy()
        self.rho = 0.0
        self.q_ox_tot = 0.0
        S = {}
        t_list = []
        snaps = []
        self._record(S)
        t_list.append(0.0)
        snaps.append(self._snapshot())
        evs = list(self.events)
        ei = 0
        dt = min(1e-3 if self.sc.dt_max < 0.1 else 0.05, self.sc.dt_max)
        n_snap = 360
        self.t_state = np.full((5, self.N), np.inf)
        self.t_state[0] = 0.0
        snap_dt = self.t_end / n_snap
        next_snap = snap_dt
        last_live = _time.time()
        nstep = 0
        dt_min = 1e-5
        while self.t < self.t_end - 1e-9:
            while ei < len(evs) and evs[ei].t <= self.t + 1e-9:
                self._apply_event(evs[ei])
                ei += 1
            self._protection()
            if self.t >= self.t_end - 1e-9:
                break
            # step limits
            limit = self.sc.dt_max
            if self.t < 1.0 and self.sc.dt_max > 1.0:
                limit = min(limit, 0.2)
            if ei < len(evs):
                limit = min(limit, max(evs[ei].t - self.t, dt_min))
            if self.scram_t is not None and self.t < self.scram_t + rt.scram_time:
                limit = min(limit, 0.02 if rt.scram_time < 5 else 0.05)
            for b in self.banks.values():
                if b["depth"] != b["target"]:
                    limit = min(limit, max(0.02 / max(b["speed"], 1e-9), 1e-3))
            if self.method == "tau-leap" and getattr(self, "lam_max", 0) > 0:
                limit = min(limit, 0.3 / self.lam_max)
            r_prev = getattr(self, "rho", 0.0)
            if r_prev > 0:
                tau = rt.Lambda / (r_prev - self.beta) if r_prev > 0.9 * self.beta else \
                    (self.beta - r_prev) / (0.08 * r_prev) + rt.Lambda / max(r_prev, 1e-9)
                limit = min(limit, max(0.25 * tau, 2 * rt.Lambda))
            if self.n_avg > 0.02 and not (self.controller and self._held_power_ok(max(limit, 60.0))):
                # at power the feedback loops (void, Doppler, moderator) act on the fuel time constant
                limit = min(limit, 10.0)
            lat_lim = self._lateral_limit()
            limit = min(limit, lat_lim)
            dt = float(np.clip(dt, dt_min, limit))
            dt = min(dt, self.t_end - self.t)
            n0, Tf0, Tc0, rho0, M0 = self.n, self.Tf.copy(), self.Tc.copy(), getattr(self, "rho", 0.0), self.M
            ps0 = self.pshape.copy()
            self.step(dt)
            nstep += 1
            for s_ in range(1, 5):
                new = self.F & (self.state >= s_) & np.isinf(self.t_state[s_])
                if new.any():
                    self.t_state[s_][new] = self.t
            # adapt the next step
            # relative power change; a rising power is always resolved, a decaying tiny one is not
            dn = abs(math.log(max(self.n, 1e-30) / max(n0, 1e-30)))
            if not (self.n > n0 and self.rho > 0):
                dn *= min(1.0, max(self.n, n0) / 0.02)
            dT = float(np.max(np.abs(self.Tf - Tf0)[self.F])) if self.F.any() else 0.0
            dTc = float(np.max(np.abs(self.Tc - Tc0)))
            dr = abs(self.rho - rho0) / self.beta / (1.0 + max(-min(self.rho, rho0) / self.beta, 0.0))
            dM = abs(self.M - M0) / max(self.rt.M_nom * 1e3, 1.0)
            # change of the power shape (spatial modes, xenon tilts) between steps
            dP = float(np.max(np.abs(self.pshape - ps0))) / max(float(ps0.max()), 1e-30) if self.n_avg > 1e-3 else 0.0
            ratio = max(dn / 0.08, dT / 30.0, dTc / 30.0, dr / 0.05, dM / 0.01, dP / 0.03, 1e-3)
            dt = dt * float(np.clip(0.9 / ratio, 0.3, 1.6))
            if self.t >= next_snap - 1e-9 or self.t >= self.t_end - 1e-9:
                snaps.append(self._snapshot())
                next_snap = (int(self.t / snap_dt) + 1) * snap_dt
            self._record(S)
            t_list.append(self.t)
            if progress is not None and nstep % 20 == 0:
                progress(self.t / self.t_end)
            if cancelled is not None and nstep % 10 == 0 and cancelled():
                raise Cancelled
            if live is not None and _time.time() - last_live > 0.25:
                last_live = _time.time()
                live(self._partial(t_list, S, snaps))
            if max_wall and _time.time() - wall0 > max_wall:
                self._log(self.t, "stopped: wall-clock limit")
                break
            if self.LH["failed"] and "stop_after_breach" not in self.flags:
                self.flags.add("stop_after_breach")
        res = self._partial(t_list, S, snaps)
        res.runtime = _time.time() - wall0
        res.summary["steps"] = nstep
        return res

    def _C_outer(self):
        if self.has_mod and self.rt.moderator == "graphite":
            return self.m_mod * self.cp_mod(self.Tm)
        return self.m_clad * self.clad_props["cp"](self.Tc) + self.m_struct * self.rt.struct_cp

    def _lateral_limit(self):
        G = getattr(self, "_G_lat_sum", None)
        if G is None:
            return np.inf
        C = self._C_outer()
        G = G.ravel()
        m = G > 0
        if not m.any():
            return np.inf
        return float(0.45 * np.min(np.maximum(C[m], 1.0) / G[m]))

    def _partial(self, t_list, S, snaps):
        series = {k: np.asarray(v) for k, v in S.items()}
        summ = self._summary(series)
        return Result(self.sc.key, self.rt.key, self.lat.nx, self.lat.nz, np.asarray(t_list), series,
                      list(snaps), list(self.log), summ, self.F.copy(), cell_times=self.t_state.copy())

    def _summary(self, s):
        def mx(k):
            v = s.get(k)
            return float(np.nanmax(v)) if v is not None and len(v) and np.isfinite(v).any() else float("nan")
        out = dict(peak_power=mx("n"), peak_fuel_T=mx("T_fuel_max"), peak_clad_T=mx("T_clad_max"),
                   peak_enthalpy=mx("H_fuel_max"), H2_kg=float(s["H2"][-1]) if "H2" in s else 0.0,
                   ECR_max=mx("ECR_max"),
                   molten_max=mx("frac_molten"), relocated=float(s["frac_relocated"][-1]) if "frac_relocated" in s else 0.0,
                   damaged_or_worse=float(s["frac_damaged"][-1] + s["frac_molten"][-1] + s["frac_relocated"][-1])
                   if "frac_damaged" in s else 0.0,
                   failed_or_worse=float(1 - s["frac_intact"][-1]) if "frac_intact" in s else 0.0,
                   release=float(s["release"][-1]) if "release" in s else 0.0,
                   lower_head=self.LH["m"] * 100, vessel_failed=bool(self.LH["failed"]),
                   min_DNBR=float(np.nanmin(s["DNBR"])) if "DNBR" in s and np.isfinite(s["DNBR"]).any() else float("nan"))
        firsts = {}
        for t, msg in self.log:
            for key in ("first cladding failure", "first loss", "first molten", "first relocation",
                        "core uncovery", "lower-head failure", "scram"):
                if msg.startswith(key) and key not in firsts:
                    firsts[key] = t
        out["times"] = firsts
        return out


def run(scenario, **kw):
    return Simulator(scenario, **{k: v for k, v in kw.items() if k in ("params", "method", "seed", "nx", "nz", "overrides")}).run()
