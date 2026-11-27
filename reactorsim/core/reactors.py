"""Reactor designs: geometry, thermal hydraulics, neutronics, feedback, protection and damage data.

Each design is a vertical slice through the core, discretised into nx columns (fuel channels or
assembly rows) and nz axial levels. Numbers are representative of the named plant class at
hot full power, mid-cycle; references are listed per design and collected in the README.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np


@dataclass
class ECCSystem:
    name: str
    p_max: float            # injects only below this pressure (MPa)
    flow: float             # capacity (kg/s)
    volume: float           # water available (t); np.inf for an unlimited source
    T: float = 310.0        # injected water temperature (K)
    level_control: bool = True   # throttled to hold inventory at nominal (pumped systems)
    needs_ac: bool = True   # lost in station blackout
    needs_dc: bool = False  # lost when batteries deplete
    p_min: float = 0.0      # steam-driven systems stall below this pressure


@dataclass
class ReactorType:
    key: str
    name: str
    description: str
    refs: list

    # ---------------------------------------------------------------- core and fuel
    power_MW: float
    height_m: float
    width_m: float                      # equivalent core diameter (slice width)
    nx: int = 16
    nz: int = 18
    fuel_pattern: str = ""              # per column: 'F' fuel, 'R' graphite reflector
    fuel: str = "UO2"
    clad: str = "zircaloy"
    n_rods: int = 50952                 # rods (or compact stacks) spanning the full height
    r_pellet: float = 4.096e-3
    r_clad_in: float = 4.178e-3
    r_clad_out: float = 4.75e-3
    h_gap: float = 5700.0               # pellet-clad gap conductance (W/m2 K)
    smear: float = 1.0                  # fuel smear density fraction (metal fuel 0.75)
    zr_extra: float = 0.0               # extra Zr per unit clad Zr (channel boxes, pressure tubes)
    struct_mass_t: float = 0.0          # extra structure lumped into the clad node (t)
    struct_cp: float = 500.0
    A_conv_m2: float | None = None      # convective area override (HTGR coolant holes)
    G_fc_override: float | None = None  # fuel->clad conductance override, whole core (W/K)
    k_lateral: float = 1.0              # effective cell-to-cell conductivity of the outer node (W/m K)
    lateral_radiation: bool = True      # add 4 sigma T^3 * pitch radiative conductivity
    pitch: float = 0.0126
    UA_boundary: float = 2.0e3          # core edge -> vessel / cavity sink (W/K)
    T_boundary: float = 330.0

    # ---------------------------------------------------------------- moderator / graphite node
    moderator: str | None = None        # None, 'graphite', 'd2o'
    mod_mass_t: float = 0.0
    mod_power_frac: float = 0.0         # fraction of fission energy deposited in the moderator
    mod_T_nom: float = 340.0
    G_mod_cool: float = 0.0             # moderator -> coolant conductance, whole core (W/K)
    G_clad_mod: float = 0.0             # clad -> moderator conductance, whole core (W/K)
    rad_area_mod: float = 0.0           # clad -> moderator radiating area (m2), CANDU tubes
    mod_sink_MW: float = 0.0            # moderator heat exchangers (CANDU)
    feedback_mod_node: str = "coolant"  # which temperature drives alpha_mod: coolant/moderator/clad

    # ---------------------------------------------------------------- coolant system
    coolant: str = "water"
    flow_direction: str = "up"          # 'up' (vertical channels) or 'cross' (CANDU horizontal)
    p_nom: float = 15.5
    p_relief: float = 16.2
    p_back: float = 0.2                 # containment / suppression pool pressure (MPa)
    T_in: float = 565.0
    flow: float = 17700.0               # core flow (kg/s)
    h_forced: float = 34000.0           # nominal forced-convection coefficient (W/m2 K)
    pump_tau: float = 12.0              # pump coast-down time constant (s)
    nat_circ: float = 0.04              # natural-circulation flow fraction
    boiling_core: bool = False          # BWR / RBMK direct cycle
    T_feed: float = 489.0               # feedwater temperature (boiling types) (K)
    nc_level: float = 0.6               # natural circulation needs M > M_top + nc_level (M_nom - M_top)
    slip: float = 1.6
    void_ref: float = 0.0               # core-average void at nominal
    M_nom: float = 260.0                # primary inventory (t)
    M_top: float = 60.0                 # inventory with level at top of active fuel (t)
    M_bot: float = 45.0                 # inventory with level at bottom of active fuel (t)
    flow_area_m2: float = 5.0
    C_struct_MJK: float = 80.0          # primary metal heat capacity (MJ/K)
    has_sg: bool = True
    sg_mass_t: float = 200.0            # secondary water in all steam generators (t)
    p_sec: float = 7.6                  # SG safety-valve pressure (MPa)
    afw_flow: float = 60.0              # auxiliary feedwater (kg/s)
    afw_turbine: bool = True            # steam-driven AFW survives an SBO (on DC power)
    eccs: list = field(default_factory=list)
    UA_aux: float = 0.0                 # passive decay-heat removal (IC, DRACS) (W/K)
    T_aux: float = 373.0
    battery_h: float = 8.0
    rcp_seal_leak: float = 0.0          # kg/s once pumps lose seal cooling (SBO)

    # ---------------------------------------------------------------- neutronics
    excess: float = 0.05                # reactivity with all control absorbers out at hot full power,
                                        # equilibrium Xe/Sm (calibrates k_inf); the controller's range
    k_inf: float = 1.15                 # initial guess for that calibration
    M2: float = 60.0                    # migration area (cm2)
    phi_nom: float = 3.0e13             # average thermal flux (n/cm2 s)
    fissions: dict = field(default_factory=lambda: {"U235": 0.65, "Pu239": 0.30, "U238": 0.05})
    beta_eff: float = 0.0060
    Lambda: float = 2.0e-5
    sigma_xe: float = 2.6e-18           # spectrum-averaged effective cross sections (cm2)
    sigma_sm: float = 5.0e-20
    reflector_savings_cm: float = 8.0
    k_inf_edge_boost: float = 0.03      # loading-pattern flattening
    refl_D: float = 0.9
    refl_Sa: float = 3e-4

    # ---------------------------------------------------------------- feedback (dk/k)
    alpha_D: float = -2.5e-5            # Doppler at T_f_ref (per K), sqrt(T) law
    alpha_fexp: float = 0.0             # fuel axial expansion (per K of fuel)
    alpha_mod: float = -2.5e-4          # moderator / coolant temperature (per K)
    alpha_void: float = -0.15           # per unit void fraction
    alpha_radial: float = 0.0           # core radial expansion (per K of inlet temperature)
    T_f_ref: float = 900.0
    T_mod_ref: float = 580.0

    # ---------------------------------------------------------------- control and protection
    scram_worth: float = 8.0            # $
    scram_delay: float = 0.5
    scram_time: float = 2.2             # full-stroke insertion time (s)
    rods_from: str = "top"
    displacer: bool = False             # RBMK graphite displacers (positive scram effect)
    displacer_dk: float = 0.0
    rod_absorber_fail_T: float = 1100.0 # control-rod material failure / relocation (K)
    debris_k: float = 0.9               # k_inf factor of molten / relocated debris (lattice geometry lost)
    boron_ppm: float = 0.0              # soluble boron in the coolant at the initial state
    eccs_boron_ppm: float = 0.0         # boron in emergency injection water
    boron_worth_pcm: float = -8.0       # pcm per ppm

    # ---------------------------------------------------------------- damage progression
    T_clad_fail: float = 1100.0         # burst / failure, 50 % hazard point (K)
    T_clad_fail_w: float = 40.0
    clad_fail_rate: float = 0.5         # hazard at saturation (1/s)
    H_pcmi_fail: float = 150.0          # radial-average fuel enthalpy for PCMI failure (cal/g)
    H_fragment: float = 230.0           # fuel fragmentation / loss of coolable geometry (cal/g)
    T_damage: float = 1477.0            # embrittlement criterion (PCT, 10 CFR 50.46)
    ECR_limit: float = 0.17
    T_liquefy: float = 2550.0           # U-Zr-O liquefaction (K)
    T_liquefy_w: float = 70.0
    melt_rate: float = 0.05
    reloc_tau: float = 90.0             # molten material slumps on this time scale (s)
    damage_rate: float = 0.01           # failed -> damaged hazard at saturation (1/s)
    lower_head_tau_h: float = 1.5       # dry lower head fails on this time scale once loaded
    core_volume_m3: float | None = None
    t_op_days: float = 600.0            # operating time for decay heat
    R_capture: float = 0.6              # U-239 atoms per fission

    # ---------------------------------------------------------------- derived helpers
    def fuel_columns(self) -> np.ndarray:
        pat = self.fuel_pattern or "F" * self.nx
        return np.array([c == "F" for c in pat], bool)

    def with_lattice(self, nx=None, nz=None):
        if nx is None and nz is None:
            return self
        nx = nx or self.nx
        pat = self.fuel_pattern
        if pat:
            idx = (np.arange(nx) + 0.5) / nx * len(pat)
            pat = "".join(pat[int(i)] for i in idx)
        return replace(self, nx=nx, nz=nz or self.nz, fuel_pattern=pat)


# ======================================================================== designs
PWR_ECCS = [
    ECCSystem("HPI", p_max=12.5, flow=50.0, volume=1500.0, T=305.0),
    ECCSystem("Accumulators", p_max=4.2, flow=900.0, volume=100.0, T=310.0,
              level_control=False, needs_ac=False),
    ECCSystem("LPI", p_max=1.4, flow=450.0, volume=1500.0, T=305.0),
]

PWR = ReactorType(
    key="pwr", excess=0.060,  name="PWR (4-loop, 3411 MWt)",
    description="Pressurized light-water reactor, 193 17x17 assemblies, Zircaloy-clad UO2, "
                "borated light-water coolant/moderator at 15.5 MPa. Strongly negative moderator "
                "temperature and void coefficients; I-135/Xe-135 drive axial xenon oscillations.",
    refs=["westinghouse", "todreas", "duderstadt", "ans51", "cathcart", "nureg1465"],
    power_MW=3411.0, height_m=3.66, width_m=3.37, nx=16, nz=18,
    n_rods=50952, r_pellet=4.096e-3, r_clad_in=4.178e-3, r_clad_out=4.75e-3,
    zr_extra=0.15, struct_mass_t=25.0, k_lateral=2.0,
    p_nom=15.5, p_relief=16.2, T_in=565.0, flow=17700.0, h_forced=34000.0, pump_tau=12.0,
    nat_circ=0.04, M_nom=260.0, M_top=60.0, M_bot=45.0, flow_area_m2=4.75, C_struct_MJK=90.0,
    has_sg=True, sg_mass_t=200.0, p_sec=7.6, afw_flow=60.0, afw_turbine=True, eccs=PWR_ECCS,
    rcp_seal_leak=5.0, battery_h=8.0,
    k_inf=1.15, M2=60.0, phi_nom=3.0e13, beta_eff=0.0060, Lambda=2.0e-5,
    fissions={"U235": 0.62, "Pu239": 0.31, "U238": 0.07},
    alpha_D=-2.6e-5, alpha_mod=-2.5e-4, alpha_void=-0.15, T_f_ref=900.0, T_mod_ref=582.0,
    scram_worth=8.0, scram_delay=0.4, scram_time=2.2, rods_from="top",
    rod_absorber_fail_T=1100.0, boron_ppm=900.0, eccs_boron_ppm=2400.0, boron_worth_pcm=-8.0,
)

BWR = ReactorType(
    key="bwr", excess=0.030,  name="BWR/4 (2381 MWt)",
    description="Boiling-water reactor with jet-pump recirculation, 548 8x8 assemblies in Zircaloy "
                "channel boxes, direct steam cycle at 7 MPa, Mark I suppression pool. Bottom-entry "
                "cruciform B4C blades. Negative void coefficient couples power to flow and pressure.",
    refs=["bwr_tech", "todreas", "fukushima_inpo", "gauntt", "ans51", "cathcart"],
    power_MW=2381.0, height_m=3.71, width_m=4.03, nx=16, nz=18,
    n_rods=32880, r_pellet=5.21e-3, r_clad_in=5.32e-3, r_clad_out=6.15e-3,
    zr_extra=0.8, struct_mass_t=20.0, k_lateral=2.0,
    p_nom=7.0, p_relief=7.4, T_in=549.0, flow=7700.0, h_forced=30000.0, pump_tau=6.0,
    nat_circ=0.30, boiling_core=True, T_feed=489.0, nc_level=0.3, slip=1.8, void_ref=0.40,
    M_nom=230.0, M_top=140.0, M_bot=115.0, flow_area_m2=9.0, C_struct_MJK=110.0,
    has_sg=False, sg_mass_t=0.0, afw_flow=0.0, afw_turbine=False,
    eccs=[ECCSystem("HPCI", p_max=8.0, flow=268.0, volume=1500.0, T=310.0, needs_ac=False,
                    needs_dc=True, p_min=1.0),
          ECCSystem("RCIC", p_max=8.0, flow=38.0, volume=np.inf, T=310.0, needs_ac=False,
                    needs_dc=True, p_min=0.4),
          ECCSystem("Core spray / LPCI", p_max=2.0, flow=700.0, volume=np.inf, T=305.0)],
    battery_h=8.0,
    k_inf=1.12, M2=120.0, phi_nom=3.0e13, beta_eff=0.0055, Lambda=4.0e-5,
    fissions={"U235": 0.60, "Pu239": 0.32, "U238": 0.08},
    alpha_D=-2.5e-5, alpha_mod=-1.0e-5, alpha_void=-0.10, T_f_ref=850.0, T_mod_ref=556.0,
    scram_worth=15.0, scram_delay=0.3, scram_time=3.5, rods_from="bottom",
    rod_absorber_fail_T=1500.0,
)

RBMK = ReactorType(
    key="rbmk", debris_k=0.8, excess=0.012,  name="RBMK-1000 (1986 design)",
    description="Graphite-moderated, light-water-cooled pressure-tube reactor: 1661 vertical channels "
                "in a 1850 t graphite stack, boiling in the channels at 7 MPa. Large positive steam void "
                "coefficient at low operational reactivity margin; control rods carry 4.5 m graphite "
                "displacers below the absorber, so the first metre of a scram adds reactivity at the "
                "core bottom. Slow 18 s scram stroke.",
    refs=["insag7", "insag1", "rbmk_almenas", "ans51"],
    power_MW=3200.0, height_m=7.0, width_m=11.8, nx=16, nz=20,
    fuel="UO2", clad="zr-nb", n_rods=29898, r_pellet=5.75e-3, r_clad_in=5.9e-3, r_clad_out=6.8e-3,
    zr_extra=1.0, k_lateral=20.0, pitch=0.25,
    moderator="graphite", mod_mass_t=1850.0, mod_power_frac=0.055, mod_T_nom=850.0,
    G_mod_cool=0.60e6, feedback_mod_node="moderator",
    p_nom=7.0, p_relief=7.6, T_in=543.0, flow=10400.0, h_forced=28000.0, pump_tau=30.0,
    nat_circ=0.15, boiling_core=True, T_feed=438.0, nc_level=0.5, slip=1.5, void_ref=0.45,
    M_nom=1100.0, M_top=500.0, M_bot=300.0, flow_area_m2=8.6, C_struct_MJK=400.0,
    has_sg=False, sg_mass_t=0.0, afw_flow=0.0, afw_turbine=False, eccs=[],
    k_inf=1.07, M2=280.0, phi_nom=4.0e13, beta_eff=0.0048, Lambda=1.0e-3,
    fissions={"U235": 0.55, "Pu239": 0.40, "U238": 0.05}, reflector_savings_cm=60.0,
    k_inf_edge_boost=0.01,
    alpha_D=-1.2e-5, alpha_mod=-1.0e-5, alpha_void=+0.024, T_f_ref=900.0, T_mod_ref=850.0,
    scram_worth=10.0, scram_delay=0.3, scram_time=18.0, rods_from="top",
    displacer=True, displacer_dk=0.020,
    T_clad_fail=1050.0, H_pcmi_fail=150.0, H_fragment=220.0, rod_absorber_fail_T=1300.0,
)

RBMK_MOD = replace(
    RBMK, key="rbmk-mod", name="RBMK-1000 (post-1986 modifications)",
    description="RBMK after the post-Chernobyl back-fits: 2.4 % enrichment and extra fixed absorbers "
                "cut the void coefficient to about +0.7 beta, a minimum margin of 43-48 rods, redesigned "
                "rods without the lower water column and a fast scram system (2.5 s).",
    refs=["insag7", "rbmk_almenas", "ans51"],
    alpha_void=+0.0034, displacer=False, scram_time=2.5, scram_worth=14.0, k_inf=1.055,
)

CANDU = ReactorType(
    key="candu", debris_k=0.7, excess=0.015,  name="CANDU-6 (2064 MWt)",
    description="Pressurized heavy-water reactor: 380 horizontal Zircaloy pressure tubes with natural-UO2 "
                "37-element bundles inside a calandria of cool (340 K) D2O moderator, which is a heat "
                "sink of last resort. Positive coolant void reactivity, high thermal flux and natural "
                "uranium make it very sensitive to Xe-135 (poison-out after about 35 min).",
    refs=["candu_rouben", "candu6_tech", "ans51", "cathcart"],
    power_MW=2064.0, height_m=7.6, width_m=7.6, nx=16, nz=16,
    fuel="UO2", clad="zircaloy", n_rods=14060, r_pellet=6.075e-3, r_clad_in=6.12e-3,
    r_clad_out=6.54e-3, zr_extra=2.4, k_lateral=1.0,
    moderator="d2o", mod_mass_t=265.0, mod_power_frac=0.045, mod_T_nom=340.0,
    G_clad_mod=9.0e3, rad_area_mod=730.0, mod_sink_MW=110.0, feedback_mod_node="moderator",
    coolant="d2o", flow_direction="cross", p_nom=10.0, p_relief=11.0, T_in=539.0, flow=8400.0,
    h_forced=40000.0, pump_tau=15.0, nat_circ=0.04,
    M_nom=130.0, M_top=60.0, M_bot=30.0, flow_area_m2=1.5, C_struct_MJK=60.0,
    has_sg=True, sg_mass_t=170.0, p_sec=4.7, afw_flow=40.0, afw_turbine=False,
    eccs=[ECCSystem("HP ECC", p_max=5.5, flow=200.0, volume=200.0, T=305.0, level_control=False,
                    needs_ac=False),
          ECCSystem("MP/LP ECC", p_max=1.1, flow=200.0, volume=np.inf, T=310.0)],
    k_inf=1.07, M2=240.0, phi_nom=1.3e14, beta_eff=0.0058, Lambda=9.0e-4,
    fissions={"U235": 0.50, "Pu239": 0.45, "U238": 0.05}, reflector_savings_cm=55.0,
    k_inf_edge_boost=0.02,
    alpha_D=-5.0e-6, alpha_mod=+3.0e-5, alpha_void=+0.013, T_f_ref=1000.0, T_mod_ref=340.0,
    scram_worth=14.0, scram_delay=0.3, scram_time=1.5, rods_from="top",
)

HTGR = ReactorType(
    key="htgr", debris_k=1.0, excess=0.040, core_volume_m3=59.1,  name="HTGR (prismatic modular, 350 MWt)",
    description="Helium-cooled, graphite-moderated modular reactor with TRISO-coated particle fuel in "
                "prismatic blocks around an inner reflector. Low power density, a large graphite heat "
                "capacity and conduction to the reactor-cavity cooling system keep fuel below about "
                "1900 K after loss of forced cooling, without any active system.",
    refs=["mhtgr", "gougar", "ans51"],
    power_MW=350.0, height_m=7.93, width_m=5.6, nx=16, nz=16,
    fuel_pattern="RRRFFFRRRRFFFRRR",
    fuel="TRISO", clad="graphite", n_rods=13860, r_pellet=6.225e-3, r_clad_in=6.35e-3,
    r_clad_out=8.0e-3, h_gap=1200.0, struct_mass_t=420.0, struct_cp=1500.0,
    A_conv_m2=2823.0, k_lateral=10.0, pitch=0.035, UA_boundary=7.0e3, T_boundary=330.0,
    feedback_mod_node="clad",
    coolant="helium", p_nom=7.0, p_relief=7.5, p_back=0.1, T_in=532.0, flow=157.0, h_forced=2200.0,
    pump_tau=20.0, nat_circ=0.0, M_nom=5.0, M_top=0.0, M_bot=0.0, flow_area_m2=1.4,
    C_struct_MJK=200.0, has_sg=True, sg_mass_t=40.0, p_sec=17.0, afw_flow=5.0, afw_turbine=False,
    eccs=[],
    k_inf=1.15, M2=300.0, phi_nom=3.5e13, beta_eff=0.0065, Lambda=8.0e-4,
    fissions={"U235": 0.80, "Pu239": 0.18, "U238": 0.02}, reflector_savings_cm=10.0,
    k_inf_edge_boost=0.0,
    alpha_D=-4.0e-5, alpha_mod=-2.0e-5, alpha_void=0.0, T_f_ref=1100.0, T_mod_ref=900.0,
    scram_worth=10.0, scram_delay=0.5, scram_time=5.0, rods_from="top",
    T_clad_fail=2050.0, T_clad_fail_w=60.0, clad_fail_rate=2e-4, H_pcmi_fail=1e9, H_fragment=1e9,
    T_damage=2300.0, T_liquefy=2900.0, damage_rate=1e-4, rod_absorber_fail_T=2500.0,
)

SFR_METAL = ReactorType(
    key="sfr-metal", excess=0.010,  name="SFR, metal fuel (pool type, 1000 MWt)",
    description="Sodium-cooled fast reactor with U-Pu-Zr metal fuel in HT9 steel cladding, pool-type "
                "primary system near atmospheric pressure. Fast spectrum: negligible Xe-135, short "
                "neutron generation time, positive sodium-void worth offset by axial fuel expansion "
                "and radial core expansion, which give an inherent shutdown in unprotected loss of flow.",
    refs=["ebr2", "waltar", "ans51"],
    power_MW=1000.0, height_m=1.0, width_m=2.6, nx=14, nz=12,
    fuel="metal", clad="steel", n_rods=48780, r_pellet=2.9e-3, r_clad_in=3.19e-3,
    r_clad_out=3.75e-3, h_gap=60000.0, smear=0.75, struct_mass_t=30.0, k_lateral=5.0, pitch=0.009,
    coolant="sodium", p_nom=0.25, p_relief=0.4, p_back=0.1, T_in=628.0, flow=5080.0,
    h_forced=1.0e5, pump_tau=12.0, nat_circ=0.05,
    M_nom=800.0, M_top=0.0, M_bot=0.0, flow_area_m2=1.6, C_struct_MJK=300.0,
    has_sg=True, sg_mass_t=1e6, p_sec=0.0, afw_flow=0.0, afw_turbine=False, eccs=[],
    UA_aux=1.2e5, T_aux=400.0,
    k_inf=1.30, M2=250.0, phi_nom=1.0e13, beta_eff=0.0035, Lambda=4.0e-7,
    fissions={"U235": 0.05, "Pu239": 0.80, "U238": 0.15}, sigma_xe=1.0e-22, sigma_sm=1.0e-22,
    reflector_savings_cm=15.0, k_inf_edge_boost=0.05,
    alpha_D=-1.5e-6, alpha_fexp=-4.0e-6, alpha_mod=+2.0e-6, alpha_void=+0.017, alpha_radial=-1.2e-5,
    T_f_ref=850.0, T_mod_ref=705.0,
    scram_worth=12.0, scram_delay=0.2, scram_time=1.0, rods_from="top",
    T_clad_fail=1150.0, T_clad_fail_w=25.0, clad_fail_rate=0.05, H_pcmi_fail=1e9, H_fragment=1e9,
    T_damage=1250.0, T_liquefy=1430.0, T_liquefy_w=30.0, reloc_tau=20.0, rod_absorber_fail_T=1600.0,
    debris_k=1.0,
)

SFR_OXIDE = replace(
    SFR_METAL, key="sfr-oxide", name="SFR, oxide fuel (pool type, 1000 MWt)",
    description="Sodium-cooled fast reactor with (U,Pu)O2 fuel. The oxide runs much hotter (low "
                "conductivity) and stores more heat, axial expansion feedback is weak and the sodium "
                "void worth large, so an unprotected loss of flow can reach sodium boiling.",
    refs=["waltar", "ans51"],
    fuel="MOX", r_pellet=3.2e-3, r_clad_in=3.3e-3, r_clad_out=3.75e-3, h_gap=6000.0, smear=0.92,
    alpha_D=-7.0e-6, alpha_fexp=-1.0e-6, alpha_void=+0.025, alpha_radial=-6.0e-6, T_f_ref=1500.0,
    beta_eff=0.0036, T_liquefy=3000.0, T_liquefy_w=60.0, T_damage=1300.0, k_inf=1.28,
)

REACTORS = {r.key: r for r in (PWR, BWR, RBMK, RBMK_MOD, CANDU, HTGR, SFR_METAL, SFR_OXIDE)}


def get(key: str) -> ReactorType:
    return REACTORS[key]
