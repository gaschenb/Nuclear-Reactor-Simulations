"""Built-in accident and operational scenarios.

A scenario names a reactor design, an initial state (power, flow, optional power history that
sets the I/Xe/Pm/Sm inventory) and a list of timed events. Event arguments and times may refer
to scenario parameters as "$name"; the GUI exposes those parameters as sliders.

Event kinds
  scram                       reactor trip (rods after the design's delay and stroke time)
  pump_trip / pump_restart    main coolant pumps coast down / restart (args: frac)
  flow_ramp                   pumps to a new flow fraction (args: frac, tau)
  sbo                         loss of all AC: pumps, AC-powered injection, main feed, moderator cooling
  ac_restore                  AC power back
  loss_of_feed                main feedwater lost
  aux_feed                    auxiliary feedwater available (args: on)
  break                       primary break (args: area m2, loc 'liquid' | 'steam')
  depressurize                open relief/ADS valves (args: area m2)
  eccs                        enable/disable an injection system (args: name, on)
  inject                      ad-hoc injection (args: flow kg/s, T, p_max, name)
  turbine_trip                boiling types: lose the turbine (args: bypass fraction)
  msiv_close                  boiling types: isolate the vessel from the turbine
  rod_bank                    move a regulating bank (args: name, depth, speed)
  rod_eject                   eject the calibrated rod (args: time)
  setpoint                    power controller target (args: power); controller on/off (args: on)
  rps                         enable/disable the automatic protection system (args: on)
  stop                        end the run
"""
from __future__ import annotations

from dataclasses import dataclass, field

H = 3600.0


@dataclass
class Event:
    t: object
    kind: str
    args: dict = field(default_factory=dict)


@dataclass
class Scenario:
    key: str
    name: str
    reactor: str
    description: str
    t_end: object
    power0: float = 1.0
    flow0: float = 1.0
    feed0: float = 1.0
    history: list = field(default_factory=list)      # [(duration s, power fraction), ...] before t=0
    events: list = field(default_factory=list)
    rps: bool = True
    controller: bool = False                          # hold power with the uniform absorber
    dt_max: float = 20.0
    params: dict = field(default_factory=dict)        # name -> (default, lo, hi, label)
    banks: dict = field(default_factory=dict)         # name -> dict(cols, depth, worth, from)
    eject: dict | None = None                         # dict(col, worth $)
    eccs_off: list = field(default_factory=list)
    log_time: bool = False                            # plot time axes logarithmically
    core: dict = field(default_factory=dict)          # reactor-parameter overrides (may use "$param")
    tags: tuple = ()

    def value(self, v, params):
        if isinstance(v, str) and v.startswith("$"):
            return params[v[1:]]
        return v

    def resolved(self, params=None):
        p = {k: v[0] for k, v in self.params.items()}
        p.update(params or {})
        evs = []
        for e in self.events:
            evs.append(Event(float(self.value(e.t, p)), e.kind,
                             {k: self.value(v, p) for k, v in e.args.items()}))
        t_end = float(p["t_end"]) if "t_end" in p else float(self.value(self.t_end, p))
        return sorted(evs, key=lambda e: e.t), t_end, p


S = []

# ------------------------------------------------------------------------------- PWR
S += [
    Scenario("pwr-steady", "Steady full power", "pwr",
             "Hot full power with equilibrium xenon and samarium; reference state.",
             t_end=600.0, controller=True, dt_max=60.0),
    Scenario("pwr-lbloca", "Large-break LOCA, ECCS available", "pwr",
             "Double-ended cold-leg break. Blowdown empties the vessel in ~20 s, clad heats adiabatically "
             "until accumulators and low-pressure injection reflood the core. Design-basis: peak clad "
             "temperature should stay below 1477 K and ECR below 17 %.",
             t_end="$t_end", events=[Event(0.0, "break", dict(area="$area", loc="liquid")),
                                     Event(0.0, "pump_trip")],
             params=dict(area=(0.40, 0.01, 0.8, "break area (m²)"), t_end=(1200.0, 300, 7200, "duration (s)")),
             dt_max=2.0, tags=("design basis",)),
    Scenario("pwr-lbloca-nolpi", "Large-break LOCA, low-pressure injection failed", "pwr",
             "As the design-basis LOCA, but low-pressure injection fails: accumulators give one partial "
             "reflood, then the core boils dry and melts.",
             t_end="$t_end", events=[Event(0.0, "break", dict(area="$area", loc="liquid")),
                                     Event(0.0, "pump_trip")],
             eccs_off=["LPI", "HPI"],
             params=dict(area=(0.40, 0.01, 0.8, "break area (m²)"), t_end=(5400.0, 600, 14400, "duration (s)")),
             dt_max=2.0, tags=("severe",)),
    Scenario("pwr-sbloca-tmi", "Stuck-open relief valve, HPI throttled (1979-like)", "pwr",
             "Loss of main feed, the pressurizer relief valve sticks open and high-pressure injection is "
             "throttled. The primary drains through the valve, the pumps are tripped at 100 min and the core "
             "uncovers; a pump restart at 174 min and full HPI later reflood the damaged core.",
             t_end="$t_end",
             events=[Event(0.0, "loss_of_feed"), Event(0.0, "aux_feed", dict(on=False)),
                     Event(8.0, "depressurize", dict(area=0.00113)),
                     Event(0.0, "eccs", dict(name="HPI", on=False)),
                     Event(100 * 60.0, "pump_trip"),
                     Event(139 * 60.0, "depressurize", dict(area=0.0)),
                     Event(174 * 60.0, "inject", dict(flow=300.0, T=560.0, p_max=20.0, name="pump surge", volume=25.0)),
                     Event(200 * 60.0, "eccs", dict(name="HPI", on=True)),
                     Event(480.0, "aux_feed", dict(on=True))],
             params=dict(t_end=(5 * H, 3600, 8 * H, "duration (s)")), dt_max=5.0, tags=("severe",)),
    Scenario("pwr-sbo", "Station blackout, no auxiliary feedwater", "pwr",
             "Loss of all AC power and of the turbine-driven auxiliary feed pump. Steam generators boil dry "
             "in ~1.5 h, the primary heats up to the relief-valve setpoint, boils off through it and through "
             "the pump seals, and the core uncovers and melts. Accumulators inject once the primary depressurizes.",
             t_end="$t_end", events=[Event(0.0, "sbo"), Event(0.0, "aux_feed", dict(on=False))],
             params=dict(t_end=(6 * H, 2 * H, 12 * H, "duration (s)")), dt_max=10.0, tags=("severe",)),
    Scenario("pwr-sbo-afw", "Station blackout with turbine-driven feed until batteries deplete", "pwr",
             "Turbine-driven auxiliary feed holds the steam generators until the batteries run out, then the "
             "SBO sequence proceeds with a long delay.",
             t_end="$t_end", events=[Event(0.0, "sbo")],
             params=dict(t_end=(16 * H, 6 * H, 24 * H, "duration (s)")), dt_max=20.0, tags=("severe",)),
    Scenario("pwr-rod-ejection", "Control-rod ejection at hot zero power", "pwr",
             "A rod-housing failure ejects a central control rod in 0.1 s at hot zero power. The reactivity "
             "insertion exceeds one dollar; the prompt-critical burst is terminated by Doppler feedback "
             "within ~50 ms. Fuel enthalpy is compared with PCMI and coolability limits.",
             t_end=8.0, power0=1e-6, events=[Event(0.5, "rod_eject", dict(time=0.1))],
             eject=dict(col="center", worth="$worth"),
             params=dict(worth=(1.3, 0.5, 2.5, "rod worth ($)")), dt_max=0.05, tags=("reactivity",)),
    Scenario("pwr-xenon-osc", "Axial xenon oscillation", "pwr",
             "A part-length rod bank is inserted into the upper core for 2 h and withdrawn, with total "
             "power held constant. The I-135/Xe-135 imbalance drives a free axial oscillation with a period "
             "of about a day; Doppler and moderator feedback damp it. Weaken the Doppler coefficient to move "
             "the core toward the stability limit.",
             t_end="$t_end", controller=True,
             core=dict(alpha_D="$doppler", alpha_mod="$mtc"),
             banks=dict(PL=dict(cols="all", depth=0.0, worth=0.8, frm="top")),
             events=[Event(1800.0, "rod_bank", dict(name="PL", depth=0.5, speed=1e-3)),
                     Event(1800.0 + 2 * H, "rod_bank", dict(name="PL", depth=0.0, speed=1e-3))],
             params=dict(t_end=(96 * H, 24 * H, 144 * H, "duration (s)"),
                         doppler=(-2.6e-5, -5e-5, -0.5e-5, "Doppler coefficient (1/K)"),
                         mtc=(-2.5e-4, -7e-4, -0.5e-4, "moderator temperature coefficient (1/K)")),
             dt_max=600.0, tags=("xenon",)),
    Scenario("pwr-xenon-restart", "Trip and restart through the xenon pit", "pwr",
             "Trip from equilibrium full power, then an attempted restart to full power with all rods "
             "withdrawn. Xenon peaks ~8-10 h after the trip; if its worth exceeds the available excess "
             "reactivity the restart fails until it decays.",
             t_end="$t_end", controller=True,
             events=[Event(600.0, "setpoint", dict(power=0.0)),
                     Event("$t_restart", "setpoint", dict(power=1.0))],
             params=dict(t_restart=(6 * H, 600 + 600, 40 * H, "restart time (s)"),
                         t_end=(60 * H, 12 * H, 96 * H, "duration (s)")), dt_max=300.0, tags=("xenon",)),
    Scenario("pwr-lof", "Loss of flow (pump trip) with scram", "pwr",
             "All reactor coolant pumps trip; low-flow trip scrams the reactor; natural circulation "
             "removes decay heat. Departure from nucleate boiling is the margin of interest.",
             t_end=600.0, events=[Event(1.0, "pump_trip")], dt_max=1.0, tags=("design basis",)),
]

# ------------------------------------------------------------------------------- BWR
S += [
    Scenario("bwr-steady", "Steady full power", "bwr", "Rated power and flow, equilibrium xenon.",
             t_end=600.0, controller=True, dt_max=60.0),
    Scenario("bwr-sbo-rcic", "Extended blackout: RCIC until DC lost (2011-like)", "bwr",
             "Tsunami-induced station blackout: the steam-driven RCIC keeps the core covered until DC power "
             "and its control are lost, then the vessel boils down through the relief valves at 7.4 MPa. "
             "Core uncovery, oxidation and hydrogen, melting and slumping to the lower head follow; late "
             "depressurization and fire-truck injection arrive after the damage.",
             t_end="$t_end",
             events=[Event(0.0, "sbo"), Event(0.0, "eccs", dict(name="HPCI", on=False)),
                     Event("$t_rcic", "eccs", dict(name="RCIC", on=False)),
                     Event("$t_ads", "depressurize", dict(area=0.08)),
                     Event("$t_fire", "inject", dict(flow=8.0, T=300.0, p_max=0.8, name="fire truck"))],
             params=dict(t_rcic=(8 * H, 1 * H, 60 * H, "RCIC lost at (s)"),
                         t_ads=(13 * H, 2 * H, 70 * H, "depressurization at (s)"),
                         t_fire=(14 * H, 2 * H, 72 * H, "fire-truck injection at (s)"),
                         t_end=(20 * H, 4 * H, 80 * H, "duration (s)")), dt_max=20.0, tags=("severe",)),
    Scenario("bwr-sbo-ic", "Blackout with isolation condenser lost immediately", "bwr",
             "No isolation-condenser or RCIC cooling at all after the blackout: the fastest BWR sequence, "
             "with core damage in a few hours.",
             t_end="$t_end", events=[Event(0.0, "sbo"), Event(0.0, "eccs", dict(name="HPCI", on=False)),
                                     Event(0.0, "eccs", dict(name="RCIC", on=False))],
             params=dict(t_end=(10 * H, 3 * H, 24 * H, "duration (s)")), dt_max=10.0, tags=("severe",)),
    Scenario("bwr-turbine-trip", "Turbine trip without bypass", "bwr",
             "Turbine stop valves close with no bypass: pressure rises, voids collapse and the negative void "
             "coefficient turns into a positive reactivity insertion. The power spike is limited by "
             "Doppler feedback and the scram on high flux.",
             t_end=20.0, events=[Event(0.5, "turbine_trip", dict(bypass=0.0))], dt_max=0.02,
             tags=("reactivity",)),
    Scenario("bwr-recirc-trip", "Recirculation pump trip", "bwr",
             "Both recirculation pumps trip without scram: flow falls to natural circulation, void rises "
             "and power settles near 40 % on the natural-circulation line of the power-flow map.",
             t_end=300.0, rps=False, events=[Event(1.0, "pump_trip")], dt_max=0.5, tags=("operational",)),
    Scenario("bwr-lbloca", "Recirculation-line break with ADS and LPCI", "bwr",
             "Large suction-line break. Blowdown, core uncovery, automatic depressurization and core spray / "
             "LPCI reflood.",
             t_end=1800.0, events=[Event(0.0, "break", dict(area=0.3, loc="liquid")), Event(0.0, "pump_trip")],
             dt_max=2.0, tags=("design basis",)),
]

# ------------------------------------------------------------------------------- RBMK
S += [
    Scenario("rbmk-steady", "Steady full power", "rbmk", "Rated power, equilibrium xenon.", t_end=600.0,
             controller=True, dt_max=60.0),
    Scenario("rbmk-1986", "Low-power test with void excursion and positive scram (1986-like)", "rbmk",
             "After a day at half power and a dip to near zero, the core runs at 200 MWt, xenon-poisoned, "
             "with almost all rods withdrawn and near-saturated inlet water. Four of eight pumps coast down "
             "with the turbine; voids rise, power rises; the scram button drives rods in at 0.4 m/s and their "
             "graphite displacers add reactivity in the bottom of the core. The prompt-critical excursion "
             "fragments fuel and ruptures channels within seconds.",
             t_end=60.0, power0=0.0625, flow0=1.12, feed0=0.3,
             history=[(11 * H, 0.5), (0.5 * H, 0.01), (1.0 * H, 0.0625)],
             rps=False, controller=False,
             events=[Event(0.0, "flow_ramp", dict(frac=0.75, tau=25.0)),
                     Event("$t_az5", "scram")],
             params=dict(t_az5=(36.0, 5.0, 60.0, "scram button at (s)")), dt_max=0.05, tags=("reactivity", "severe")),
    Scenario("rbmk-xenon", "Xenon pit after power reduction", "rbmk",
             "Power reduced from 100 % to 50 % and then to 6 %: xenon builds up and the controller "
             "withdraws rods to hold power; the operational reactivity margin collapses.",
             t_end=24 * H, controller=True,
             events=[Event(1 * H, "setpoint", dict(power=0.5)), Event(12 * H, "setpoint", dict(power=0.0625))],
             dt_max=300.0, tags=("xenon",)),
    Scenario("rbmk-xenon-osc", "Radial-axial xenon instability", "rbmk",
             "A large, loosely coupled core: a short local rod insertion triggers spatial xenon oscillations "
             "that the uniform controller cannot damp.",
             t_end=72 * H, controller=True, banks=dict(LAR=dict(cols="left", depth=0.0, worth=0.5, frm="top")),
             events=[Event(1800.0, "rod_bank", dict(name="LAR", depth=0.8, speed=1e-3)),
                     Event(1800.0 + 3 * H, "rod_bank", dict(name="LAR", depth=0.0, speed=1e-3))],
             dt_max=600.0, tags=("xenon",)),
]
S.append(Scenario("rbmk-mod-1986", "Same test after the post-1986 modifications", "rbmk-mod",
                  "The 1986 test sequence on the modified design: smaller void coefficient, no positive scram, "
                  "fast scram. Power rises modestly and the scram terminates the transient.",
                  t_end=60.0, power0=0.0625, flow0=1.12, feed0=0.3,
                  history=[(11 * H, 0.5), (0.5 * H, 0.01), (1.0 * H, 0.0625)], rps=False,
                  events=[Event(0.0, "flow_ramp", dict(frac=0.75, tau=25.0)), Event("$t_az5", "scram")],
                  params=dict(t_az5=(36.0, 5.0, 60.0, "scram button at (s)")), dt_max=0.05,
                  tags=("reactivity",)))

# ------------------------------------------------------------------------------- CANDU
S += [
    Scenario("candu-steady", "Steady full power", "candu", "Rated power, equilibrium xenon.", t_end=600.0,
             controller=True, dt_max=60.0),
    Scenario("candu-lbloca", "Large LOCA: void power pulse and shutdown system 1", "candu",
             "Inlet-header break: coolant voiding adds positive reactivity and a power pulse; shutdown "
             "system 1 terminates it within about 1.5 s; emergency coolant injection refills the channels.",
             t_end=600.0, events=[Event(0.0, "break", dict(area=0.25, loc="liquid")), Event(0.0, "pump_trip")],
             dt_max=0.5, tags=("design basis", "reactivity")),
    Scenario("candu-loca-loecc", "LOCA with loss of emergency coolant injection", "candu",
             "No ECC: the channels stay dry and decay heat leaves through the pressure tubes into the "
             "cool moderator, which keeps the fuel from melting while moderator cooling works.",
             t_end=3 * H, events=[Event(0.0, "break", dict(area=0.25, loc="liquid")), Event(0.0, "pump_trip")],
             eccs_off=["HP ECC", "MP/LP ECC"], dt_max=5.0, tags=("severe",)),
    Scenario("candu-sbo", "Station blackout: moderator boil-off", "candu",
             "Loss of all AC including moderator cooling: the heat-transport system boils off, then the "
             "moderator heats to saturation and boils away over many hours, uncovering channels.",
             t_end="$t_end", events=[Event(0.0, "sbo")],
             params=dict(t_end=(30 * H, 6 * H, 60 * H, "duration (s)")), dt_max=30.0, tags=("severe",)),
    Scenario("candu-poison-out", "Trip and xenon poison-out", "candu",
             "Trip from full power and a restart attempt 45 min later. With natural uranium and a high "
             "flux, Xe-135 outruns the adjuster rods within about 35-40 min and the attempt fails: the "
             "reactor is poisoned out until the xenon decays. A restart within ~10-15 min of the trip beats it.",
             t_end="$t_end", controller=True,
             events=[Event(600.0, "setpoint", dict(power=0.0)),
                     Event("$t_restart", "setpoint", dict(power=1.0))],
             params=dict(t_restart=(600 + 45 * 60.0, 900.0, 48 * H, "restart time (s)"),
                         t_end=(12 * H, 3 * H, 96 * H, "duration (s)")), dt_max=300.0, tags=("xenon",)),
]

# ------------------------------------------------------------------------------- HTGR
S += [
    Scenario("htgr-steady", "Steady full power", "htgr", "Rated power and helium flow.", t_end=600.0,
             controller=True, dt_max=60.0),
    Scenario("htgr-dlofc", "Depressurized loss of forced cooling", "htgr",
             "Helium pressure boundary break and circulator trip with scram. No convection: decay heat "
             "conducts radially through the graphite to the vessel and the cavity cooling system. The fuel "
             "heats up slowly for several days, peaking a few hundred kelvin above its operating "
             "temperature and well below the ~1900 K where TRISO particles begin to fail.",
             t_end="$t_end", events=[Event(0.0, "break", dict(area=0.01, loc="steam")), Event(0.0, "pump_trip")],
             params=dict(t_end=(150 * H, 24 * H, 300 * H, "duration (s)")), dt_max=600.0, tags=("passive",)),
    Scenario("htgr-plofc-atws", "Pressurized loss of forced cooling without scram", "htgr",
             "Circulators trip and the rods fail to insert. The strong negative temperature coefficient "
             "shuts the reactor down by itself; it re-criticalises at low power after xenon decays.",
             t_end="$t_end", rps=False, events=[Event(0.0, "pump_trip")],
             params=dict(t_end=(72 * H, 12 * H, 150 * H, "duration (s)")), dt_max=300.0, tags=("passive", "unprotected")),
]

# ------------------------------------------------------------------------------- SFR
S += [
    Scenario("sfr-metal-steady", "Steady full power", "sfr-metal", "Rated power and flow.", t_end=600.0,
             controller=True, dt_max=60.0),
    Scenario("sfr-metal-ulof", "Unprotected loss of flow (1986 EBR-II-type test)", "sfr-metal",
             "Primary pumps trip without scram. Coolant heating and thermal expansion of fuel and core "
             "structures insert negative reactivity faster than the sodium void, and power drops to "
             "decay-heat levels with no sodium boiling.",
             t_end=900.0, rps=False, events=[Event(1.0, "pump_trip")], dt_max=1.0, tags=("passive", "unprotected")),
    Scenario("sfr-metal-utop", "Unprotected transient overpower", "sfr-metal",
             "Inadvertent withdrawal of a control rod worth 30 cents without scram; the core settles at a "
             "higher power where feedback balances the insertion.",
             t_end=600.0, rps=False, banks=dict(CR=dict(cols="center", depth=1.0, worth="$worth", frm="top")),
             events=[Event(1.0, "rod_bank", dict(name="CR", depth=0.0, speed=0.067))],
             params=dict(worth=(0.30, 0.05, 0.9, "rod worth ($)")), dt_max=0.5, tags=("unprotected",)),
    Scenario("sfr-oxide-ulof", "Unprotected loss of flow, oxide core", "sfr-oxide",
             "The same pump trip in an oxide core: weak expansion feedback and a large sodium void worth "
             "drive the outlet to sodium boiling and fuel damage.",
             t_end=600.0, rps=False, events=[Event(1.0, "pump_trip")], dt_max=0.5, tags=("unprotected", "severe")),
]

SCENARIOS = {s.key: s for s in S}


def by_reactor():
    out = {}
    for s in S:
        out.setdefault(s.reactor, []).append(s)
    return out
