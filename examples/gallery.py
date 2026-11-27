"""Regenerate the README figures into docs/figures (takes a few minutes).

    python examples/gallery.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reactorsim import plots  # noqa: E402
from reactorsim.core.ensemble import run_ensemble  # noqa: E402
from reactorsim.core.simulator import Simulator  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures")


def main():
    os.makedirs(OUT, exist_ok=True)
    H = 3600.0
    for key in ("pwr-sbo", "rbmk-1986", "sfr-metal-ulof", "pwr-sbloca-tmi", "htgr-dlofc"):
        r = Simulator(key).run()
        plots.scenario_figure(r, os.path.join(OUT, f"{key}.png"))
        print("wrote", key)
        if key == "pwr-sbo":
            plots.lattice_figure(r, os.path.join(OUT, "pwr-sbo-states.png"), [2.8 * H, 3.0 * H, 3.2 * H, 3.4 * H, 3.7 * H],
                                 field="state")
            plots.lattice_figure(r, os.path.join(OUT, "pwr-sbo-fuel.png"), [2.8 * H, 3.0 * H, 3.2 * H, 3.4 * H, 3.7 * H])
    r = Simulator("pwr-xenon-osc").run()
    te = r.t[-1]
    plots.lattice_figure(r, os.path.join(OUT, "pwr-xenon-osc.png"), [0.0, te * 0.1, te * 0.2, te * 0.3, te * 0.4],
                         field="q3", cmap="inferno", label="power density (W/cm³)")
    plots.xenon_figure(os.path.join(OUT, "xenon.png"))
    plots.kinetics_figure(os.path.join(OUT, "kinetics.png"))
    ens = run_ensemble("pwr-sbo", n=int(os.environ.get("GALLERY_N", "12")))
    plots.ensemble_figure(ens, os.path.join(OUT, "ensemble-pwr-sbo.png"))
    for row in ens.table():
        print(row)


if __name__ == "__main__":
    main()
