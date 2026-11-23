"""Synthetic test data and self-check for the twisted-cell anchoring fit.

    ~/.venv/base/bin/python twist_demo.py --check        # closed form vs stepwise Jones
    ~/.venv/base/bin/python twist_demo.py OUTDIR         # write demo spectra + manifest, then fit

The demo cell: E7 (Li 2005), d = 20 µm, W = 9e-6 J/m², K22 = 6.2 pN, rubbing
at 10° in the lab, pattern rotated by δ = +1°, regions θ = 0°, +80°, −80°.
Each region: crossed + parallel pairs at 4 rotations of the polarizer pair,
multiplied by a fake lamp spectrum, 0.5 % noise.
"""
import os
import sys

import numpy as np

import twist_model as tm
from dispersion import MATERIALS

TRUE = dict(d=20.0, W=9e-6, K22=6.2e-12, psi_rub=10.0, delta=1.0)
THETAS = (0.0, 80.0, -80.0)


def check():
    dn = MATERIALS["E7"]["Li et al. 2005 (25 °C)"].delta_n
    lam = np.array([0.47, 0.55, 0.63])
    for (pin, pout, P, A) in [(87, 0, 30, 100), (-40, 20, 0, 90), (5, 5, 45, 45)]:
        a = tm.transmission(lam, dn(lam), 20.0, pin * tm.DEG, pout * tm.DEG,
                            P * tm.DEG, A * tm.DEG)
        b = tm.transmission_stepwise(lam, dn(lam), 20.0, pin * tm.DEG,
                                     pout * tm.DEG, P * tm.DEG, A * tm.DEG)
        print(f"ψ_BY={pin:4}°, ψ_rub={pout:3}°: max |closed − stepwise| = {np.max(abs(a - b)):.1e}")


def make(outdir, seed=1):
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    dn = MATERIALS["E7"]["Li et al. 2005 (25 °C)"].delta_n
    lam_nm = np.linspace(430, 720, 900)
    lam = lam_nm / 1000
    lamp = 1000 * np.exp(-((lam_nm - 580) / 140) ** 2) + 50
    rows = []
    for th in THETAS:
        psi_by, phi, D = tm._region_angles(th, TRUE["delta"], TRUE["W"],
                                           TRUE["K22"], TRUE["d"], TRUE["psi_rub"])
        print(f"θ = {th:+.0f}°: true Φ = {np.degrees(phi):+.3f}°, lag Δ = {np.degrees(D):.3f}°")
        for rot in (0.0, 22.5, 45.0, 67.5):
            for off in (90.0, 0.0):
                P, A = rot, (rot + off) % 180
                T = tm.transmission(lam, dn(lam), TRUE["d"], psi_by,
                                    TRUE["psi_rub"] * tm.DEG, P * tm.DEG, A * tm.DEG)
                I = lamp * T + rng.normal(0, 0.005 * lamp.max(), lam.size)
                name = f"theta{th:+.0f}_P{P:g}_A{A:g}.csv".replace(".", "p", 0)
                np.savetxt(os.path.join(outdir, name), np.c_[lam_nm, I],
                           delimiter=",", header="lambda_nm,counts", comments="")
                rows.append(tm.Row(os.path.join(outdir, name), th, P, A))
    tm.write_manifest(os.path.join(outdir, "manifest.csv"), rows)
    return rows


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        check()
        sys.exit()
    out = sys.argv[1] if len(sys.argv) > 1 else "twist_demo_data"
    rows = make(out)
    spectra, warn = tm.build_spectra(rows, "pair")
    dsets = MATERIALS["E7"]
    s = tm.FitSettings(dn_func=dsets["Li et al. 2005 (25 °C)"].delta_n,
                       d_range=(15.0, 25.0))
    pp, rfs, jf, pairs, syst = tm.run_all(spectra, s, progress=print)
    print(tm.format_report(rfs, jf, pairs, s, syst, warn))
    print(f"\nTRUE: d = {TRUE['d']}, ψ_rub = {TRUE['psi_rub']}, δ = {TRUE['delta']}, W = {TRUE['W']:.3g}")
