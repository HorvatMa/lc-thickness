"""Azimuthal anchoring W of a patterned (BY) surface from twisted-cell spectra.

Cell: light enters through the patterned (weak, BY) plate and leaves through
the rubbed (strong) plate. Planar, zero pretilt, director angle linear in z:

    ψ(z) = ψ_BY + Φ z/d,     Φ = ψ_rub − ψ_BY   (bulk twist, |Φ| < 90°)

Jones matrix of the twisted slab in the local director frame
(X = sqrt(Φ² + Γ²/4), Γ = 2π Δn(λ) d / λ):

    M = [[cos X − i(Γ/2) sin X/X,   Φ sin X/X],
         [−Φ sin X/X,               cos X + i(Γ/2) sin X/X]]

checked against a 3000-layer stepwise product (agreement ~1e-7, see
twist_demo.py --check). T(λ; P, A) = |â·R(ψ_rub) M R(−ψ_BY) p̂|², with the
polarizer P and analyzer A given as lab angles of their transmission axes.

Anchoring (Rapini–Papoular, F_s = ½ W sin²Δ on the BY plate only, rubbed
plate taken as infinitely strong). Region k is patterned with its BY easy
axis at θ_k from the rubbing; the whole pattern may be rotated by an unknown
assembly offset δ. With S_k = θ_k + δ the set twist, the BY director lags its
easy axis by Δ_k, where

    K22 (|S_k| − Δ_k)/d = (W/2) sin 2Δ_k ,      |Φ_k| = |S_k| − Δ_k .

The joint fit determines d, ψ_rub, δ and W from all spectra of all regions.
Only b = K22/W (extrapolation length) is set by the optics; W scales with the
K22 that is assumed.
"""
from dataclasses import dataclass, field
import csv
import json
import os
import re

import numpy as np
from scipy.optimize import least_squares, brentq

import model as base_model  # load_table, spectrum_from_table (unchanged)

DEG = np.pi / 180.0


# ------------------------------------------------------------------ optics
def transmission(lam_um, dn, d_um, psi_by, psi_rub, P, A):
    """T(λ) of the twisted slab between polarizer P and analyzer A (radians).

    lam_um, dn: arrays of the same shape; d_um may be an array broadcastable
    against them (used for the d scan).
    """
    phi = psi_rub - psi_by
    G = 2.0 * np.pi * dn * d_um / lam_um
    X = np.sqrt(phi ** 2 + (G / 2.0) ** 2)
    sx = np.sinc(X / np.pi)            # sin X / X, safe at X = 0
    m11 = np.cos(X) - 1j * (G / 2.0) * sx
    m12 = phi * sx
    pl, pt = np.cos(P - psi_by), np.sin(P - psi_by)
    al, at = np.cos(A - psi_rub), np.sin(A - psi_rub)
    ol = m11 * pl + m12 * pt
    ot = -m12 * pl + np.conj(m11) * pt
    return np.abs(al * ol + at * ot) ** 2


def transmission_stepwise(lam_um, dn, d_um, psi_by, psi_rub, P, A, n=3000):
    """Brute-force product of n thin uniform layers (for checking only)."""
    out = []
    for lam, dni in zip(np.atleast_1d(lam_um), np.atleast_1d(dn)):
        g = np.pi * dni * (d_um / n) / lam
        Mtot = np.eye(2, dtype=complex)
        for i in range(n):
            a = psi_by + (psi_rub - psi_by) * (i + 0.5) / n
            c, s = np.cos(a), np.sin(a)
            R = np.array([[c, -s], [s, c]])
            Mtot = R @ np.diag([np.exp(-1j * g), np.exp(1j * g)]) @ R.T @ Mtot
        o = Mtot @ np.array([np.cos(P), np.sin(P)])
        out.append(abs(np.cos(A) * o[0] + np.sin(A) * o[1]) ** 2)
    return np.array(out)


# ------------------------------------------------------------------ anchoring
def wrap_twist(a):
    """Wrap an axis angle difference (rad) into (−π/2, π/2]."""
    return (a + np.pi / 2) % np.pi - np.pi / 2


def lag(S, W, K22, d_um):
    """Lag Δ ≥ 0 (rad) of the BY director behind its easy axis.

    Smallest root of K22(|S| − Δ)/d = (W/2) sin 2Δ (the branch that is
    continuous with strong anchoring). Returns |S| if W → 0.
    """
    s = abs(S)
    if s < 1e-12:
        return 0.0
    k = K22 / (d_um * 1e-6)
    f = lambda D: k * (s - D) - 0.5 * W * np.sin(2 * D)
    grid = np.linspace(0.0, s, 200)
    vals = f(grid)
    idx = np.nonzero(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]
    if len(idx) == 0:
        return s if vals[-1] >= 0 else 0.0
    i = idx[0]
    return brentq(f, grid[i], grid[i + 1], xtol=1e-14)


def W_from_twist(phi_abs, delta, K22, d_um):
    """W = 2 K22 |Φ| / (d sin 2Δ) (J/m²; angles in rad)."""
    return 2.0 * K22 * phi_abs / (d_um * 1e-6 * np.sin(2 * delta))


# ------------------------------------------------------------------ data
@dataclass
class Spectrum:
    """One normalized spectrum: region pattern angle θ, polarizer, analyzer."""
    theta: float          # BY easy axis relative to rubbing, deg (as patterned)
    P: float              # deg, lab
    A: float              # deg, lab
    lam_nm: np.ndarray
    T: np.ndarray
    label: str = ""


@dataclass
class Row:
    """One measured file as listed in the GUI table / manifest."""
    path: str
    theta: float
    P: float
    A: float


ANGLE_RE = {"P": re.compile(r"(?:^|[^A-Za-z])P[_=-]?(-?\d+(?:[.p]\d+)?)", re.I),
            "A": re.compile(r"(?:^|[^A-Za-z])A[_=-]?(-?\d+(?:[.p]\d+)?)", re.I)}


def angles_from_name(path):
    """Guess (P, A) from a file name like 'cell_P30_A120.csv' (or None)."""
    name = os.path.splitext(os.path.basename(path))[0]
    out = []
    for key in ("P", "A"):
        m = ANGLE_RE[key].search(name)
        out.append(float(m.group(1).replace("p", ".")) if m else None)
    return tuple(out)


def read_manifest(path):
    """CSV with header file,theta,P,A (paths relative to the manifest)."""
    rows = []
    base = os.path.dirname(os.path.abspath(path))
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            p = r["file"].strip()
            if not os.path.isabs(p):
                p = os.path.join(base, p)
            rows.append(Row(p, float(r["theta"]), float(r["P"]), float(r["A"])))
    return rows


def write_manifest(path, rows):
    base = os.path.dirname(os.path.abspath(path))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "theta", "P", "A"])
        for r in rows:
            w.writerow([os.path.relpath(r.path, base), r.theta, r.P, r.A])


def _load(path, lam_col, int_col):
    return base_model.spectrum_from_table(base_model.load_table(path),
                                          lam_col, int_col)


def build_spectra(rows, mode, lam_col=0, int_col=1, ref_path=None,
                  dark_path=None, min_ref_fraction=0.05):
    """Normalize the raw files.

    mode: "pair"  T = S/(S + S'), S' = same region and P, analyzer ±90°
                  (T(P,A) + T(P,A+90°) = 1 for any lossless cell);
          "lamp"  T = (S − D)/(R − D), R = parallel polarizers, no cell;
          "T"     files already contain T.
    Returns (list of Spectrum, list of warnings).
    """
    warn = []
    dark = _load(dark_path, lam_col, int_col) if dark_path else None
    ref = _load(ref_path, lam_col, int_col) if (mode == "lamp" and ref_path) else None
    if mode == "lamp" and ref is None:
        raise ValueError("Lamp mode needs a reference spectrum.")
    raw = [_load(r.path, lam_col, int_col) for r in rows]
    out = []
    for i, (r, (lam, I)) in enumerate(zip(rows, raw)):
        S = I.astype(float)
        D = np.interp(lam, *dark) if dark is not None else 0.0
        if mode == "T":
            T = S
        elif mode == "lamp":
            R = np.interp(lam, *ref, left=np.nan, right=np.nan) - D
            S = S - D
            T = np.full_like(S, np.nan)
            ok = np.isfinite(R) & (R > min_ref_fraction * np.nanmax(R))
            T[ok] = S[ok] / R[ok]
        else:
            partner = [j for j, q in enumerate(rows) if j != i
                       and abs(q.theta - r.theta) < 1e-6
                       and abs(((q.P - r.P) + 90) % 180 - 90) < 0.5
                       and abs(((q.A - r.A) % 180) - 90) < 0.5]
            if not partner:
                warn.append(f"{os.path.basename(r.path)}: no partner with "
                            f"A ± 90° at the same P and θ; skipped.")
                continue
            lam2, I2 = raw[partner[0]]
            S2 = np.interp(lam, lam2, I2, left=np.nan, right=np.nan) - D
            S = S - D
            tot = S + S2
            T = np.full_like(S, np.nan)
            ok = np.isfinite(tot) & (tot > min_ref_fraction * np.nanmax(tot))
            T[ok] = S[ok] / tot[ok]
        out.append(Spectrum(r.theta, r.P, r.A, lam, T,
                            os.path.basename(r.path)))
    return out, warn


def load_measurement(folder, min_ref_fraction=0.05):
    """Spectra from a twist_acquire measurement folder (measurement.json).

    Each spectrum: S = (I − D(t))/t, with I offset-corrected counts, D(t) the
    mean of the start and end dark at the same exposure t (if only one exposure
    is missing, the nearest dark is scaled ∝ t). Pairs crossed/parallel at the
    same θ and ρ give T = S⊥/(S⊥ + S∥), returned at the crossed P, A.
    Returns (list of Spectrum, warnings, metadata dict).
    """
    with open(os.path.join(folder, "measurement.json")) as f:
        meta = json.load(f)
    warn = []

    def read(fn):
        lam, I = _load(os.path.join(folder, fn), 0, 1)
        return lam, I

    darks = {}
    for d in meta.get("darks", []):
        darks.setdefault(float(d["exposure_ms"]), []).append(read(d["file"]))
    whens = {d["when"] for d in meta.get("darks", [])}
    if not darks:
        warn.append("No dark spectra: only the shielded-pixel offset is subtracted.")
    elif whens != {"start", "end"}:
        warn.append(f"Dark spectra only at {', '.join(sorted(whens))}.")

    def dark(lam, t):
        if not darks:
            return 0.0
        if t in darks:
            return np.mean([np.interp(lam, *x) for x in darks[t]], axis=0)
        tn = min(darks, key=lambda x: abs(np.log(x / t)))
        warn.append(f"No dark at {t:g} ms; used {tn:g} ms scaled.")
        return np.mean([np.interp(lam, *x) for x in darks[tn]], axis=0) * t / tn

    S = {}
    for e in meta["spectra"]:
        lam, I = read(e["file"])
        t = float(e["exposure_ms"])
        S[(e["theta"], e["rho"], e["pol"])] = (lam, (I - dark(lam, t)) / t, e)
        if e.get("max_raw", 0) >= meta.get("saturation_raw", np.inf):
            warn.append(f"{e['file']} is saturated.")
    out = []
    for (th, rho, pol), (lam, Sx, e) in sorted(S.items()):
        if pol != "crossed":
            continue
        if (th, rho, "parallel") not in S:
            warn.append(f"{e['file']}: no parallel partner; skipped.")
            continue
        lam2, Sp, _ = S[(th, rho, "parallel")]
        Sp = np.interp(lam, lam2, Sp, left=np.nan, right=np.nan)
        tot = Sx + Sp
        T = np.full_like(Sx, np.nan)
        ok = np.isfinite(tot) & (tot > min_ref_fraction * np.nanmax(tot))
        T[ok] = Sx[ok] / tot[ok]
        out.append(Spectrum(float(th), float(e["P"]), float(e["A"]), lam, T,
                            e["file"]))
    return out, warn, meta


# ------------------------------------------------------------------ fitting
@dataclass
class FitSettings:
    dn_func: callable                 # Δn(λ_µm)
    K22: float = 6.2e-12              # N
    lam_range: tuple = (450.0, 700.0) # nm
    d_range: tuple = (5.0, 40.0)      # µm, scan range
    free_ab: bool = True              # per-spectrum amplitude a and offset b
    share_d: bool = True
    fit_delta: bool = True            # False: δ fixed at delta_fixed
    delta_fixed: float = 0.0          # deg
    psi_rub_guess: float = None       # deg; None = scan 0–180°
    max_points: int = 300             # λ points per spectrum used in the fit


@dataclass
class Prepared:
    lam: list
    dn: list
    T: list
    spectra: list


def prepare(spectra, s: FitSettings):
    lams, dns, Ts, keep = [], [], [], []
    for sp in spectra:
        m = (np.isfinite(sp.T) & (sp.lam_nm >= s.lam_range[0])
             & (sp.lam_nm <= s.lam_range[1]))
        lam = sp.lam_nm[m] / 1000.0
        T = sp.T[m]
        if lam.size < 10:
            continue
        if lam.size > s.max_points:
            idx = np.linspace(0, lam.size - 1, s.max_points).round().astype(int)
            lam, T = lam[idx], T[idx]
        lams.append(lam)
        dns.append(s.dn_func(lam))
        Ts.append(T)
        keep.append(sp)
    if not keep:
        raise ValueError("No data in the λ range.")
    return Prepared(lams, dns, Ts, keep)


def _ab(m, y, free):
    """Best a, b for y ≈ a m + b (rows = last axis)."""
    if not free:
        return 1.0, 0.0
    n = m.shape[-1]
    sm, sy = m.sum(-1), y.sum(-1)
    smm, smy = (m * m).sum(-1), (m * y).sum(-1)
    det = n * smm - sm ** 2
    det = np.where(np.abs(det) < 1e-12, 1e-12, det)
    a = (n * smy - sm * sy) / det
    b = (sy - a * sm) / n
    return a, b


def _residuals_angles(pp, idx, d, psi_by, psi_rub, free_ab):
    """Residual vector and per-spectrum (a, b) for given director angles.

    idx: indices into pp of the spectra; psi_by may differ per spectrum.
    """
    res, ab = [], []
    for k, i in enumerate(idx):
        sp = pp.spectra[i]
        m = transmission(pp.lam[i], pp.dn[i], d[k], psi_by[k], psi_rub,
                         sp.P * DEG, sp.A * DEG)
        a, b = _ab(m, pp.T[i], free_ab)
        res.append(a * m + b - pp.T[i])
        ab.append((float(a), float(b)))
    return np.concatenate(res), ab


def _d_step(pp, lo):
    lam_min = min(l.min() for l in pp.lam)
    dn_max = max(np.max(np.abs(x)) for x in pp.dn)
    return 0.02 * lam_min / (np.pi * dn_max)


def _scan_region(pp, idx, theta, s: FitSettings):
    """Grid over (ψ_rub, Φ, d) for one region; returns sorted candidates."""
    d_grid = np.arange(s.d_range[0], s.d_range[1], _d_step(pp, s.d_range[0]))
    if s.psi_rub_guess is None:
        psi_grid = np.arange(0.0, 90.0, 7.5)   # ψ_rub ≡ ψ_rub + 90°
    else:
        psi_grid = s.psi_rub_guess + np.arange(-10.0, 10.1, 5.0)
    t = abs(theta)
    sgn = -1.0 if theta >= 0 else 1.0       # Φ ≈ −θ
    phi_grid = sorted({sgn * t * f for f in (1.0, 0.9, 0.75, 0.5)}
                      | ({-4.0, 0.0, 4.0} if t < 10 else set()))
    cands = []
    for psi in psi_grid:
        for phi in phi_grid:
            if abs(phi) >= 89.9:
                continue
            tot = np.zeros(d_grid.size)
            for i in idx:
                sp = pp.spectra[i]
                m = transmission(pp.lam[i][None, :], pp.dn[i][None, :],
                                 d_grid[:, None], (psi - phi) * DEG,
                                 psi * DEG, sp.P * DEG, sp.A * DEG)
                a, b = _ab(m, pp.T[i][None, :], s.free_ab)
                a, b = np.atleast_1d(a), np.atleast_1d(b)
                r = a[:, None] * m + b[:, None] - pp.T[i][None, :]
                tot += (r ** 2).sum(-1)
            j = int(np.argmin(tot))
            cands.append((tot[j], d_grid[j], psi, phi))
    cands.sort(key=lambda c: c[0])
    return cands


@dataclass
class RegionFit:
    theta: float
    n_spectra: int
    d: float
    d_err: float
    psi_rub: float
    psi_rub_err: float
    phi: float            # Φ = ψ_rub − ψ_BY, deg
    phi_err: float
    rms: float


def _cov(r):
    J = r.jac
    dof = max(r.fun.size - J.shape[1], 1)
    s2 = np.sum(r.fun ** 2) / dof
    try:
        return np.linalg.pinv(J.T @ J) * s2
    except np.linalg.LinAlgError:
        return np.full((J.shape[1],) * 2, np.nan)


def fit_region(pp, idx, theta, s: FitSettings, n_starts=6):
    """Free fit of one region: d, ψ_rub, Φ (no anchoring model)."""
    cands = _scan_region(pp, idx, theta, s)

    def res(p):
        d, psi, phi = p
        return _residuals_angles(pp, idx, [d] * len(idx),
                                 [(psi - phi) * DEG] * len(idx),
                                 psi * DEG, s.free_ab)[0]

    best = None
    for c in cands[:n_starts]:
        x0 = [c[1], c[2], c[3]]
        r = least_squares(res, x0, bounds=([s.d_range[0], -360, -89.9],
                                           [s.d_range[1], 360, 89.9]),
                          x_scale=[0.1, 1.0, 1.0])
        if best is None or r.cost < best.cost:
            best = r
    C = _cov(best)
    e = np.sqrt(np.abs(np.diag(C)))
    d, psi, phi = best.x
    return RegionFit(theta, len(idx), d, e[0], psi % 90.0, e[1], phi, e[2],
                     float(np.sqrt(np.mean(best.fun ** 2))))


@dataclass
class JointFit:
    thetas: list
    d: list               # per region (equal if shared)
    d_err: list
    psi_rub: float
    psi_rub_err: float
    delta: float          # δ, deg
    delta_err: float
    W: float              # J/m²
    logW_err: float
    b_um: float           # K22/W
    K22: float
    phi: list             # predicted Φ_k, deg
    lag: list             # Δ_k, deg
    rms: float
    ab: list
    at_bound: bool
    fit_delta: bool


def _region_angles(theta, delta, W, K22, d, psi_rub):
    S = wrap_twist((theta + delta) * DEG)
    D = lag(S, W, K22, d)
    phi_abs = abs(S) - D
    psi_by = psi_rub * DEG + np.sign(S) * phi_abs      # director at BY plate
    return psi_by, -np.sign(S) * phi_abs, D            # Φ = ψ_rub − ψ_BY


def fit_joint(pp, s: FitSettings, region_fits):
    """Fit d, ψ_rub, δ, log10 W to all spectra with the anchoring model."""
    thetas = sorted({sp.theta for sp in pp.spectra})
    idx_of = {t: [i for i, sp in enumerate(pp.spectra) if sp.theta == t]
              for t in thetas}
    nreg = len(thetas)
    fit_delta = s.fit_delta and nreg >= 2
    nd = 1 if s.share_d else nreg

    def unpack(p):
        p = list(p)
        ds = p[:nd]; p = p[nd:]
        psi = p.pop(0)
        delta = p.pop(0) if fit_delta else s.delta_fixed
        logW = p.pop(0)
        return ds, psi, delta, logW

    def res(p, return_ab=False):
        ds, psi, delta, logW = unpack(p)
        out, abs_ = [], []
        for k, t in enumerate(thetas):
            d = ds[0] if s.share_d else ds[k]
            psi_by, _, _ = _region_angles(t, delta, 10 ** logW, s.K22, d, psi)
            r, ab = _residuals_angles(pp, idx_of[t], [d] * len(idx_of[t]),
                                      [psi_by] * len(idx_of[t]), psi * DEG,
                                      s.free_ab)
            out.append(r); abs_.extend(ab)
        r = np.concatenate(out)
        return (r, abs_) if return_ab else r

    rf = {f.theta: f for f in region_fits}
    d0 = [np.mean([f.d for f in region_fits])] if s.share_d else \
         [rf[t].d for t in thetas]
    # ψ_rub is defined only mod 90° (T is even in Δn: a director field
    # rotated by 90° gives the same spectra). Weighted circular mean with
    # period 90°, weights 1/σ²; every region value is also tried as a start.
    ang = np.array([f.psi_rub for f in region_fits]) * 4 * DEG
    w = 1.0 / np.maximum([f.psi_rub_err for f in region_fits], 0.01) ** 2
    psi_mean = (np.degrees(np.arctan2((w * np.sin(ang)).sum(),
                                      (w * np.cos(ang)).sum())) / 4) % 90
    psi_starts = [psi_mean] + [psi_mean + ((f.psi_rub - psi_mean + 45) % 90 - 45)
                               for f in region_fits]
    psi0 = psi_mean

    lo = [s.d_range[0]] * nd + [psi0 - 60] + ([-20.0] if fit_delta else []) + [-9.0]
    hi = [s.d_range[1]] * nd + [psi0 + 60] + ([20.0] if fit_delta else []) + [-1.0]
    best = None
    starts = [(p0, w0, dl0) for p0 in psi_starts
              for w0 in (-7.0, -6.0, -5.0, -4.0, -3.0)
              for dl0 in ((-2.0, 0.0, 2.0) if fit_delta else (None,))]
    for p0, logW0, dl0 in starts:
        if True:
            x0 = list(d0) + [p0] + ([dl0] if fit_delta else []) + [logW0]
            r = least_squares(res, x0, bounds=(lo, hi),
                              x_scale=[0.1] * nd + [1.0] + ([1.0] if fit_delta else []) + [0.3])
            if best is None or r.cost < best.cost:
                best = r
    C = _cov(best)
    e = np.sqrt(np.abs(np.diag(C)))
    ds, psi, delta, logW = unpack(best.x)
    de = list(e[:nd])
    psi_e = e[nd]
    delta_e = e[nd + 1] if fit_delta else 0.0
    logW_e = e[-1]
    W = 10 ** logW
    phis, lags = [], []
    for k, t in enumerate(thetas):
        d = ds[0] if s.share_d else ds[k]
        _, phi, D = _region_angles(t, delta, W, s.K22, d, psi)
        phis.append(np.degrees(phi)); lags.append(np.degrees(D))
    _, ab = res(best.x, return_ab=True)
    at_bound = logW > hi[-1] - 0.05 or logW < lo[-1] + 0.05
    dl = ds * nreg if s.share_d else ds
    dle = de * nreg if s.share_d else de
    return JointFit(thetas, list(dl), list(dle), psi % 90, psi_e, delta,
                    delta_e, W, logW_e, s.K22 / W * 1e6, s.K22, phis, lags,
                    float(np.sqrt(np.mean(best.fun ** 2))), ab, at_bound,
                    fit_delta)


def model_curves(pp, s: FitSettings, jf: JointFit, lam_nm_list):
    """Fitted T on the full λ grid of each spectrum (for plotting/saving)."""
    out = []
    k_ab = 0
    for t_i, t in enumerate(jf.thetas):
        d = jf.d[t_i]
        psi_by, _, _ = _region_angles(t, jf.delta, jf.W, s.K22, d, jf.psi_rub)
        for i, sp in enumerate(pp.spectra):
            if sp.theta != t:
                continue
            lam_um = lam_nm_list[i] / 1000.0
            m = transmission(lam_um, s.dn_func(lam_um), d, psi_by,
                             jf.psi_rub * DEG, sp.P * DEG, sp.A * DEG)
            a, b = jf.ab[k_ab]; k_ab += 1
            out.append((i, a * m + b))
    out.sort(key=lambda x: x[0])
    return [m for _, m in out]


def pair_estimate(region_fits, K22):
    """Analytic cross-check from a ±θ pair of free region fits.

    |Φ±| = θ ± δ − Δ±  →  Δ ≈ θ − (|Φ+| + |Φ−|)/2,  δ ≈ (|Φ+| − |Φ−|)/2.
    Returns list of dicts (one per ±θ pair found).
    """
    out = []
    by = {round(f.theta, 6): f for f in region_fits}
    for t in sorted({round(f.theta, 6) for f in region_fits if f.theta > 0}):
        if t in by and round(-t, 6) in by:
            fp, fm = by[t], by[round(-t, 6)]
            pa, ma = abs(fp.phi), abs(fm.phi)
            D = t - (pa + ma) / 2
            dlt = (pa - ma) / 2
            d = (fp.d + fm.d) / 2
            W = W_from_twist((pa + ma) / 2 * DEG, D * DEG, K22, d) if D > 0 else np.inf
            D_err = 0.5 * np.hypot(fp.phi_err, fm.phi_err)
            out.append(dict(theta=t, lag=D, lag_err=D_err, delta=dlt, d=d, W=W))
    return out


def format_report(region_fits, jf: JointFit, pairs, s: FitSettings,
                  systematics=None, warnings=()):
    L = []
    L.append("=== Free fit per region (no anchoring model) ===")
    L.append("θ_set = BY easy axis vs rubbing as patterned; Φ = ψ_rub − ψ_BY")
    for f in region_fits:
        L.append(f"θ = {f.theta:+.1f}°  ({f.n_spectra} spectra): d = {f.d:.3f} ± {f.d_err:.3f} µm, "
                 f"ψ_rub = {f.psi_rub:.2f} ± {f.psi_rub_err:.2f}°, "
                 f"Φ = {f.phi:+.2f} ± {f.phi_err:.2f}°, rms = {f.rms:.4f}")
    for f in region_fits:
        if abs(f.theta) > 1 and np.sign(f.phi) == np.sign(f.theta):
            L.append(f"  ! θ = {f.theta:+.0f}°: twist sign opposite to the expected (Φ ≈ −θ). "
                     "Check that P/A angles and θ use the same rotation sense.")
    L.append("")
    L.append("=== Joint fit with anchoring (Rapini–Papoular, rubbed plate strong) ===")
    if s.share_d:
        L.append(f"d       = {jf.d[0]:.3f} ± {jf.d_err[0]:.3f} µm (shared)")
    else:
        for t, d, e in zip(jf.thetas, jf.d, jf.d_err):
            L.append(f"d(θ={t:+.0f}°) = {d:.3f} ± {e:.3f} µm")
    L.append(f"ψ_rub   = {jf.psi_rub:.2f} ± {jf.psi_rub_err:.2f}° (lab, mod 90°: spectra cannot tell the director from its normal)")
    if jf.fit_delta:
        L.append(f"δ       = {jf.delta:+.2f} ± {jf.delta_err:.2f}° (pattern rotation vs rubbing)")
    else:
        L.append(f"δ       = {jf.delta:+.2f}° FIXED (single region: W is only as good as this assumption)")
    rel = np.log(10) * jf.logW_err
    L.append(f"W       = {jf.W:.3g} J/m²  (+{(np.exp(rel) - 1) * 100:.0f} / −{(1 - np.exp(-rel)) * 100:.0f} %)"
             f"   with K22 = {jf.K22 * 1e12:.2f} pN")
    L.append(f"b=K22/W = {jf.b_um:.3g} µm  (independent of K22)")
    for t, phi, D in zip(jf.thetas, jf.phi, jf.lag):
        L.append(f"  region θ = {t:+.1f}°: Φ = {phi:+.2f}°, lag Δ = {D:.2f}°")
    L.append(f"rms residual = {jf.rms:.4f}")
    if jf.at_bound:
        L.append("  ! W at the fit bound: the lag is not resolved (anchoring too strong for this d) "
                 "or the twist is lost (too weak). Treat W as a limit only.")
    elif rel > 0.7:
        L.append("  ! W poorly determined (> factor 2). Use a thinner cell or larger θ.")
    if pairs:
        L.append("")
        L.append("=== Analytic ±θ cross-check (from the free fits) ===")
        for p in pairs:
            L.append(f"θ = ±{p['theta']:.1f}°: Δ = {p['lag']:.2f} ± {p['lag_err']:.2f}°, "
                     f"δ = {p['delta']:+.2f}°, W = {p['W']:.3g} J/m²")
    if systematics:
        L.append("")
        L.append("=== Δn(λ) dataset spread (refit) ===")
        for name, d, W in systematics:
            L.append(f"{name}: d = {d:.3f} µm, W = {W:.3g} J/m²")
    if warnings:
        L.append("")
        L.extend("! " + w for w in warnings)
    return "\n".join(L)


def run_all(spectra, s: FitSettings, extra_dn=None, progress=None):
    """Free region fits, joint fit, ±θ check, optional Δn systematics."""
    pp = prepare(spectra, s)
    thetas = sorted({sp.theta for sp in pp.spectra})
    rfs = []
    for t in thetas:
        if progress: progress(f"Scanning region θ = {t:+.0f}° …")
        idx = [i for i, sp in enumerate(pp.spectra) if sp.theta == t]
        rfs.append(fit_region(pp, idx, t, s))
    if progress: progress("Joint fit …")
    jf = fit_joint(pp, s, rfs)
    pairs = pair_estimate(rfs, s.K22)
    syst = []
    for name, f in (extra_dn or {}).items():
        if progress: progress(f"Systematics: {name} …")
        s2 = FitSettings(**{**s.__dict__, "dn_func": f,
                            "d_range": (0.8 * min(jf.d), 1.2 * max(jf.d))})
        pp2 = prepare(spectra, s2)
        rf2 = [fit_region(pp2, [i for i, sp in enumerate(pp2.spectra) if sp.theta == t], t, s2)
               for t in thetas]
        j2 = fit_joint(pp2, s2, rf2)
        syst.append((name, j2.d[0], j2.W))
    return pp, rfs, jf, pairs, syst
