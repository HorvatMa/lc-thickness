"""Spectrum loading and thickness fit for a planar LC cell between polarizers.

Planar cell, director at 45° to crossed polarizers, normal incidence:

    T(λ) = sin²(π Δn(λ) d / λ)          (parallel polarizers: cos²)

The fitted model is T = a·sin²(…) + b, where a absorbs polarizer/azimuth
imperfections and reflection losses and b absorbs leakage/stray light.
In reflection the light crosses the layer twice: Δn → 2Δn (callers pass the
effective birefringence passes·Δn(λ) as delta_n_func).

Without a reference spectrum, d follows from the wavelengths of the
transmission minima, Δn(λ_k) d/λ_k = integer (crossed polarizers), which
do not depend on the lamp spectrum (fit_minima).
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks


# ---------------------------------------------------------------- loading
def load_table(path):
    """Read the numeric rows of a spectrometer CSV/TXT export.

    Delimiter (',', ';', tab, whitespace) and decimal comma are detected
    automatically; header and footer lines that are not numeric are skipped.
    Returns a 2-D array (rows × columns).
    """
    with open(path, "r", errors="replace") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    def split(ln, delim):
        parts = ln.split(delim) if delim else ln.split()
        return [p.strip() for p in parts if p.strip() != ""]

    def to_float(s, dec_comma):
        if dec_comma:
            s = s.replace(",", ".")
        return float(s)

    best = None
    for delim, dec_comma in ((",", False), (";", True), (";", False),
                             ("\t", True), ("\t", False), (None, True),
                             (None, False)):
        rows = []
        for ln in lines:
            parts = split(ln, delim)
            if len(parts) < 2:
                continue
            try:
                rows.append([to_float(p, dec_comma) for p in parts])
            except ValueError:
                continue
        if not rows:
            continue
        # keep rows with the most common number of columns
        counts = {}
        for r in rows:
            counts[len(r)] = counts.get(len(r), 0) + 1
        ncol = max(counts, key=counts.get)
        rows = [r for r in rows if len(r) == ncol]
        if best is None or len(rows) > len(best) or (
                len(rows) == len(best) and ncol > len(best[0])):
            best = rows
    if not best:
        raise ValueError(f"No numeric data found in {path}")
    return np.array(best, dtype=float)


def to_nm(lam):
    """Convert a wavelength column to nm (µm or m inputs are detected)."""
    med = np.median(lam)
    if med < 1e-3:
        return lam * 1e9
    if med < 10:
        return lam * 1e3
    return lam


def spectrum_from_table(table, lam_col=0, int_col=1):
    """Return (λ [nm], I), sorted by wavelength with duplicates removed."""
    if max(lam_col, int_col) >= table.shape[1]:
        raise ValueError(f"File has only {table.shape[1]} columns")
    lam = to_nm(table[:, lam_col])
    inten = table[:, int_col]
    lam, idx = np.unique(lam, return_index=True)
    return lam, inten[idx]


def smooth(y, n):
    """Centred moving average over n points (n ≤ 1: unchanged)."""
    if n <= 1:
        return y
    kernel = np.ones(n) / n
    ypad = np.pad(y, (n // 2, n - 1 - n // 2), mode="edge")
    return np.convolve(ypad, kernel, mode="valid")


def normalized_transmission(lam_s, I_s, lam_r, I_r, lam_d=None, I_d=None,
                            min_ref_fraction=0.05):
    """T = (S − D)/(R − D) on the sample wavelength grid.

    Points where the dark-corrected reference is below min_ref_fraction of
    its maximum are masked (returned as NaN).
    """
    R = np.interp(lam_s, lam_r, I_r, left=np.nan, right=np.nan)
    S = I_s.astype(float)
    if lam_d is not None:
        D = np.interp(lam_s, lam_d, I_d)
        S = S - D
        R = R - D
    T = np.full_like(S, np.nan)
    good = np.isfinite(R) & (R > min_ref_fraction * np.nanmax(R))
    T[good] = S[good] / R[good]
    return T


def pair_transmission(lam_x, I_x, lam_p, I_p, lam_d=None, I_d=None,
                      min_ref_fraction=0.05):
    """Crossed transmission from a crossed/parallel pair of the same spot.

    For a non-absorbing sample T_x + T_p = 1, so T = S_x/(S_x + S_p) with
    S = I − D. Lamp, spectrometer and polarizer spectra cancel.
    """
    Sx = I_x.astype(float)
    Sp = np.interp(lam_x, lam_p, I_p, left=np.nan, right=np.nan)
    if lam_d is not None:
        D = np.interp(lam_x, lam_d, I_d)
        Sx = Sx - D
        Sp = Sp - D
    return normalized_transmission(lam_x, Sx, lam_x, Sx + Sp,
                                   min_ref_fraction=min_ref_fraction)


# ---------------------------------------------------------------- model
def basis(lam_um, dn, d_um, crossed=True):
    s = np.sin(np.pi * dn * d_um / lam_um) ** 2
    return s if crossed else 1.0 - s


def model(lam_um, dn, d_um, a=1.0, b=0.0, crossed=True):
    return a * basis(lam_um, dn, d_um, crossed) + b


@dataclass
class FitResult:
    d: float
    d_err: float
    a: float
    b: float
    rms: float
    n_points: int
    d_grid: np.ndarray = field(repr=False)
    rms_grid: np.ndarray = field(repr=False)
    second_d: float = np.nan     # next-best local minimum of the scan
    second_rms: float = np.nan
    minima: np.ndarray = None    # minima mode: detected minima [nm]
    orders: np.ndarray = None    # minima mode: assigned interference orders
    n_first: int = 0             # minima mode: how many minima are from file 1


def scan(lam_um, T, dn, d_grid, free_ab=True, crossed=True, chunk=256,
         ab_bounds=None):
    """RMS residual on a grid of thicknesses.

    With free_ab, a and b are solved by linear least squares at every d
    (a ≤ 0 is rejected, since it would invert the fringe pattern).
    ab_bounds = ((a_lo, a_hi), (b_lo, b_hi)) clips a and b to these ranges
    (alternating clipped updates; close to the bounded optimum).
    """
    N = lam_um.size
    k = np.pi * dn / lam_um
    out = np.empty(d_grid.size)
    a_out = np.ones(d_grid.size)
    b_out = np.zeros(d_grid.size)
    St0 = T.sum()
    for i in range(0, d_grid.size, chunk):
        dd = d_grid[i:i + chunk]
        s = np.sin(np.outer(dd, k)) ** 2
        if not crossed:
            s = 1.0 - s
        if free_ab:
            Sss = (s * s).sum(1)
            Ss = s.sum(1)
            St = (s * T).sum(1)
            det = Sss * N - Ss**2
            det[det == 0] = np.nan
            a = (St * N - Ss * St0) / det
            b = (Sss * St0 - Ss * St) / det
            if ab_bounds is not None:
                (a_lo, a_hi), (b_lo, b_hi) = ab_bounds
                a = np.clip(np.nan_to_num(a, nan=a_lo), a_lo, a_hi)
                for _ in range(2):
                    b = np.clip((St0 - a * Ss) / N, b_lo, b_hi)
                    a = np.clip((St - b * Ss) / Sss, a_lo, a_hi)
                b = np.clip((St0 - a * Ss) / N, b_lo, b_hi)
            rss = ((a[:, None] * s + b[:, None] - T) ** 2).sum(1)
            rss[~(a > 0)] = np.inf
            a_out[i:i + chunk] = a
            b_out[i:i + chunk] = b
        else:
            rss = ((s - T) ** 2).sum(1)
        out[i:i + chunk] = np.sqrt(rss / N)
    return out, a_out, b_out


def default_step(lam_um, dn):
    """Thickness step that changes the phase by ≤ 0.03 rad everywhere."""
    return 0.03 * lam_um.min() / (np.pi * np.max(np.abs(dn)))


def fit_thickness(lam_nm, T, delta_n_func, d_min, d_max, d0=None,
                  free_ab=True, crossed=True, max_grid=400_000,
                  ab_bounds=None):
    """Fit the cell thickness d [µm].

    A brute-force scan over [d_min, d_max] finds the global minimum (the
    χ² landscape has many local minima, one per interference order); it is
    then refined with nonlinear least squares. If d0 is given, the
    refinement starts from the best scan point (the scan range should
    already be centred on d0). ab_bounds (with free_ab) limits a and b,
    e.g. ((0.8, 1.05), (-0.05, 0.1)) for the crossed/parallel pair, where
    physically a ≈ 1 and b ≈ 0.
    """
    ok = np.isfinite(T)
    lam_um = lam_nm[ok] / 1000.0
    T = T[ok]
    if lam_um.size < 5:
        raise ValueError("Too few valid data points in the fit range")
    dn = delta_n_func(lam_um)

    d_min = max(d_min, 1e-3)
    step = default_step(lam_um, dn)
    n_grid = int(np.ceil((d_max - d_min) / step)) + 1
    if n_grid > max_grid:
        n_grid = max_grid
    d_grid = np.linspace(d_min, d_max, n_grid)
    if not free_ab:
        ab_bounds = None
    rms_grid, a_grid, b_grid = scan(lam_um, T, dn, d_grid, free_ab, crossed,
                                    ab_bounds=ab_bounds)
    if not np.any(np.isfinite(rms_grid)):
        raise ValueError("Scan failed (no finite residuals)")
    i0 = int(np.nanargmin(rms_grid))

    # next-best local minimum, at least half an order away
    order_sep = 0.5 * np.median(lam_um) / np.median(dn)
    loc = np.where((rms_grid[1:-1] < rms_grid[:-2])
                   & (rms_grid[1:-1] <= rms_grid[2:]))[0] + 1
    loc = loc[np.abs(d_grid[loc] - d_grid[i0]) > order_sep]
    second_d = second_rms = np.nan
    if loc.size:
        j = loc[np.argmin(rms_grid[loc])]
        second_d, second_rms = d_grid[j], rms_grid[j]

    # local refinement
    if free_ab:
        p0 = [d_grid[i0], a_grid[i0], b_grid[i0]]

        def resid(p):
            return model(lam_um, dn, p[0], p[1], p[2], crossed) - T
    else:
        p0 = [d_grid[i0]]

        def resid(p):
            return model(lam_um, dn, p[0], 1.0, 0.0, crossed) - T

    if ab_bounds is not None:
        ab_lo = [ab_bounds[0][0], ab_bounds[1][0]]
        ab_hi = [ab_bounds[0][1], ab_bounds[1][1]]
    else:
        ab_lo, ab_hi = [-np.inf] * 2, [np.inf] * 2
    n_ab = len(p0) - 1
    lo = [max(d_grid[i0] - 2 * step, 1e-3)] + ab_lo[:n_ab]
    hi = [d_grid[i0] + 2 * step] + ab_hi[:n_ab]
    eps = 1e-9   # least_squares needs a start strictly inside the bounds
    p0 = np.clip(p0, np.array(lo) + eps, np.array(hi) - eps)
    res = least_squares(resid, p0, bounds=(lo, hi), x_scale="jac")
    # polish with d free (bounds on d only kept the solver in the right order)
    lo[0], hi[0] = 1e-3, np.inf
    res = least_squares(resid, np.clip(res.x, np.array(lo) + eps,
                                       np.array(hi) - eps),
                        bounds=(lo, hi), x_scale="jac")

    N, P = T.size, len(p0)
    rss = float(np.sum(res.fun**2))
    try:
        cov = np.linalg.inv(res.jac.T @ res.jac) * rss / max(N - P, 1)
        d_err = float(np.sqrt(cov[0, 0]))
    except np.linalg.LinAlgError:
        d_err = np.nan

    a, b = (res.x[1], res.x[2]) if free_ab else (1.0, 0.0)
    return FitResult(d=float(res.x[0]), d_err=d_err, a=float(a), b=float(b),
                     rms=float(np.sqrt(rss / N)), n_points=N,
                     d_grid=d_grid, rms_grid=rms_grid,
                     second_d=float(second_d), second_rms=float(second_rms))


def wavelength_window(lam_nm, T, lam_min, lam_max):
    m = (lam_nm >= lam_min) & (lam_nm <= lam_max)
    return lam_nm[m], T[m]


# ---------------------------------------------------------------- minima
def find_minima(lam_nm, S, prominence=0.1):
    """Wavelengths [nm] of the spectral minima of S.

    prominence is relative to the range of S. Each minimum is refined by a
    parabola through the neighbouring points.
    """
    ok = np.isfinite(S)
    lam_nm, S = lam_nm[ok], S[ok]
    span = S.max() - S.min()
    idx, _ = find_peaks(-S, prominence=prominence * span)
    out = []
    for j, i in enumerate(idx):
        gaps = [i - idx[j - 1] if j > 0 else np.inf,
                idx[j + 1] - i if j + 1 < len(idx) else np.inf]
        gap = min(gaps) if np.isfinite(min(gaps)) else 40
        w = max(2, int(gap / 8))
        lo, hi = max(i - w, 0), min(i + w + 1, lam_nm.size)
        x0 = lam_nm[i]
        if hi - lo >= 3:
            c2, c1, _ = np.polyfit(lam_nm[lo:hi] - x0, S[lo:hi], 2)
            v = -c1 / (2 * c2) if c2 > 0 else 0.0
            if abs(v) <= lam_nm[hi - 1] - lam_nm[lo]:
                x0 += v
        out.append(x0)
    return np.array(out)


def fit_minima(lam_min_nm, delta_n_func, d_min, d_max, crossed=True,
               lam_min2_nm=None):
    """Thickness from the minima positions alone.

    Crossed polarizers: Δn(λ_k) d/λ_k = q_k with integer q_k; parallel:
    half-integer. Consecutive minima (sorted from blue to red) are assumed
    to be consecutive orders, q_k = q_0 − k. For each q_0, d follows by
    linear least squares; the q_0 with the smallest residual (in units of
    order) inside [d_min, d_max] wins. A missed or spurious minimum breaks
    the assumption — check the markers in the plot.

    lam_min2_nm: optional minima of a second spectrum of the same spot taken
    with the other polarizer setting (orders shifted by ½). Both sets share
    d; each has its own q_0. Only positions are used, so the two spectra
    may have arbitrary, different intensity scales.
    """
    sets = [(np.sort(np.asarray(lam_min_nm)) / 1000.0,
             0.0 if crossed else 0.5)]
    if lam_min2_nm is not None and len(lam_min2_nm):
        sets.append((np.sort(np.asarray(lam_min2_nm)) / 1000.0,
                     0.5 if crossed else 0.0))
    if sets[0][0].size < 2:
        raise ValueError("Need at least 2 minima in the fit range")
    g_sets = [delta_n_func(lam) / lam for lam, _ in sets]  # orders per µm
    q0_ranges = []
    for (lam, shift), g in zip(sets, g_sets):
        lo = lam.size - 1 + (1 if shift == 0 else 0)
        hi = max(int(np.ceil(d_max * g[0])) + 1, lo)
        q0_ranges.append(range(lo, hi + 1))

    g = np.concatenate(g_sets)
    cands = []                              # (rms, d, q)
    for q0s in np.array(np.meshgrid(*q0_ranges)).reshape(len(sets), -1).T:
        q = np.concatenate([q0 - np.arange(lam.size) + shift
                            for q0, (lam, shift) in zip(q0s, sets)])
        d = float(np.sum(q * g) / np.sum(g * g))
        if d < d_min or d > d_max:
            continue
        cands.append((float(np.sqrt(np.mean((g * d - q) ** 2))), d, q,
                      int(q0s[0])))
    if not cands:
        raise ValueError("No interference order gives d inside the scan range")
    cands.sort(key=lambda c: c[0])
    rms, d, q, _ = cands[0]
    K = g.size
    resid = g * d - q
    d_err = (float(np.sqrt(np.sum(resid**2) / (K - 1)) / np.sqrt(np.sum(g * g)))
             if K > 1 else np.nan)
    # next-best assignment with a clearly different d (> ¼ order away)
    sep = 0.25 / np.mean(g)
    second = next((c for c in cands[1:] if abs(c[1] - d) > sep), None)
    # scan plot: one point per order q_0 of the first set (best partner)
    best = {}
    for c in cands:
        best.setdefault(c[3], c)
    ds = np.array([c[1] for c in best.values()])
    rr = np.array([c[0] for c in best.values()])
    o = np.argsort(ds)
    return FitResult(d=d, d_err=d_err, a=np.nan, b=np.nan, rms=rms,
                     n_points=K, d_grid=ds[o], rms_grid=rr[o],
                     second_d=second[1] if second else np.nan,
                     second_rms=second[0] if second else np.nan,
                     minima=np.concatenate([lam for lam, _ in sets]) * 1000,
                     orders=q,
                     n_first=sets[0][0].size)


def predicted_minima(lam_nm, delta_n_func, d, crossed=True):
    """Wavelengths [nm] where the model has zero transmission."""
    lam = np.linspace(lam_nm.min(), lam_nm.max(), 20000) / 1000
    phase = delta_n_func(lam) * d / lam - (0.0 if crossed else 0.5)
    fl = np.floor(phase)
    j = np.where(fl[1:] != fl[:-1])[0]
    # linear interpolation of the integer crossing
    t = (np.maximum(fl[j], fl[j + 1]) - phase[j]) / (phase[j + 1] - phase[j])
    return (lam[j] + t * (lam[j + 1] - lam[j])) * 1000
