"""Guided acquisition of twisted-cell spectra (ASEQ spectrometer, rotation stage).

The app tells you what to set (region, stage angle ρ, crossed/parallel
polarizers, lamp blocked for darks). You set it and press OK (or Enter); the
app sets the exposure, acquires the spectrum and saves it into one measurement
folder together with ``measurement.json``. ``twist_app.py`` only needs that
folder.

Conventions (see the note "Anchoring strength from twisted-cell spectra"):
- polarizer fixed at P0 = 0° (defines the lab zero); analyzer at 90° (crossed)
  or 0° (parallel);
- rotating the sample by ρ is equivalent to rotating both polarizers by −ρ, so
  the angles stored for the fit are P = −s·ρ, A = A_lab − s·ρ, where s = +1 if
  the stage angle increases in the same sense as θ (e.g. counter-clockwise as
  seen on the camera), else −1.

Run:   uv run twist_acquire.py               (real spectrometer)
       uv run twist_acquire.py --simulate     (fake twisted cell, no hardware)
"""
import argparse
import datetime as dt
import glob
import json
import os
import sys
import time

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

APP_VERSION = 1
META = "measurement.json"
FIRST_ACTIVE, N_DARK_PIXELS = 32, 32      # frame layout, as in aseq.py


def ladder_ms(max_ms):
    """Auto-exposure ladder 0.5, 1, 2, 5, 10, … ms up to ``max_ms`` (the GUI
    safety maximum); ``max_ms`` itself is added if it is not a ladder value."""
    out, dec = [0.5], 1.0
    while True:
        for f in (1, 2, 5):
            t = f * dec
            if t > max_ms + 1e-9:
                if max_ms > out[-1] + 1e-9:
                    out.append(float(max_ms))
                return [t for t in out if t <= max_ms + 1e-9] or [float(max_ms)]
            out.append(t)
        dec *= 10


# ------------------------------------------------------------------ device
def find_aseq_dir():
    """Locate the aseq-spectrometer repo (env ASEQ_DIR, or a sibling folder)."""
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [os.environ.get("ASEQ_DIR", "")] + [
        os.path.join(os.path.dirname(here), n)
        for n in ("aseq_spectrometer", "aseq-spectrometer")]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "aseq.py")):
            return c
    return None


def open_spectrometer(simulate=False, max_exposure_ms=1000.0):
    if simulate:
        return FakeSpectrometer(max_exposure_ms)
    d = find_aseq_dir()
    if d is None:
        raise RuntimeError("aseq.py not found: clone HorvatMa/aseq-spectrometer next to "
                           "lc_thickness or set the ASEQ_DIR environment variable.")
    sys.path.insert(0, d)
    import aseq
    return aseq.ASEQSpectrometer(max_exposure_ms=max_exposure_ms)


class FakeSpectrometer:
    """Simulated LR2 looking at a twisted E7 cell (for testing without hardware).

    The app tells it the current state via ``state`` (theta, rho, pol, dark).
    """
    serial = "SIMULATED"

    def __init__(self, max_exposure_ms=1000.0, d_um=20.0, W=9e-6, delta=1.0,
                 psi_rub=10.0, sense=1, seed=0):
        import twist_model as tm
        from dispersion import MATERIALS
        self.tm, self.max_exposure_ms = tm, max_exposure_ms
        self.dn = MATERIALS["E7"]["Li et al. 2005 (25 °C)"].delta_n
        self.wavelengths = np.linspace(294.9, 999.4, 3653)
        lam = self.wavelengths
        self.lamp = 9000 * np.exp(-((lam - 600) / 120) ** 2) * (lam > 400)  # counts/100 ms
        self.par = dict(d=d_um, W=W, delta=delta, psi=psi_rub, sense=sense)
        self.rng = np.random.default_rng(seed)
        self.exp = 10.0
        self.state = dict(dark=True)

    def set_exposure(self, ms):
        if ms > self.max_exposure_ms:
            raise ValueError("exposure above limit")
        self.exp = ms
        return ms

    @property
    def exposure_ms(self):
        return self.exp

    def acquire_frames(self, n=1):
        time.sleep(min(self.exp * n / 1000, 0.3))
        lam = self.wavelengths / 1000
        st = self.state
        if st.get("dark"):
            sig = 0.0 * lam
        else:
            p, tm = self.par, self.tm
            psi_by, _, _ = tm._region_angles(st["theta"], p["delta"], p["W"], 6.2e-12,
                                             p["d"], p["psi"])
            rho = p["sense"] * st["rho"] * tm.DEG
            A = (90.0 if st["pol"] == "crossed" else 0.0) * tm.DEG
            T = tm.transmission(lam, self.dn(lam), p["d"], psi_by,
                                p["psi"] * tm.DEG, -rho, A - rho)
            sig = self.lamp * T
        dark_current = 3.0 * (self.exp / 100)        # counts, flat
        frames = []
        for _ in range(n):
            s = sig * self.exp / 100 + dark_current + 2500
            s = s + self.rng.normal(0, 8 + 0.01 * np.sqrt(np.abs(s)) * 10, s.size)
            f = np.full(3694, 2500.0)
            f[:N_DARK_PIXELS] = 2500 + self.rng.normal(0, 5, N_DARK_PIXELS)
            f[FIRST_ACTIVE:FIRST_ACTIVE + s.size] = s
            frames.append(np.clip(f, 0, 16383))
        return np.array(frames)

    def close(self):
        pass


def grab(spec, exposure_ms, n_avg, discard=0):
    """Offset-corrected mean spectrum, raw maximum and offset (same as aseq.measure).

    The driver (aseq.py, fixed 23. 11. 2026) waits for the CCD to settle after new
    acquisition parameters; after an exposure change the first frame can still be
    ~3 % low, so ``discard`` leading frames (default 1) are dropped.
    """
    if abs(spec.exposure_ms - exposure_ms) > 1e-6:
        spec.set_exposure(exposure_ms)
    frames = np.asarray(spec.acquire_frames(n_avg + discard), dtype=float)[discard:]
    wl = spec.wavelengths
    active = frames[:, FIRST_ACTIVE:FIRST_ACTIVE + len(wl)]
    offset = frames[:, 4:N_DARK_PIXELS].mean()
    return wl, active.mean(0) - offset, float(active.max()), float(offset)


# ------------------------------------------------------------------ steps
def build_steps(cfg):
    """List of steps: dicts with key, kind, text and the acquisition state."""
    steps = [dict(key="dark_start", kind="dark", when="start",
                  text="Block the lamp (dark spectra, start).")]
    for th in cfg["regions_deg"]:
        steps.append(dict(key=f"move_{th:+g}", kind="move", theta=th,
                          text=f"Move to region θ = {th:+g}° (translate only, do not rotate).\n"
                               f"Unblock the lamp."))
        for rho in cfg["rho_deg"]:
            for pol in ("crossed", "parallel"):
                steps.append(dict(
                    key=f"R{th:+g}_rho{rho:g}_{pol}", kind="sample", theta=th,
                    rho=rho, pol=pol,
                    text=f"Region θ = {th:+g}°\nStage ρ = {rho:g}°\n"
                         f"Polarizers {pol.upper()}"))
    steps.append(dict(key="dark_end", kind="dark", when="end",
                      text="Block the lamp (dark spectra, end)."))
    return steps


def angles(cfg, rho, pol):
    s = cfg["stage_sense"]
    A_lab = 90.0 if pol == "crossed" else 0.0
    return -s * rho, A_lab - s * rho


def fname(step):
    if step["kind"] == "sample":
        return f"R{step['theta']:+g}_rho{step['rho']:g}_{step['pol']}.txt"
    raise ValueError


# ------------------------------------------------------------------ pattern
# Same format as dynamic_cell/patterns/patterns.ipynb: 1200 x 1920 array of
# angles in degrees (1.0 x 1.5 mm), saved with np.save. The values are the
# exposure POLARIZATION angle (BY easy axis is perpendicular to it), wrapped
# into [90°, 270°) like the notebook, 0° along
# the long (1920 px) axis, counter-clockwise as seen on the camera; row 0 is
# the bottom (imshow origin="lower", as in the notebook).
PATTERN_RES = (1200, 1920)
PATTERN_SIZE_MM = (1.0, 1.5)

# 5 x 7 bitmap glyphs for the region labels
_GLYPHS = {
    "+": ["00000", "00100", "00100", "11111", "00100", "00100", "00000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    "°": ["01100", "10010", "10010", "01100", "00000", "00000", "00000"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11111", "00010", "00100", "00010", "00001", "10001", "01110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
}


def text_mask(text, scale):
    """Boolean mask (rows top→bottom) of ``text`` drawn with the 5x7 glyphs."""
    cols = []
    for ch in text:
        g = np.array([[c == "1" for c in row] for row in _GLYPHS[ch]])
        cols += [g, np.zeros((7, 1), bool)]
    m = np.hstack(cols[:-1]) if cols else np.zeros((7, 0), bool)
    return np.kron(m, np.ones((scale, scale), bool))


def label_text(theta):
    sign = "+" if theta > 0 else "-" if theta < 0 else ""
    return f"{sign}{abs(theta):g}°"


def pol_angle(easy_axis_deg):
    """Exposure polarization angle that writes the given BY easy axis.

    Wrapped into [90°, 270°), as in patterns.ipynb (values below 90° or above
    270° get +180°); the polarization is only defined mod 180°.
    """
    v = (easy_axis_deg + 90.0) % 180.0
    return v + 180.0 if v < 90.0 else v


def make_pattern(thetas, label_easy_deg=0.0, rub_deg=0.0, scale=14, bottom_px=80):
    """Pattern for the twisted-cell anchoring measurement.

    The image is split along the long axis into len(thetas) equal parts (two
    regions → halves, the first θ on the left). Region k has its easy axis at
    rub + θ_k. Its label (e.g. "+80°") is written near the bottom centre of
    the region with easy axis ``label_easy_deg`` (default: along the rubbing,
    i.e. untwisted, so it shows up against the twisted region).
    """
    ny, nx = PATTERN_RES
    pol = np.zeros(PATTERN_RES)
    edges = np.linspace(0, nx, len(thetas) + 1).round().astype(int)
    for k, th in enumerate(thetas):
        j0, j1 = edges[k], edges[k + 1]
        pol[:, j0:j1] = pol_angle(rub_deg + th)
        sc = scale
        while sc > 2 and text_mask(label_text(th), sc).shape[1] > 0.8 * (j1 - j0):
            sc -= 1                                       # shrink to fit narrow regions
        m = text_mask(label_text(th), sc)[::-1]          # flip: row 0 = bottom
        h, w = m.shape
        c0 = (j0 + j1) // 2 - w // 2
        sub = pol[bottom_px:bottom_px + h, c0:c0 + w]
        sub[m] = pol_angle(label_easy_deg)
    return pol


def draw_pattern(ax, pol, thetas, label_easy_deg):
    """Draw a pattern (as from make_pattern) on ``ax``; returns the image."""
    ext = [0, PATTERN_SIZE_MM[1], 0, PATTERN_SIZE_MM[0]]
    im = ax.imshow(pol[::2, ::2], origin="lower", cmap="twilight_shifted", vmin=90,
                   vmax=270, extent=ext, interpolation="nearest")
    edges = np.linspace(0, PATTERN_SIZE_MM[1], len(thetas) + 1)
    for k, th in enumerate(thetas):
        ax.text((edges[k] + edges[k + 1]) / 2, 0.93 * PATTERN_SIZE_MM[0],
                f"θ = {label_text(th)}\neasy axis {th % 180:g}° · pol {pol_angle(th):g}°",
                ha="center", va="top", fontsize=8, color="w")
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm), rubbing along x"); ax.set_ylabel("y (mm)")
    ax.set_title(f"Pattern preview ({PATTERN_RES[1]}×{PATTERN_RES[0]} px); labels: easy axis "
                 f"{label_easy_deg:g}° · pol {pol_angle(label_easy_deg):g}°", fontsize=9)
    return im


def save_pattern(path, thetas, label_easy_deg=0.0):
    """Save .npy (degrees), a preview .png and a .json with the parameters."""
    pol = make_pattern(thetas, label_easy_deg)
    base = os.path.splitext(path)[0]
    np.save(base + ".npy", pol)
    fig = Figure(figsize=(8, 5.5), tight_layout=True)
    ax = fig.add_subplot(111)
    im = draw_pattern(ax, pol, thetas, label_easy_deg)
    fig.colorbar(im, ax=ax, label="exposure polarization (°)", shrink=0.7)
    FigureCanvas(fig).print_png(base + ".png")
    info = dict(app="twist_acquire", version=APP_VERSION,
                created=dt.datetime.now().isoformat(timespec="seconds"),
                file=os.path.basename(base + ".npy"), resolution=list(PATTERN_RES),
                size_mm=list(PATTERN_SIZE_MM), values="exposure polarization angle (deg, in [90, 270)), "
                "0 along the long axis, CCW on camera, row 0 = bottom",
                rubbing_deg=0.0, regions_deg=list(thetas),
                region_pol_deg=[pol_angle(t) for t in thetas],
                label_easy_axis_deg=label_easy_deg,
                label_pol_deg=pol_angle(label_easy_deg))
    with open(base + ".json", "w") as f:
        json.dump(info, f, indent=1)
    return base + ".npy"


class Worker(QtCore.QObject):
    """Runs one acquisition in a thread."""
    done = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)
    status = QtCore.pyqtSignal(str)

    def __init__(self, spec, cfg, step, last_exp):
        super().__init__()
        self.spec, self.cfg, self.step, self.last_exp = spec, cfg, step, last_exp

    def run(self):
        try:
            self.done.emit(self.acquire())
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")

    def dark_exposures(self):
        c = self.cfg
        if c["exposure_mode"] == "manual":
            return sorted({c["exp_crossed_ms"], c["exp_parallel_ms"]})
        if self.step.get("when") == "end":
            used = sorted({float(s["exposure_ms"]) for s in c.get("spectra", [])})
            if used:                      # end darks only at the exposures used
                return used
        return ladder_ms(c["max_exposure_ms"])

    def auto_exposure(self, pol):
        c = self.cfg
        sat = c["saturation_raw"]
        ladder = ladder_ms(c["max_exposure_ms"])
        t = self.last_exp.get(pol, 10.0 if pol == "crossed" else 2.0)
        t = min(ladder, key=lambda x: abs(x - t))
        lo, hi = c["auto_lambda_nm"]
        for _ in range(12):
            self.status.emit(f"Auto exposure: test at {t:g} ms …")
            wl, s, raw_max, off = grab(self.spec, t, 1, c.get("discard_frames", 1))
            m = (wl >= lo) & (wl <= hi)
            peak = float(s[m].max())
            if raw_max >= sat:
                i = ladder.index(t)
                if i == 0:
                    return t
                t = ladder[i - 1]
                continue
            ideal = t * c["auto_target"] * (sat - off) / max(peak, 1.0)
            new = max([x for x in ladder if x <= ideal], default=ladder[0])
            if new == t:
                return t
            if new > t and raw_max < c["auto_target"] * sat and new == ladder[-1] and t == ladder[-1]:
                return t
            t = new
        return t

    def acquire(self):
        c, st = self.cfg, self.step
        n = c["n_avg"]
        if st["kind"] == "dark":
            out = []
            for t in self.dark_exposures():
                self.status.emit(f"Dark {st['when']}: {t:g} ms × {n}")
                wl, s, raw_max, off = grab(self.spec, t, n, c.get("discard_frames", 1))
                out.append((t, wl, s, raw_max, off))
            return ("dark", out)
        pol = st["pol"]
        if c["exposure_mode"] == "manual":
            t = c["exp_crossed_ms"] if pol == "crossed" else c["exp_parallel_ms"]
        else:
            t = self.auto_exposure(pol)
        self.status.emit(f"Acquiring {t:g} ms × {n} …")
        wl, s, raw_max, off = grab(self.spec, t, n, c.get("discard_frames", 1))
        return ("sample", [(t, wl, s, raw_max, off)])


# ------------------------------------------------------------------ GUI
def dspin(lo, hi, val, dec=2, step=1.0, suffix=""):
    w = QtWidgets.QDoubleSpinBox()
    w.setRange(lo, hi); w.setDecimals(dec); w.setSingleStep(step); w.setValue(val)
    if suffix:
        w.setSuffix(suffix)
    return w


def parse_list(text):
    return [float(x) for x in text.replace(";", ",").replace(" ", ",").split(",") if x.strip()]


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, simulate=False):
        super().__init__()
        self.simulate = simulate
        self.setWindowTitle("Twisted-cell acquisition" + (" [SIMULATED]" if simulate else ""))
        self.qs = QtCore.QSettings("lc_thickness", "twist_acquire")
        self.spec = None
        self.folder = None
        self.meta = None
        self.steps = []
        self.cur = 0
        self.last_exp = {}
        self.busy = False

        # ---- setup form
        self.sample = QtWidgets.QLineEdit(self.qs.value("sample", "cell"))
        self.base = QtWidgets.QLineEdit(self.qs.value("base", os.path.expanduser("~")))
        bb = QtWidgets.QPushButton("…"); bb.setFixedWidth(30); bb.clicked.connect(self.pick_base)
        self.regions = QtWidgets.QLineEdit(self.qs.value("regions", "+80, -80"))
        self.rhos = QtWidgets.QLineEdit(self.qs.value("rhos", "0, 22.5, 45, 67.5"))
        self.sense = QtWidgets.QComboBox()
        self.sense.addItems(["Stage angle increases in the SAME sense as θ",
                             "Stage angle increases OPPOSITE to θ"])
        self.navg = QtWidgets.QSpinBox(); self.navg.setRange(1, 100)
        self.navg.setValue(int(self.qs.value("navg", 5)))
        self.discard = QtWidgets.QSpinBox(); self.discard.setRange(0, 20)
        self.discard.setValue(int(self.qs.value("discard", 1)))
        self.expmode = QtWidgets.QComboBox(); self.expmode.addItems(["Auto", "Manual"])
        self.exp_x = dspin(0.01, 100000, float(self.qs.value("exp_x", 200)), 2, 10, " ms")
        self.exp_p = dspin(0.01, 100000, float(self.qs.value("exp_p", 20)), 2, 10, " ms")
        self.maxexp = dspin(0.01, 100000, float(self.qs.value("maxexp", 1000)), 2, 100, " ms")
        self.sat = QtWidgets.QSpinBox(); self.sat.setRange(1000, 65535)
        self.sat.setValue(int(self.qs.value("sat", 16000)))
        self.target = dspin(0.1, 0.95, 0.7, 2, 0.05)
        self.alo = dspin(300, 1000, 480, 0, 10, " nm")
        self.ahi = dspin(300, 1000, 720, 0, 10, " nm")
        self.expmode.currentIndexChanged.connect(self.mode_changed)
        self.label_easy = dspin(-90, 180, float(self.qs.value("label_easy", 0)), 1, 5, " °")
        self.btn_pattern = QtWidgets.QPushButton("Generate pattern .npy…")
        self.btn_pattern.clicked.connect(self.generate_pattern)
        self._pat_timer = QtCore.QTimer(self); self._pat_timer.setSingleShot(True)
        self._pat_timer.setInterval(300)
        self._pat_timer.timeout.connect(self.preview_pattern)
        self.regions.textChanged.connect(self._pat_timer.start)
        self.label_easy.valueChanged.connect(self._pat_timer.start)

        form = QtWidgets.QFormLayout()
        form.addRow("Sample name", self.sample)
        fb = QtWidgets.QHBoxLayout(); fb.addWidget(self.base); fb.addWidget(bb)
        form.addRow("Save under", fb)
        form.addRow("Regions θ (°)", self.regions)
        pr = QtWidgets.QHBoxLayout()
        pr.addWidget(QtWidgets.QLabel("label easy axis")); pr.addWidget(self.label_easy)
        pr.addWidget(self.btn_pattern)
        form.addRow("  pattern", pr)
        form.addRow("Stage angles ρ (°)", self.rhos)
        form.addRow("Stage sense", self.sense)
        form.addRow("Averages per spectrum", self.navg)
        form.addRow("Discard first frames", self.discard)
        form.addRow("Exposure", self.expmode)
        form.addRow("  crossed", self.exp_x)
        form.addRow("  parallel", self.exp_p)
        form.addRow("  max (safety)", self.maxexp)
        form.addRow("  auto: target of saturation", self.target)
        ar = QtWidgets.QHBoxLayout(); ar.addWidget(self.alo); ar.addWidget(self.ahi)
        form.addRow("  auto: λ window", ar)
        form.addRow("Saturation (raw counts)", self.sat)
        self.setup_box = QtWidgets.QGroupBox("Setup")
        self.setup_box.setLayout(form)
        self.btn_start = QtWidgets.QPushButton("Connect && start new measurement")
        self.btn_resume = QtWidgets.QPushButton("Resume folder…")
        self.btn_start.clicked.connect(self.start_new)
        self.btn_resume.clicked.connect(self.resume)

        # ---- steps + instruction
        self.list = QtWidgets.QListWidget()
        self.list.itemClicked.connect(self.step_clicked)
        self.instr = QtWidgets.QLabel("Set up and press Connect.")
        f = self.instr.font(); f.setPointSize(f.pointSize() + 8); f.setBold(True)
        self.instr.setFont(f)
        self.instr.setWordWrap(True)
        self.instr.setMinimumHeight(140)
        self.btn_ok = QtWidgets.QPushButton("OK — acquire  (Enter)")
        self.btn_ok.setEnabled(False)
        self.btn_ok.clicked.connect(self.do_step)
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Return), self, self.do_step)
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Enter), self, self.do_step)
        self.info = QtWidgets.QLabel("")

        left = QtWidgets.QWidget(); ll = QtWidgets.QVBoxLayout(left)
        ll.addWidget(self.setup_box)
        hb = QtWidgets.QHBoxLayout(); hb.addWidget(self.btn_start); hb.addWidget(self.btn_resume)
        ll.addLayout(hb)
        ll.addWidget(QtWidgets.QLabel("Steps (click one to redo it):"))
        ll.addWidget(self.list, 1)

        self.fig = Figure(figsize=(7, 4), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        right = QtWidgets.QWidget(); rl = QtWidgets.QVBoxLayout(right)
        rl.addWidget(self.instr)
        rl.addWidget(self.btn_ok)
        rl.addWidget(self.info)
        rl.addWidget(self.canvas, 1)

        split = QtWidgets.QSplitter(); split.addWidget(left); split.addWidget(right)
        split.setSizes([450, 900])
        self.setCentralWidget(split)
        self.status = self.statusBar()
        self.mode_changed()
        self.preview_pattern()

    # ---------------------------------------------------------- setup
    def mode_changed(self, *_):
        manual = self.expmode.currentText() == "Manual"
        self.exp_x.setEnabled(manual); self.exp_p.setEnabled(manual)
        self.target.setEnabled(not manual)

    def pick_base(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Save under", self.base.text())
        if d:
            self.base.setText(d)

    def pattern_thetas(self):
        thetas = parse_list(self.regions.text())
        if not thetas:
            raise ValueError("Give at least one region θ.")
        if any(abs(abs(t) - 90) < 1e-6 for t in thetas):
            raise ValueError("θ = ±90° gives a degenerate twist sense; use e.g. ±80°.")
        return thetas

    def preview_pattern(self):
        """Show the pattern for the current regions in the plot (setup phase only)."""
        if not self.setup_box.isVisible() and self.meta is not None:
            return
        self.ax.clear()
        try:
            thetas = self.pattern_thetas()
            draw_pattern(self.ax, make_pattern(thetas, self.label_easy.value()),
                         thetas, self.label_easy.value())
        except Exception as e:
            self.ax.set_axis_off()
            self.ax.set_title(f"Pattern preview: {e}", fontsize=9)
        self.canvas.draw_idle()

    def generate_pattern(self):
        try:
            thetas = self.pattern_thetas()
            name = "twist_pattern_" + "_".join(f"{t:+g}" for t in thetas) + ".npy"
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save pattern", os.path.join(self.base.text(), name), "NumPy (*.npy)")
            if not path:
                return
            out = save_pattern(path, thetas, self.label_easy.value())
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Pattern", str(e)); return
        self.qs.setValue("label_easy", self.label_easy.value())
        pols = ", ".join(f"{label_text(t)} → pol {pol_angle(t):g}°" for t in thetas)
        self.status.showMessage(f"Saved {out} (+ .png, .json): {pols}")

    def config(self):
        regions = parse_list(self.regions.text())
        rhos = parse_list(self.rhos.text())
        if not regions or not rhos:
            raise ValueError("Give at least one region and one stage angle.")
        if any(abs(abs(t) - 90) < 1e-6 for t in regions):
            raise ValueError("θ = ±90° gives a degenerate twist sense; use e.g. ±80°.")
        return dict(
            regions_deg=regions, rho_deg=rhos,
            stage_sense=1 if self.sense.currentIndex() == 0 else -1,
            n_avg=self.navg.value(), discard_frames=self.discard.value(),
            exposure_mode="manual" if self.expmode.currentText() == "Manual" else "auto",
            exp_crossed_ms=self.exp_x.value(), exp_parallel_ms=self.exp_p.value(),
            max_exposure_ms=self.maxexp.value(), saturation_raw=self.sat.value(),
            auto_target=self.target.value(),
            auto_lambda_nm=[self.alo.value(), self.ahi.value()])

    def save_settings(self):
        for k, w in (("sample", self.sample), ("base", self.base),
                     ("regions", self.regions), ("rhos", self.rhos)):
            self.qs.setValue(k, w.text())
        for k, w in (("navg", self.navg), ("discard", self.discard), ("exp_x", self.exp_x), ("exp_p", self.exp_p),
                     ("maxexp", self.maxexp), ("sat", self.sat)):
            self.qs.setValue(k, w.value())

    def connect(self, cfg):
        if self.spec is None:
            self.status.showMessage("Connecting …")
            QtWidgets.QApplication.processEvents()
            self.spec = open_spectrometer(self.simulate, cfg["max_exposure_ms"])
        else:
            self.spec.max_exposure_ms = cfg["max_exposure_ms"]

    def start_new(self):
        try:
            cfg = self.config()
            self.connect(cfg)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Start", str(e)); return
        self.save_settings()
        stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M")
        name = "".join(ch if ch.isalnum() or ch in "-_+" else "_" for ch in self.sample.text())
        self.folder = os.path.join(self.base.text(), f"{stamp}_{name}")
        os.makedirs(self.folder, exist_ok=False)
        self.meta = dict(app="twist_acquire", version=APP_VERSION, sample=self.sample.text(),
                         created=dt.datetime.now().isoformat(timespec="seconds"),
                         spectrometer=getattr(self.spec, "serial", "?"),
                         simulated=self.simulate,
                         P0_deg=0.0, A_crossed_lab_deg=90.0, A_parallel_lab_deg=0.0,
                         **cfg, spectra=[], darks=[])
        self.write_meta()
        self.begin()

    def resume(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Measurement folder", self.base.text())
        if not d:
            return
        try:
            with open(os.path.join(d, META)) as f:
                meta = json.load(f)
            self.connect(meta)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Resume", str(e)); return
        self.folder, self.meta = d, meta
        self.begin()

    def begin(self):
        self.setup_box.setVisible(False)
        self.btn_start.setEnabled(False); self.btn_resume.setEnabled(False)
        self.steps = build_steps(self.meta)
        done = {s["key"] for s in self.meta["spectra"]} | {f"dark_{d['when']}" for d in self.meta["darks"]}
        moved = {s["theta"] for s in self.meta["spectra"]}
        for st in self.steps:
            st["done"] = st["key"] in done or (st["kind"] == "move" and st["theta"] in moved)
        for s in self.meta["spectra"]:
            self.last_exp[s["pol"]] = s["exposure_ms"]
        self.refresh_list()
        self.goto_next_undone(0)
        self.btn_ok.setEnabled(True)
        self.status.showMessage(f"Saving to {self.folder}")

    # ---------------------------------------------------------- steps
    def refresh_list(self):
        self.list.clear()
        for i, st in enumerate(self.steps):
            mark = "✓ " if st["done"] else "   "
            txt = st["text"].replace("\n", " · ")
            it = QtWidgets.QListWidgetItem(mark + txt)
            if i == self.cur:
                f = it.font(); f.setBold(True); it.setFont(f)
            self.list.addItem(it)
        if 0 <= self.cur < len(self.steps):
            self.list.setCurrentRow(self.cur)

    def goto_next_undone(self, start):
        for i in list(range(start, len(self.steps))) + list(range(0, start)):
            if not self.steps[i]["done"]:
                self.cur = i
                break
        else:
            self.cur = len(self.steps)
        self.show_current()

    def show_current(self):
        if self.cur >= len(self.steps):
            self.instr.setText("All done.\nAnalyse with twist_app.py → “Measurement folder”.")
            self.btn_ok.setEnabled(False)
        else:
            st = self.steps[self.cur]
            self.instr.setText(st["text"] + ("\n(redo)" if st["done"] else ""))
            self.btn_ok.setText("OK — continue  (Enter)" if st["kind"] == "move"
                                else "OK — acquire  (Enter)")
            self.btn_ok.setEnabled(True)
        self.refresh_list()

    def step_clicked(self, item):
        if self.busy:
            return
        self.cur = self.list.row(item)
        self.show_current()

    def do_step(self):
        if self.busy or not self.btn_ok.isEnabled() or self.cur >= len(self.steps):
            return
        st = self.steps[self.cur]
        if st["kind"] == "move":
            st["done"] = True
            self.goto_next_undone(self.cur + 1)
            return
        if isinstance(self.spec, FakeSpectrometer):
            self.spec.state = (dict(dark=True) if st["kind"] == "dark" else
                               dict(dark=False, theta=st["theta"], rho=st["rho"], pol=st["pol"]))
        self.busy = True
        self.btn_ok.setEnabled(False)
        old = getattr(self, "_thread", None)
        if old is not None and old.isRunning():
            old.quit(); old.wait()
        self._thread = QtCore.QThread()
        self.worker = Worker(self.spec, self.meta, st, self.last_exp)
        self.worker.moveToThread(self._thread)
        self._thread.started.connect(self.worker.run)
        self.worker.status.connect(self.status.showMessage)
        self.worker.done.connect(self.acquired)
        self.worker.failed.connect(self.acq_failed)
        self.worker.done.connect(self._thread.quit)
        self.worker.failed.connect(self._thread.quit)
        self._thread.start()

    def acq_failed(self, msg):
        self.busy = False
        self.btn_ok.setEnabled(True)
        QtWidgets.QMessageBox.warning(self, "Acquisition", msg)

    def acquired(self, res):
        self.busy = False
        kind, data = res
        st = self.steps[self.cur]
        now = dt.datetime.now().isoformat(timespec="seconds")
        sat = self.meta["saturation_raw"]
        if kind == "dark":
            self.meta["darks"] = [d for d in self.meta["darks"] if d["when"] != st["when"]]
            for t, wl, s, raw_max, off in data:
                fn = f"dark_{st['when']}_{t:g}ms.txt"
                self.save(fn, wl, s, t, raw_max, off, f"dark {st['when']}")
                self.meta["darks"].append(dict(file=fn, when=st["when"], exposure_ms=t,
                                               n_avg=self.meta["n_avg"], time=now,
                                               max_raw=raw_max, offset=off))
            self.plot(wl, data[-1][2], f"dark {st['when']} ({data[-1][0]:g} ms)")
            self.info.setText(f"{len(data)} dark spectra saved.")
        else:
            t, wl, s, raw_max, off = data[0]
            fn = fname(st)
            P, A = angles(self.meta, st["rho"], st["pol"])
            self.save(fn, wl, s, t, raw_max, off,
                      f"theta {st['theta']:g} rho {st['rho']:g} {st['pol']} P {P:g} A {A:g}")
            self.meta["spectra"] = [x for x in self.meta["spectra"] if x["key"] != st["key"]]
            self.meta["spectra"].append(dict(
                key=st["key"], file=fn, theta=st["theta"], rho=st["rho"], pol=st["pol"],
                P=P, A=A, exposure_ms=t, n_avg=self.meta["n_avg"], time=now,
                max_raw=raw_max, offset=off))
            self.last_exp[st["pol"]] = t
            warn = "  ⚠ SATURATED" if raw_max >= sat else ""
            self.info.setText(f"{fn}: exposure {t:g} ms, max raw {raw_max:.0f} / {sat}{warn}")
            self.plot(wl, s, f"{fn}  ({t:g} ms)")
            if warn:
                QtWidgets.QMessageBox.warning(
                    self, "Saturated", "The spectrum is saturated. Lower the exposure "
                    "(manual mode) or the saturation level, and redo this step.")
        st["done"] = True
        self.write_meta()
        self.goto_next_undone(self.cur + 1)

    # ---------------------------------------------------------- io
    def save(self, fn, wl, s, t, raw_max, off, what):
        hdr = (f"twist_acquire v{APP_VERSION} | {what} | exposure_ms {t:g} | "
               f"n_avg {self.meta['n_avg']} | offset {off:.1f} (subtracted) | "
               f"max_raw {raw_max:.0f}\nwavelength_nm\tcounts")
        np.savetxt(os.path.join(self.folder, fn), np.column_stack([wl, s]),
                   fmt=["%.3f", "%.3f"], delimiter="\t", header=hdr)

    def write_meta(self):
        tmp = os.path.join(self.folder, META + ".tmp")
        with open(tmp, "w") as f:
            json.dump(self.meta, f, indent=1)
        os.replace(tmp, os.path.join(self.folder, META))

    def plot(self, wl, s, title):
        self.ax.clear()
        self.ax.set_aspect("auto")
        self.ax.plot(wl, s, lw=0.8)
        self.ax.set_xlabel("λ (nm)"); self.ax.set_ylabel("counts (offset-corrected)")
        self.ax.set_title(title, fontsize=9)
        self.ax.set_xlim(350, 900)
        self.canvas.draw_idle()

    def closeEvent(self, ev):
        if self.busy:
            ev.ignore(); return
        self.save_settings()
        if self.spec is not None:
            try:
                self.spec.close()
            except Exception:
                pass
        super().closeEvent(ev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", action="store_true", help="fake spectrometer + twisted cell")
    a = ap.parse_args()
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(simulate=a.simulate)
    w.resize(1350, 800)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
