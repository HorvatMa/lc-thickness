"""LC cell thickness from the transmission spectrum between crossed polarizers.

Run:  ~/.venv/base/bin/python app.py
"""
import os
import sys

import numpy as np
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

import model
from dispersion import CUSTOM, MATERIALS, Cauchy

LAM_REF_NM = 550.0  # wavelength at which the retardation is reported

MODE_LAMP, MODE_PAIR, MODE_MINIMA = range(3)
MODES = ["Lamp reference  T = S/R",
         "Crossed + parallel pair  T = S⊥/(S⊥+S∥)",
         "No reference (minima positions)"]


def dspin(lo, hi, val, decimals=2, step=1.0, suffix=""):
    w = QtWidgets.QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(decimals)
    w.setSingleStep(step)
    w.setValue(val)
    if suffix:
        w.setSuffix(suffix)
    return w


class FileRow(QtWidgets.QWidget):
    """Path field + browse/clear buttons for one spectrum file."""
    changed = QtCore.pyqtSignal()

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.edit = QtWidgets.QLineEdit()
        self.edit.setReadOnly(True)
        browse = QtWidgets.QPushButton("…")
        browse.setFixedWidth(30)
        clear = QtWidgets.QPushButton("×")
        clear.setFixedWidth(30)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.edit)
        lay.addWidget(browse)
        lay.addWidget(clear)
        browse.clicked.connect(self.browse)
        clear.clicked.connect(self.clear)

    def path(self):
        return self.edit.text() or None

    def browse(self):
        start = self.settings.value("last_dir", os.path.expanduser("~"))
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select spectrum", start,
            "Spectra (*.csv *.txt *.dat *.tsv);;All files (*)")
        if path:
            self.settings.setValue("last_dir", os.path.dirname(path))
            self.edit.setText(path)
            self.changed.emit()

    def clear(self):
        self.edit.clear()
        self.changed.emit()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LC cell thickness — crossed-polarizer spectrum")
        self.settings = QtCore.QSettings("FMF", "lc_thickness")
        self.result = None
        self.fit_curve = None
        self.sample = self.ref = self.dark = None
        self.T = self.S = self.minima = None
        self.span = None    # draggable fit window in the transmission plot
        self.S2 = self.minima2 = None

        # ------------------------------------------------ left panel
        panel = QtWidgets.QWidget()
        pl = QtWidgets.QVBoxLayout(panel)

        g_files = QtWidgets.QGroupBox("Spectra")
        f = QtWidgets.QFormLayout(g_files)
        self.geometry = QtWidgets.QComboBox()
        self.geometry.addItems(["Transmission (single pass)",
                                "Reflection (double pass, 2Δn·d)"])
        f.addRow("Geometry:", self.geometry)
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(MODES)
        self.mode.setToolTip(
            "Lamp reference: R = lamp with parallel polarizers, no cell "
            "(reflection: mirror).\nPair: same spot measured with crossed and "
            "parallel polarizers; cancels lamp and polarizer spectra.\n"
            "No reference: only the wavelengths of the minima are used; "
            "optionally add the same spot with the other polarizer setting.\n"
            "Only the minima mode is insensitive to intensity changes between "
            "measurements (e.g. a hand-held spectrometer).")
        f.addRow("Normalization:", self.mode)
        self.file_sample = FileRow(self.settings)
        self.file_ref = FileRow(self.settings)
        self.file_dark = FileRow(self.settings)
        self.lbl_sample = QtWidgets.QLabel()
        self.lbl_ref = QtWidgets.QLabel()
        f.addRow(self.lbl_sample, self.file_sample)
        f.addRow(self.lbl_ref, self.file_ref)
        f.addRow("Dark (optional):", self.file_dark)
        self.dark_auto = QtWidgets.QCheckBox("No dark file: subtract median below")
        self.dark_auto.setChecked(True)
        self.dark_auto.setToolTip(
            "Subtract from each spectrum its own median count below this "
            "wavelength.\nOnly valid if no light reaches the spectrometer there "
            "(polarizers/glass block the UV).")
        self.dark_lam = dspin(200, 1000, 320, 0, 10, " nm")
        dr = QtWidgets.QHBoxLayout()
        dr.addWidget(self.dark_auto)
        dr.addWidget(self.dark_lam)
        f.addRow(dr)
        self.col_lam = QtWidgets.QSpinBox()
        self.col_lam.setRange(1, 50)
        self.col_lam.setValue(1)
        self.col_int = QtWidgets.QSpinBox()
        self.col_int.setRange(1, 50)
        self.col_int.setValue(2)
        cols = QtWidgets.QHBoxLayout()
        cols.addWidget(QtWidgets.QLabel("λ col"))
        cols.addWidget(self.col_lam)
        cols.addWidget(QtWidgets.QLabel("I col"))
        cols.addWidget(self.col_int)
        f.addRow("Columns:", cols)
        self.polarizers = QtWidgets.QComboBox()
        self.polarizers.addItems(["crossed  (T = sin²)", "parallel (T = cos²)"])
        f.addRow("Polarizers:", self.polarizers)
        pl.addWidget(g_files)

        g_lc = QtWidgets.QGroupBox("Liquid crystal Δn(λ)")
        f = QtWidgets.QFormLayout(g_lc)
        self.material = QtWidgets.QComboBox()
        self.material.addItems(list(MATERIALS) + [CUSTOM])
        self.dataset = QtWidgets.QComboBox()
        f.addRow("Material:", self.material)
        f.addRow("Dataset:", self.dataset)
        self.coef = {}
        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("A"), 0, 1)
        grid.addWidget(QtWidgets.QLabel("B [µm²]"), 0, 2)
        grid.addWidget(QtWidgets.QLabel("C [µm⁴]"), 0, 3)
        for r, ax in enumerate(("e", "o"), start=1):
            grid.addWidget(QtWidgets.QLabel(f"n_{ax}"), r, 0)
            for c, name in enumerate(("A", "B", "C"), start=1):
                w = dspin(-10, 10, 0, decimals=5, step=0.001)
                self.coef[f"{name}_{ax}"] = w
                grid.addWidget(w, r, c)
        f.addRow(grid)
        self.lc_info = QtWidgets.QLabel()
        self.lc_info.setWordWrap(True)
        self.lc_info.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        f.addRow(self.lc_info)
        pl.addWidget(g_lc)

        g_fit = QtWidgets.QGroupBox("Fit")
        f = QtWidgets.QFormLayout(g_fit)
        self.lam_min = dspin(200, 2000, 470, 0, 10, " nm")
        self.lam_max = dspin(200, 2000, 700, 0, 10, " nm")
        lr = QtWidgets.QHBoxLayout()
        lr.addWidget(self.lam_min)
        lr.addWidget(QtWidgets.QLabel("–"))
        lr.addWidget(self.lam_max)
        f.addRow("λ range:", lr)
        self.smooth = QtWidgets.QSpinBox()
        self.smooth.setRange(1, 101)
        self.smooth.setValue(1)
        self.smooth.setSuffix(" pts")
        f.addRow("Smoothing:", self.smooth)
        self.min_ref = dspin(0, 0.9, 0.05, 2, 0.01)
        self.min_ref.setToolTip("Ignore wavelengths where the reference is "
                                "below this fraction of its maximum")
        f.addRow("Min. reference:", self.min_ref)
        self.prominence = dspin(0.01, 0.9, 0.10, 2, 0.01)
        self.prominence.setToolTip("Minimum depth of a minimum, as a fraction "
                                   "of the spectrum's range (minima mode)")
        f.addRow("Min. prominence:", self.prominence)
        self.amp_mode = QtWidgets.QComboBox()
        self.amp_mode.addItems(["Free", "Constrained", "Fixed (a = 1, b = 0)"])
        self.amp_mode.setToolTip(
            "T = a·sin²(πΔn d/λ) + b.\nPair mode: physically a ≈ 1, b ≈ 0, "
            "so constrain them.\nLamp mode: a absorbs intensity changes "
            "between sample and reference, so leave it free.")
        f.addRow("Amplitude a, offset b:", self.amp_mode)
        self.a_lo = dspin(0, 5, 0.8, 2, 0.05)
        self.a_hi = dspin(0, 5, 1.05, 2, 0.05)
        self.b_lo = dspin(-1, 1, -0.05, 2, 0.01)
        self.b_hi = dspin(-1, 1, 0.10, 2, 0.01)
        br = QtWidgets.QHBoxLayout()
        for lbl, w in (("a", self.a_lo), ("–", self.a_hi),
                       ("b", self.b_lo), ("–", self.b_hi)):
            br.addWidget(QtWidgets.QLabel(lbl))
            br.addWidget(w)
        f.addRow("Bounds:", br)
        pl.addWidget(g_fit)

        g_d = QtWidgets.QGroupBox("Thickness")
        f = QtWidgets.QFormLayout(g_d)
        self.use_est = QtWidgets.QCheckBox("Use first estimate")
        self.use_est.setChecked(True)
        f.addRow(self.use_est)
        self.d_est = dspin(0.1, 1000, 25, 2, 0.5, " µm")
        self.d_win = dspin(0.1, 500, 5, 2, 0.5, " µm")
        er = QtWidgets.QHBoxLayout()
        er.addWidget(self.d_est)
        er.addWidget(QtWidgets.QLabel("±"))
        er.addWidget(self.d_win)
        f.addRow("Estimate:", er)
        self.d_min = dspin(0.1, 1000, 1, 2, 1, " µm")
        self.d_max = dspin(0.1, 1000, 60, 2, 1, " µm")
        sr = QtWidgets.QHBoxLayout()
        sr.addWidget(self.d_min)
        sr.addWidget(QtWidgets.QLabel("–"))
        sr.addWidget(self.d_max)
        f.addRow("Full scan:", sr)
        pl.addWidget(g_d)

        btns = QtWidgets.QHBoxLayout()
        self.btn_fit = QtWidgets.QPushButton("Fit thickness")
        self.btn_fit.setDefault(True)
        self.btn_save = QtWidgets.QPushButton("Save results…")
        self.btn_save.setEnabled(False)
        btns.addWidget(self.btn_fit)
        btns.addWidget(self.btn_save)
        pl.addLayout(btns)

        self.out = QtWidgets.QPlainTextEdit()
        self.out.setReadOnly(True)
        self.out.setMinimumHeight(180)
        pl.addWidget(self.out, 1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(500)

        # ------------------------------------------------ plots
        self.fig = Figure(figsize=(9, 7), constrained_layout=True)
        self.canvas = FigureCanvas(self.fig)
        axs = self.fig.subplots(2, 2)
        self.ax_raw, self.ax_dn, self.ax_T, self.ax_scan = axs.ravel()
        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.addWidget(NavigationToolbar(self.canvas, self))
        rl.addWidget(self.canvas)

        split = QtWidgets.QSplitter()
        split.addWidget(scroll)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        self.setCentralWidget(split)
        self.resize(1400, 850)

        # ------------------------------------------------ signals
        for w in (self.file_sample, self.file_ref, self.file_dark):
            w.changed.connect(self.data_changed)
        self.dark_auto.toggled.connect(self.data_changed)
        self.dark_lam.valueChanged.connect(self.data_changed)
        for w in (self.col_lam, self.col_int, self.smooth):
            w.valueChanged.connect(self.data_changed)
        for w in (self.min_ref, self.lam_min, self.lam_max):
            w.valueChanged.connect(self.data_changed)
        self.polarizers.currentIndexChanged.connect(self.data_changed)
        self.geometry.currentIndexChanged.connect(self.invalidate_fit)
        self.mode.currentIndexChanged.connect(self.mode_changed)
        self.prominence.valueChanged.connect(self.data_changed)
        self.material.currentIndexChanged.connect(self.material_changed)
        self.dataset.currentIndexChanged.connect(self.dataset_changed)
        for w in self.coef.values():
            w.valueChanged.connect(self.lc_changed)
        self.use_est.toggled.connect(self.update_d_enabled)
        self.amp_mode.currentIndexChanged.connect(self.amp_mode_changed)
        for w in (self.a_lo, self.a_hi, self.b_lo, self.b_hi):
            w.valueChanged.connect(self.invalidate_fit)
        self.btn_fit.clicked.connect(self.run_fit)
        self.btn_save.clicked.connect(self.save)

        self.material_changed()
        self.update_d_enabled()
        self.mode_changed()

    def mode_changed(self):
        mode = self.mode.currentIndex()
        self.lbl_sample.setText({MODE_LAMP: "Measured (cell):",
                                 MODE_PAIR: "Measured (crossed):",
                                 MODE_MINIMA: "Measured (pol. as set):"}[mode])
        self.lbl_ref.setText({MODE_LAMP: "Reference (lamp):",
                              MODE_PAIR: "Measured (parallel):",
                              MODE_MINIMA: "Other pol. (optional):"}[mode])
        if mode == MODE_PAIR:
            self.polarizers.setCurrentIndex(0)
        self.polarizers.setEnabled(mode != MODE_PAIR)
        self.prominence.setEnabled(mode == MODE_MINIMA)
        self.amp_mode.setEnabled(mode != MODE_MINIMA)
        self.amp_mode.setCurrentIndex(1 if mode == MODE_PAIR else 0)
        self.amp_mode_changed()
        self.min_ref.setEnabled(mode != MODE_MINIMA)
        self.data_changed()

    def amp_mode_changed(self, *_):
        on = (self.amp_mode.currentIndex() == 1
              and self.mode.currentIndex() != MODE_MINIMA)
        for w in (self.a_lo, self.a_hi, self.b_lo, self.b_hi):
            w.setEnabled(on)
        self.invalidate_fit()

    def ab_bounds(self):
        if self.amp_mode.currentIndex() != 1:
            return None
        return ((self.a_lo.value(), self.a_hi.value()),
                (self.b_lo.value(), self.b_hi.value()))

    def crossed(self):
        return self.polarizers.currentIndex() == 0

    def passes(self):
        return 2 if self.geometry.currentIndex() == 1 else 1

    def dn_eff(self, cauchy):
        """Effective birefringence passes·Δn(λ) seen by the light."""
        p = self.passes()
        return lambda lam_um: p * cauchy.delta_n(lam_um)
    # ------------------------------------------------------------ LC model
    def current_cauchy(self):
        mat = self.material.currentText()
        if mat == CUSTOM:
            v = {k: w.value() for k, w in self.coef.items()}
            return Cauchy(v["A_e"], v["B_e"], v["C_e"],
                          v["A_o"], v["B_o"], v["C_o"],
                          reference="user-defined")
        return MATERIALS[mat][self.dataset.currentText()]

    def material_changed(self):
        mat = self.material.currentText()
        self.dataset.blockSignals(True)
        self.dataset.clear()
        if mat == CUSTOM:
            self.dataset.addItem("user-defined")
        else:
            self.dataset.addItems(list(MATERIALS[mat]))
        self.dataset.blockSignals(False)
        for w in self.coef.values():
            w.setEnabled(mat == CUSTOM)
        self.dataset_changed()

    def dataset_changed(self):
        if self.material.currentText() != CUSTOM:
            c = self.current_cauchy()
            for k, w in self.coef.items():
                w.blockSignals(True)
                w.setValue(getattr(c, k))
                w.blockSignals(False)
        self.lc_changed()

    def lc_changed(self):
        c = self.current_cauchy()
        lam = LAM_REF_NM / 1000
        txt = (f"Δn({LAM_REF_NM:.0f} nm) = {c.delta_n(lam):.4f}   "
               f"(n_e = {c.n_e(lam):.4f}, n_o = {c.n_o(lam):.4f})")
        if c.reference != "user-defined":
            lo, hi = c.valid_nm
            txt += (f"<br>T: {c.temperature}; fitted range {lo}–{hi} nm"
                    f"<br><small>{c.reference}. doi:{c.doi}</small>")
        self.lc_info.setText(txt)
        self.invalidate_fit()

    def update_d_enabled(self):
        est = self.use_est.isChecked()
        self.d_est.setEnabled(est)
        self.d_win.setEnabled(est)
        self.d_min.setEnabled(not est)
        self.d_max.setEnabled(not est)

    # ------------------------------------------------------------ data
    def load(self, row):
        path = row.path()
        if not path:
            return None
        table = model.load_table(path)
        return model.spectrum_from_table(table, self.col_lam.value() - 1,
                                         self.col_int.value() - 1)

    def data_changed(self, *_):
        self.sample = self.ref = self.dark = None
        self.T = None      # (λ, T) for the lamp/pair modes
        self.S = self.S2 = None        # (λ, S − D) for the minima mode
        self.minima = self.minima2 = None
        mode = self.mode.currentIndex()
        try:
            self.sample = self.load(self.file_sample)
            self.ref = self.load(self.file_ref)
            self.dark = self.load(self.file_dark)
        except Exception as e:
            self.out.setPlainText(f"Error reading file:\n{e}")
        self.dark_auto.setEnabled(self.dark is None)
        self.dark_lam.setEnabled(self.dark is None)
        if self.dark is None and self.dark_auto.isChecked():
            self.sample = self.subtract_uv_dark(self.sample)
            self.ref = self.subtract_uv_dark(self.ref)
        n = self.smooth.value()
        d = self.dark if self.dark is not None else (None, None)
        dark_I = model.smooth(d[1], n) if d[1] is not None else None
        if self.sample is not None and mode == MODE_MINIMA:
            self.S, self.minima = self.spectrum_minima(self.sample, d[0], dark_I)
            if self.ref is not None:
                self.S2, self.minima2 = self.spectrum_minima(self.ref, d[0],
                                                             dark_I)
        elif self.sample is not None and self.ref is not None:
            lam, I_s = self.sample
            func = (model.pair_transmission if mode == MODE_PAIR
                    else model.normalized_transmission)
            self.T = (lam, func(lam, model.smooth(I_s, n), self.ref[0],
                                model.smooth(self.ref[1], n), d[0], dark_I,
                                self.min_ref.value()))
        self.invalidate_fit()

    def subtract_uv_dark(self, spec):
        if spec is None:
            return None
        lam, I = spec
        m = lam < self.dark_lam.value()
        return (lam, I - np.median(I[m])) if m.sum() >= 5 else spec

    def spectrum_minima(self, spec, lam_d, dark_I):
        """Dark-corrected, smoothed spectrum and its minima in the λ range."""
        lam, I = spec
        S = model.smooth(I, self.smooth.value()).astype(float)
        if lam_d is not None:
            S = S - np.interp(lam, lam_d, dark_I)
        lam_w, S_w = model.wavelength_window(lam, S, self.lam_min.value(),
                                             self.lam_max.value())
        minima = (model.find_minima(lam_w, S_w, self.prominence.value())
                  if lam_w.size > 5 else None)
        return (lam, S), minima

    def invalidate_fit(self, *_):
        self.result = None
        self.fit_curve = None
        self.btn_save.setEnabled(False)
        self.redraw()

    # ------------------------------------------------------------ fit
    def scan_range(self):
        if self.use_est.isChecked():
            d0, w = self.d_est.value(), self.d_win.value()
            return max(d0 - w, 0.05), d0 + w
        return self.d_min.value(), self.d_max.value()

    def fit_one(self, cauchy, scan_range=None):
        lo, hi = scan_range or self.scan_range()
        if self.mode.currentIndex() == MODE_MINIMA:
            return model.fit_minima(self.minima, self.dn_eff(cauchy), lo, hi,
                                    crossed=self.crossed(),
                                    lam_min2_nm=self.minima2)
        lam, T = model.wavelength_window(*self.T, self.lam_min.value(),
                                         self.lam_max.value())
        return model.fit_thickness(lam, T, self.dn_eff(cauchy), lo, hi,
                                   free_ab=self.amp_mode.currentIndex() != 2,
                                   ab_bounds=self.ab_bounds(),
                                   crossed=self.crossed())

    def run_fit(self):
        if self.mode.currentIndex() == MODE_MINIMA:
            if self.minima is None or self.minima.size < 2:
                self.out.setPlainText("Need a measured spectrum with at least 2 "
                                      "minima in the λ range (lower the "
                                      "prominence or widen the range).")
                return
        elif self.T is None:
            self.out.setPlainText("Load both spectra first.")
            return
        lo, hi = self.scan_range()
        if hi <= lo:
            self.out.setPlainText("Thickness scan range is empty.")
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            c = self.current_cauchy()
            r = self.fit_one(c)
            r_full = None
            if self.use_est.isChecked():
                lo_f, hi_f = 0.5, max(60.0, hi)
                try:
                    r_full = self.fit_one(c, (lo_f, hi_f))
                except ValueError:
                    pass
            others = []
            mat = self.material.currentText()
            if mat != CUSTOM:
                for label, c2 in MATERIALS[mat].items():
                    if label != self.dataset.currentText():
                        others.append((label, self.fit_one(c2).d))
        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.out.setPlainText(f"Fit failed: {e}")
            return
        QtWidgets.QApplication.restoreOverrideCursor()

        self.result = r
        lam = (self.S if self.S is not None else self.T)[0]
        if self.mode.currentIndex() == MODE_MINIMA:
            # model zeros for the set polarizers and for the other setting
            self.fit_curve = (lam, model.predicted_minima(
                lam, self.dn_eff(c), r.d, self.crossed()),
                model.predicted_minima(lam, self.dn_eff(c), r.d,
                                       not self.crossed()))
        else:
            self.fit_curve = (lam, model.model(
                lam / 1000, self.dn_eff(c)(lam / 1000), r.d, r.a, r.b,
                self.crossed()))
        self.btn_save.setEnabled(True)
        self.out.setPlainText(self.report(c, r, others, lo, hi, r_full))
        self.redraw()

    def report(self, c, r, others, lo, hi, r_full=None):
        lam_ref = LAM_REF_NM / 1000
        gamma = c.delta_n(lam_ref) * r.d * 1000
        mat = self.material.currentText()
        ds = self.dataset.currentText()
        minima_mode = self.mode.currentIndex() == MODE_MINIMA
        lines = [
            f"Material: {mat} — {ds}",
            f"{self.geometry.currentText()}; {self.mode.currentText()}; "
            f"{'crossed' if self.crossed() else 'parallel'} polarizers",
            f"d = {r.d:.3f} ± {r.d_err:.3f} µm   (± = statistical only)",
            f"Γ({LAM_REF_NM:.0f} nm) = Δn·d = {gamma:.0f} nm   "
            f"(order Γ/λ = {gamma / LAM_REF_NM:.2f})",
        ]
        if self.passes() == 2:
            lines.append(f"   double pass: total retardation 2Γ = {2 * gamma:.0f} nm")
        if minima_mode:
            n1, n2 = r.n_first, r.n_points - r.n_first
            q1 = r.orders[:n1]
            lines += [
                f"{n1} minima (orders {q1[0]:g} … {q1[-1]:g})"
                + (f" + {n2} from the other-polarizer spectrum "
                   f"(orders {r.orders[n1]:g} … {r.orders[-1]:g})" if n2 else ""),
                f"RMS residual = {r.rms:.4f} orders",
                f"λ = {self.lam_min.value():.0f}–{self.lam_max.value():.0f} nm, "
                f"scan {lo:.2f}–{hi:.2f} µm",
            ]
            if np.isfinite(r.second_d):
                lines.append(f"Next-best order assignment: d = {r.second_d:.3f} µm "
                             f"(RMS {r.second_rms:.4f} orders) — "
                             f"contrast {r.second_rms / max(r.rms, 1e-12):.1f}×")
            lines.append("Assumes consecutive minima = consecutive orders; "
                         "check that no minimum is missed or spurious.")
            if n2 and self.passes() == 2 and self.crossed():
                lines.append("⚠ Reflection: the parallel-polarizer spectrum "
                             "contains the front-surface reflection, so its "
                             "minima are not zeros and shift on a sloped "
                             "spectrum. Prefer the crossed minima alone.")
        else:
            lines += [
                f"a = {r.a:.3f}, b = {r.b:.3f}, RMS residual = {r.rms:.4f}",
                f"N = {r.n_points} points, λ = {self.lam_min.value():.0f}–"
                f"{self.lam_max.value():.0f} nm, scan {lo:.2f}–{hi:.2f} µm",
            ]
            if np.isfinite(r.second_d):
                lines.append(f"Next-best minimum: d = {r.second_d:.3f} µm "
                             f"(RMS {r.second_rms:.4f}) — "
                             f"contrast {r.second_rms / r.rms:.1f}×")
        if abs(r.d - lo) < 1e-3 * (hi - lo) + 0.01 or \
                abs(r.d - hi) < 1e-3 * (hi - lo) + 0.01:
            lines.append("⚠ Best d is at the edge of the scan range — widen it.")
        if r_full is not None and r_full.rms < 0.7 * r.rms \
                and not lo <= r_full.d <= hi:
            lines.append(f"⚠ A much better fit exists OUTSIDE the estimate "
                         f"window: d = {r_full.d:.3f} µm (RMS {r_full.rms:.4f} vs "
                         f"{r.rms:.4f}). Move the estimate or untick it.")
        if minima_mode and r.rms > 0.1:
            lines.append("⚠ Order assignment inconsistent: probably missed or "
                         "spurious minima (lamp structure?). The minima mode "
                         "needs several clean fringes (thick cells); for thin "
                         "cells use the lamp or pair mode.")
        elif not minima_mode and r.rms > 0.1:
            lines.append("⚠ Poor fit (RMS > 0.1): check the d range, the "
                         "material, the λ range and the normalization.")
        if others:
            lines.append("Same data, other Δn datasets (systematic spread):")
            for label, d in others:
                lines.append(f"   {label}: d = {d:.3f} µm")
        if c.valid_nm[0] is not None:
            lo_v, hi_v = c.valid_nm
            if self.lam_min.value() < lo_v or self.lam_max.value() > hi_v:
                lines.append(f"⚠ Fit range extends outside the range of the "
                             f"Cauchy fit ({lo_v}–{hi_v} nm): extrapolation.")
        lines.append("Note: d ∝ 1/Δn, so d is only as accurate as Δn "
                     "(temperature!); Γ is model-independent to first order.")
        return "\n".join(lines)

    # ------------------------------------------------------------ plotting
    def redraw(self):
        for ax in (self.ax_raw, self.ax_dn, self.ax_T, self.ax_scan):
            ax.clear()

        ax = self.ax_raw
        ref_label = ("reference" if self.mode.currentIndex() == MODE_LAMP
                     else "other polarizer")
        for spec, label in ((self.sample, "measured"), (self.ref, ref_label),
                            (self.dark, "dark")):
            if spec is not None:
                ax.plot(spec[0], spec[1], lw=1, label=label)
        ax.set_xlabel("λ [nm]")
        ax.set_ylabel("counts")
        ax.set_title("Raw spectra")
        if ax.lines:
            ax.legend(fontsize=8)

        ax = self.ax_dn
        c = self.current_cauchy()
        lam = np.linspace(400, 1000, 300)
        ax.plot(lam, c.delta_n(lam / 1000), color="C3")
        if c.valid_nm[0] is not None:
            ax.axvspan(*c.valid_nm, color="C3", alpha=0.08, label="fit range of data")
            ax.legend(fontsize=8)
        ax.axvspan(self.lam_min.value(), self.lam_max.value(), color="0.5",
                   alpha=0.12)
        ax.set_xlabel("λ [nm]")
        ax.set_ylabel("Δn")
        ax.set_title("Birefringence model (grey: fit window)")

        ax = self.ax_T
        if self.S is not None:
            ax.plot(*self.S, lw=1, color="C0", label="measured − dark")
            if self.minima is not None and self.minima.size:
                ax.plot(self.minima, np.interp(self.minima, *self.S), "o",
                        mfc="none", color="C3", label="detected minima")
            if self.S2 is not None:
                lam2, S2 = self.S2
                scale = np.nanmax(self.S[1]) / max(np.nanmax(S2), 1e-12)
                ax.plot(lam2, S2 * scale, lw=1, color="C2", alpha=0.8,
                        label="other polarizer (scaled)")
                if self.minima2 is not None and self.minima2.size:
                    ax.plot(self.minima2, np.interp(self.minima2, lam2, S2) * scale,
                            "s", mfc="none", color="C3")
            if self.fit_curve is not None:
                for i, x in enumerate(self.fit_curve[1]):
                    ax.axvline(x, color="C1", lw=0.8, alpha=0.7,
                               label=f"model zeros, d = {self.result.d:.3f} µm"
                               if i == 0 else None)
                if self.S2 is not None:
                    for x in self.fit_curve[2]:
                        ax.axvline(x, color="C2", lw=0.8, ls="--", alpha=0.7)
            ax.legend(fontsize=8)
            ax.set_ylabel("counts")
            ax.set_title("Measured spectrum, minima mode (drag: fit range)")
        if self.T is not None:
            ax.plot(*self.T, lw=1, color="C0", label="measured")
            if self.fit_curve is not None:
                lam_f, T_f = self.fit_curve
                m = (lam_f >= self.lam_min.value()) & (lam_f <= self.lam_max.value())
                ax.plot(lam_f, T_f, lw=0.8, color="C1", alpha=0.35)
                ax.plot(lam_f[m], T_f[m], lw=1.5, color="C1",
                        label=f"fit, d = {self.result.d:.3f} µm")
            ax.legend(fontsize=8)
            Tv = self.T[1][np.isfinite(self.T[1])]
            if Tv.size:
                hi = np.percentile(Tv, 99.5)
                ax.set_ylim(min(-0.05, np.percentile(Tv, 0.5)), max(1.1, hi * 1.1))
        ax.set_xlabel("λ [nm]")
        if self.S is None:
            ax.set_ylabel("T (normalized)")
            ax.set_title("Normalized transmission (drag: fit range)")
        self.make_span_selector(ax, self.S is not None or self.T is not None)

        ax = self.ax_scan
        if self.result is not None:
            r = self.result
            minima_mode = r.minima is not None
            ax.plot(r.d_grid, r.rms_grid, "o" if minima_mode else "-",
                    lw=1, ms=4)
            ax.axvline(r.d, color="C1", ls="--")
            ax.set_ylim(0, np.nanmax(r.rms_grid[np.isfinite(r.rms_grid)]) * 1.05)
        ax.set_xlabel("d [µm]")
        ax.set_ylabel("RMS residual" + (" [orders]" if self.result is not None
                                        and self.result.minima is not None
                                        else ""))
        ax.set_title("Thickness scan (global search)")

        self.canvas.draw_idle()

    def make_span_selector(self, ax, active):
        """(Re)create the draggable λ window; ax.clear() removes its artists."""
        if self.span is not None:
            self.span.disconnect_events()
            self.span = None
        if not active:
            return
        self.span = SpanSelector(
            ax, self.span_selected, "horizontal", minspan=5, useblit=True,
            props=dict(facecolor="0.5", alpha=0.15), interactive=True,
            drag_from_anywhere=True)
        self.span.extents = (self.lam_min.value(), self.lam_max.value())

    def span_selected(self, lo, hi):
        # defer: the redraw replaces the selector that is calling us
        QtCore.QTimer.singleShot(0, lambda: self.apply_span(lo, hi))

    def apply_span(self, lo, hi):
        if hi - lo < 5:
            return
        refit = self.result is not None
        for w, v in ((self.lam_min, lo), (self.lam_max, hi)):
            w.blockSignals(True)
            w.setValue(round(v))
            w.blockSignals(False)
        self.data_changed()
        if refit:
            self.run_fit()

    # ------------------------------------------------------------ save
    def save(self):
        if self.result is None:
            return
        start = self.settings.value("last_dir", os.path.expanduser("~"))
        base = os.path.splitext(self.file_sample.path() or "fit")[0] + "_thickness"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save results (CSV + PNG)",
            os.path.join(start, os.path.basename(base) + ".csv"), "CSV (*.csv)")
        if not path:
            return
        base = os.path.splitext(path)[0]
        r = self.result
        header = [f"sample: {self.file_sample.path()}",
                  f"second spectrum: {self.file_ref.path()}",
                  f"dark: {self.file_dark.path()}",
                  ""] + self.out.toPlainText().splitlines() + [""]
        if r.minima is not None:
            header.append("lambda_min_nm, order, residual_orders, file (1 = measured, "
                          "2 = other polarizer)")
            g = self.dn_eff(self.current_cauchy())(r.minima / 1000) / (r.minima / 1000)
            src = np.where(np.arange(r.minima.size) < r.n_first, 1, 2)
            data = np.column_stack([r.minima, r.orders, g * r.d - r.orders, src])
        else:
            header.append("lambda_nm, T_measured, T_fit")
            lam, T = self.T
            data = np.column_stack([lam, T, self.fit_curve[1]])
        np.savetxt(base + ".csv", data, delimiter=",", fmt="%.6g",
                   header="\n".join(header), encoding="utf-8")
        self.fig.savefig(base + ".png", dpi=200)
        self.out.appendPlainText(f"\nSaved {base}.csv and {base}.png")


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
