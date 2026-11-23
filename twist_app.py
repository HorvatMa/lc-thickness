"""Anchoring strength W of a patterned surface from twisted-cell spectra (GUI).

Cell: patterned (BY) plate where the light enters, rubbed plate where it exits.
List the spectra with their region pattern angle θ (BY easy axis relative to
the rubbing, as patterned: e.g. 0, +80, −80), polarizer P and analyzer A
(lab angles, same rotation sense as θ). One region → W with δ fixed;
two or three regions → δ (pattern rotation vs rubbing) is fitted as well.

Run:  ~/.venv/base/bin/python twist_app.py
Physics and conventions: twist_model.py and the analysis note
"Anchoring strength from twisted-cell spectra" in the notes vault.
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

import twist_model as tm
from dispersion import MATERIALS

NORM_MODES = [("pair", "Crossed + parallel pairs  T = S/(S+S′)"),
              ("lamp", "Lamp reference  T = (S−D)/(R−D)"),
              ("T", "Files already contain T")]
COLS = ["θ_set (°)", "P (°)", "A (°)", "File"]


def dspin(lo, hi, val, decimals=2, step=1.0, suffix=""):
    w = QtWidgets.QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(decimals)
    w.setSingleStep(step)
    w.setValue(val)
    if suffix:
        w.setSuffix(suffix)
    return w


class FitWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(str)
    done = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, spectra, settings, extra):
        super().__init__()
        self.spectra, self.settings, self.extra = spectra, settings, extra

    def run(self):
        try:
            res = tm.run_all(self.spectra, self.settings, self.extra,
                             progress=self.progress.emit)
            self.done.emit(res)
        except Exception as e:  # report in the GUI instead of crashing
            self.failed.emit(f"{type(e).__name__}: {e}")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Twisted-cell anchoring fit")
        self.qs = QtCore.QSettings("lc_thickness", "twist_app")
        self.result = None
        self.spectra = None

        # ---------- spectra table
        self.table = QtWidgets.QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        btn_add = QtWidgets.QPushButton("Add files…")
        btn_rm = QtWidgets.QPushButton("Remove")
        btn_load = QtWidgets.QPushButton("Load list…")
        btn_save = QtWidgets.QPushButton("Save list…")
        btn_add.clicked.connect(self.add_files)
        btn_rm.clicked.connect(self.remove_rows)
        btn_load.clicked.connect(self.load_list)
        btn_save.clicked.connect(self.save_list)
        self.theta_new = dspin(-89.9, 89.9, 0.0, 1, 5.0, "°")
        tb = QtWidgets.QHBoxLayout()
        for w in (btn_add, QtWidgets.QLabel("with θ ="), self.theta_new,
                  btn_rm, btn_load, btn_save):
            tb.addWidget(w)
        tb.addStretch()

        # ---------- settings
        self.lc = QtWidgets.QComboBox()
        for mat, sets in MATERIALS.items():
            for name in sets:
                self.lc.addItem(f"{mat} – {name}", (mat, name))
        self.k22 = dspin(0.1, 50, 6.2, 2, 0.1, " pN")
        self.dmin = dspin(0.5, 200, 5.0, 2, 1, " µm")
        self.dmax = dspin(0.5, 200, 40.0, 2, 1, " µm")
        self.lmin = dspin(300, 2000, 450, 0, 10, " nm")
        self.lmax = dspin(300, 2000, 700, 0, 10, " nm")
        self.lamcol = QtWidgets.QSpinBox(); self.lamcol.setRange(1, 50); self.lamcol.setValue(1)
        self.intcol = QtWidgets.QSpinBox(); self.intcol.setRange(1, 50); self.intcol.setValue(2)
        self.norm = QtWidgets.QComboBox()
        for key, label in NORM_MODES:
            self.norm.addItem(label, key)
        self.ref = QtWidgets.QLineEdit(); self.ref.setPlaceholderText("lamp reference (lamp mode)")
        self.dark = QtWidgets.QLineEdit(); self.dark.setPlaceholderText("dark spectrum (optional)")
        ref_b = QtWidgets.QPushButton("…"); ref_b.setFixedWidth(30)
        dark_b = QtWidgets.QPushButton("…"); dark_b.setFixedWidth(30)
        ref_b.clicked.connect(lambda: self.pick(self.ref))
        dark_b.clicked.connect(lambda: self.pick(self.dark))
        self.free_ab = QtWidgets.QCheckBox("Free amplitude a and offset b per spectrum")
        self.free_ab.setChecked(True)
        self.share_d = QtWidgets.QCheckBox("Same d in all regions")
        self.share_d.setChecked(True)
        self.fit_delta = QtWidgets.QCheckBox("Fit pattern offset δ (needs ≥ 2 regions)")
        self.fit_delta.setChecked(True)
        self.delta_fixed = dspin(-30, 30, 0.0, 2, 0.1, "°")
        self.psi_guess = QtWidgets.QLineEdit()
        self.psi_guess.setPlaceholderText("blank = scan 0–90°")
        self.syst = QtWidgets.QCheckBox("Refit with the other Δn datasets (systematics)")

        form = QtWidgets.QFormLayout()
        form.addRow("LC / Δn(λ)", self.lc)
        form.addRow("K22", self.k22)
        dr = QtWidgets.QHBoxLayout(); dr.addWidget(self.dmin); dr.addWidget(self.dmax)
        form.addRow("d scan range", dr)
        lr = QtWidgets.QHBoxLayout(); lr.addWidget(self.lmin); lr.addWidget(self.lmax)
        form.addRow("λ fit range", lr)
        cr = QtWidgets.QHBoxLayout(); cr.addWidget(QtWidgets.QLabel("λ")); cr.addWidget(self.lamcol)
        cr.addWidget(QtWidgets.QLabel("I")); cr.addWidget(self.intcol); cr.addStretch()
        form.addRow("Columns", cr)
        form.addRow("Normalization", self.norm)
        rr = QtWidgets.QHBoxLayout(); rr.addWidget(self.ref); rr.addWidget(ref_b)
        form.addRow("Reference", rr)
        dk = QtWidgets.QHBoxLayout(); dk.addWidget(self.dark); dk.addWidget(dark_b)
        form.addRow("Dark", dk)
        form.addRow(self.free_ab)
        form.addRow(self.share_d)
        form.addRow(self.fit_delta)
        form.addRow("δ if fixed", self.delta_fixed)
        form.addRow("ψ_rub guess (°)", self.psi_guess)
        form.addRow(self.syst)

        self.btn_fit = QtWidgets.QPushButton("Fit")
        self.btn_fit.clicked.connect(self.run_fit)
        self.btn_out = QtWidgets.QPushButton("Save results…")
        self.btn_out.clicked.connect(self.save_results)
        self.btn_out.setEnabled(False)
        fb = QtWidgets.QHBoxLayout(); fb.addWidget(self.btn_fit); fb.addWidget(self.btn_out)

        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        ll.addLayout(tb)
        ll.addWidget(self.table, 2)
        ll.addLayout(form)
        ll.addLayout(fb)

        # ---------- plots + report
        self.fig = Figure(figsize=(8, 6), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.report = QtWidgets.QPlainTextEdit(); self.report.setReadOnly(True)
        f = self.report.font(); f.setFamily("monospace"); self.report.setFont(f)
        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        pw = QtWidgets.QWidget(); pl = QtWidgets.QVBoxLayout(pw)
        pl.addWidget(NavigationToolbar(self.canvas, self)); pl.addWidget(self.canvas)
        right.addWidget(pw); right.addWidget(self.report)
        right.setSizes([600, 300])

        split = QtWidgets.QSplitter()
        split.addWidget(left); split.addWidget(right)
        split.setSizes([520, 900])
        self.setCentralWidget(split)
        self.status = self.statusBar()
        self.restore()

    # ---------------------------------------------------------- table
    def add_row(self, row: tm.Row):
        r = self.table.rowCount()
        self.table.insertRow(r)
        for c, v in enumerate((row.theta, row.P, row.A)):
            self.table.setItem(r, c, QtWidgets.QTableWidgetItem(f"{v:g}"))
        it = QtWidgets.QTableWidgetItem(os.path.basename(row.path))
        it.setData(QtCore.Qt.UserRole, row.path)
        it.setToolTip(row.path)
        it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
        self.table.setItem(r, 3, it)

    def rows(self):
        out = []
        for r in range(self.table.rowCount()):
            try:
                th, P, A = (float(self.table.item(r, c).text().replace(",", "."))
                            for c in range(3))
            except (AttributeError, ValueError):
                raise ValueError(f"Row {r + 1}: θ, P and A must be numbers.")
            out.append(tm.Row(self.table.item(r, 3).data(QtCore.Qt.UserRole), th, P, A))
        return out

    def add_files(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Spectra", self.qs.value("dir", ""),
            "Spectra (*.csv *.txt *.dat);;All files (*)")
        if not paths:
            return
        self.qs.setValue("dir", os.path.dirname(paths[0]))
        missing = 0
        for p in paths:
            P, A = tm.angles_from_name(p)
            missing += P is None or A is None
            self.add_row(tm.Row(p, self.theta_new.value(),
                                P if P is not None else 0.0,
                                A if A is not None else 90.0))
        if missing:
            self.status.showMessage(f"{missing} file(s): P/A not found in the name "
                                    "(pattern …P30_A120…); set them in the table.", 8000)

    def remove_rows(self):
        for r in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(r)

    def load_list(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Spectrum list", self.qs.value("dir", ""), "CSV (*.csv)")
        if not p:
            return
        try:
            rows = tm.read_manifest(p)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Load list", str(e)); return
        self.table.setRowCount(0)
        for r in rows:
            self.add_row(r)

    def save_list(self):
        p, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save spectrum list", self.qs.value("dir", ""), "CSV (*.csv)")
        if p:
            try:
                tm.write_manifest(p, self.rows())
            except ValueError as e:
                QtWidgets.QMessageBox.warning(self, "Save list", str(e))

    def pick(self, edit):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Spectrum", self.qs.value("dir", ""),
            "Spectra (*.csv *.txt *.dat);;All files (*)")
        if p:
            edit.setText(p)

    # ---------------------------------------------------------- fit
    def settings(self):
        mat, name = self.lc.currentData()
        g = self.psi_guess.text().strip()
        return tm.FitSettings(
            dn_func=MATERIALS[mat][name].delta_n, K22=self.k22.value() * 1e-12,
            lam_range=(self.lmin.value(), self.lmax.value()),
            d_range=(self.dmin.value(), self.dmax.value()),
            free_ab=self.free_ab.isChecked(), share_d=self.share_d.isChecked(),
            fit_delta=self.fit_delta.isChecked(),
            delta_fixed=self.delta_fixed.value(),
            psi_rub_guess=float(g.replace(",", ".")) if g else None)

    def run_fit(self):
        try:
            rows = self.rows()
            if not rows:
                raise ValueError("Add spectra first.")
            s = self.settings()
            if s.d_range[0] >= s.d_range[1] or s.lam_range[0] >= s.lam_range[1]:
                raise ValueError("Check the d and λ ranges.")
            self.spectra, self.warn = tm.build_spectra(
                rows, self.norm.currentData(), self.lamcol.value() - 1,
                self.intcol.value() - 1, self.ref.text() or None,
                self.dark.text() or None)
            if not self.spectra:
                raise ValueError("No usable spectra.\n" + "\n".join(self.warn))
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Fit", str(e)); return
        extra = {}
        if self.syst.isChecked():
            mat, name = self.lc.currentData()
            extra = {n: c.delta_n for n, c in MATERIALS[mat].items() if n != name}
        self.s = s
        self.btn_fit.setEnabled(False)
        self.thread = QtCore.QThread()
        self.worker = FitWorker(self.spectra, s, extra)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(lambda m: self.status.showMessage(m))
        self.worker.done.connect(self.fit_done)
        self.worker.failed.connect(self.fit_failed)
        self.worker.done.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.start()

    def fit_failed(self, msg):
        self.btn_fit.setEnabled(True)
        self.status.showMessage("Fit failed.", 5000)
        QtWidgets.QMessageBox.warning(self, "Fit", msg)

    def fit_done(self, res):
        self.btn_fit.setEnabled(True)
        self.btn_out.setEnabled(True)
        self.status.showMessage("Done.", 3000)
        self.result = res
        pp, rfs, jf, pairs, syst = res
        txt = tm.format_report(rfs, jf, pairs, self.s, syst, self.warn)
        if len(jf.thetas) == 1:
            txt = ("Single region: δ fixed, so W depends on the assembly angle. "
                   "Use ±θ regions for a calibrated W.\n\n" + txt)
        self.report.setPlainText(txt)
        self.draw()

    def curves(self):
        pp, rfs, jf, pairs, syst = self.result
        lam_full = [sp.lam_nm for sp in pp.spectra]
        return pp, lam_full, tm.model_curves(pp, self.s, jf, lam_full)

    def draw(self):
        pp, lam_full, fits = self.curves()
        n = len(pp.spectra)
        ncol = min(4, n)
        nrow = int(np.ceil(n / ncol))
        self.fig.clear()
        lo, hi = self.s.lam_range
        for k, (sp, f) in enumerate(zip(pp.spectra, fits)):
            ax = self.fig.add_subplot(nrow, ncol, k + 1)
            m = (sp.lam_nm >= lo) & (sp.lam_nm <= hi)
            ax.plot(sp.lam_nm[m], sp.T[m], ".", ms=1.5, color="0.45")
            ax.plot(sp.lam_nm[m], f[m], "-", lw=1.0, color="C3")
            ax.set_title(f"θ={sp.theta:+g}° P={sp.P:g}° A={sp.A:g}°", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.set_ylim(-0.05, 1.05)
        self.canvas.draw_idle()

    def save_results(self):
        if not self.result:
            return
        p, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save results (base name)", self.qs.value("dir", ""), "CSV (*.csv)")
        if not p:
            return
        base = os.path.splitext(p)[0]
        pp, lam_full, fits = self.curves()
        with open(base + ".csv", "w") as f:
            for line in self.report.toPlainText().splitlines():
                f.write("# " + line + "\n")
            f.write("spectrum,theta_deg,P_deg,A_deg,lambda_nm,T_meas,T_fit\n")
            for sp, fit in zip(pp.spectra, fits):
                for l, t, m in zip(sp.lam_nm, sp.T, fit):
                    f.write(f"{sp.label},{sp.theta:g},{sp.P:g},{sp.A:g},{l:.3f},{t:.6g},{m:.6g}\n")
        self.fig.savefig(base + ".png", dpi=200)
        self.status.showMessage(f"Saved {base}.csv/.png", 5000)

    # ---------------------------------------------------------- persistence
    WIDGETS = ("k22", "dmin", "dmax", "lmin", "lmax", "delta_fixed")

    def restore(self):
        for n in self.WIDGETS:
            v = self.qs.value(n)
            if v is not None:
                getattr(self, n).setValue(float(v))
        i = self.qs.value("lc")
        if i is not None and int(i) < self.lc.count():
            self.lc.setCurrentIndex(int(i))

    def closeEvent(self, ev):
        for n in self.WIDGETS:
            self.qs.setValue(n, getattr(self, n).value())
        self.qs.setValue("lc", self.lc.currentIndex())
        super().closeEvent(ev)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(1400, 850)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
