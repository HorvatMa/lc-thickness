# lc_thickness

A Qt GUI that finds the thickness `d` of a planar nematic cell from its spectrum. The cell sits at 45° between crossed (or parallel) polarizers. Works in transmission or reflection.

```
~/.venv/base/bin/python app.py      # needs PyQt5, numpy, scipy, matplotlib (requirements.txt)
```

The full physics write-up (Jones calculus, dispersion and group birefringence, normalization modes, reflection, systematic errors, first results) is in the notes vault: `phd-notes/doc/dynamic-cell/analysis/Cell thickness from crossed-polarizer spectra.md`.

## Model

At normal incidence, a uniform planar cell at 45° between crossed polarizers transmits

    T(λ) = sin²(π Δn(λ) d / λ)          (parallel polarizers: cos²)

(textbook result for a uniaxial retarder; no source is cited here). In **reflection** (epi-illumination, crossed polarizers) the light crosses the layer twice, so Δn·d → 2Δn·d. Front-surface reflections keep their polarization and are blocked by the analyzer.

The fitted model is `T_model = a·sin²(π p Δn(λ) d / λ) + b`, with p = 1 (transmission) or 2 (reflection). The measured T comes from one of three **normalization modes**, where D is the optional dark spectrum:

| Mode | Files | T | Comment |
|---|---|---|---|
| Lamp reference | S (cell), R (lamp) | (S−D)/(R−D) | R taken with parallel polarizers and no cell (in reflection: a mirror) |
| Crossed + parallel pair | S⊥, S∥ (same spot) | (S⊥−D)/(S⊥+S∥−2D) | uses T⊥ + T∥ = 1 for a non-absorbing cell; lamp, spectrometer **and** polarizer spectra cancel. In reflection S∥ also contains the front-surface reflection, so a < 1 (keep a free) |
| No reference | S only | – | uses only the wavelengths of the minima (see below) |

The free parameters `a` and `b` absorb imperfect polarizers, an azimuth that is not exactly 45° (factor sin²2φ), reflections and stray light. The "Amplitude a, offset b" setting has three options:
- **Free**: the default for lamp mode, where a also absorbs intensity changes between sample and reference.
- **Constrained**: the default for pair mode, where physically a ≈ 1 and b ≈ 0. Bounds default to a ∈ [0.8, 1.05], b ∈ [−0.05, 0.1].
- **Fixed**: a = 1, b = 0.

A low fitted amplitude with a large offset usually means d is in the wrong interference order. Free a and b then flatten the model curve to a compromise. With constraints, the wrong order shows up as visibly misplaced fringes.

When "Use first estimate" is ticked, the app also scans 0.5–60 µm and warns if a much better fit lies outside the estimate window.

`d` is found in two steps:

1. A brute-force scan over d. At each d, a and b are solved linearly. The thickness step changes the phase by ≤ 0.03 rad.
2. A nonlinear least-squares refinement starting from the best point of the scan.

The scan is needed because the residual has one local minimum per interference order. The *RMS vs d* plot and the "next-best minimum" line show how well the correct order is singled out.

**No-reference mode.** The zeros of S = L(λ)·sin²(…) do not depend on the lamp spectrum L(λ). The minima are detected (with a minimum prominence setting) and refined with a parabola. Then p·Δn(λ_k)·d/λ_k = q_0 − k is fitted, with integer orders (half-integer for parallel polarizers). Neighbouring minima are assumed to be neighbouring orders. For each q_0, d follows by linear least squares; the q_0 with the smallest residual in the scan range wins. A missed or spurious minimum breaks this, so check the red markers. The method needs ≳ 3 minima, so it suits thick cells; leakage on a sloped background shifts the minima slightly.

Optionally, add a spectrum of the same spot with the other polarizer setting. Its minima get half-integer orders and are fitted together with the first set (shared d, separate q_0). Only positions are used, so the two spectra may have **different intensity scales and tilts**, as with a hand-held spectrometer. In **transmission** both sets are true zeros. In **reflection** the parallel spectrum also contains the front-surface reflection, so its minima are not zeros and are shifted by the spectral slope. There, use the crossed minima alone.

**Hand-held / unstable coupling.** The lamp and pair modes divide spectra taken at different times, so a changing intensity scale or spectral tilt between them distorts T. The minima mode is insensitive to this, as long as the spectrometer records all wavelengths at the same time (array spectrometer).

**Fit range.** Drag in the transmission plot to set the λ fit window: drag inside it to move it, drag its edges to resize it, or drag elsewhere to draw a new one. The λ-range fields follow, and an existing fit is redone automatically. Zoom/pan mode of the toolbar has to be off.

* **Use first estimate** ticked: scan only `d0 ± window`.
* Unticked: scan the full range (default 1–60 µm).

## Δn(λ) data

Each index follows the extended Cauchy model n(λ) = A + B/λ² + C/λ⁴ (λ in µm). The coefficients come from the refractiveindex.info database (`database/data/other/liquid crystals/`), which transcribes these papers:

| LC | Dataset | T | Fit range | Δn(550 nm) |
|---|---|---|---|---|
| E7 | Li et al. 2005 | 25 °C | 450–656 nm | 0.2236 |
| E7 | Tkachenko et al. 2006 | not stated | 532–1700 nm | 0.2224 |
| 5CB | Li et al. 2005 | 25 °C | 450–656 nm | 0.184 |
| 5CB | Tkachenko et al. 2006 | not stated | 532–1700 nm | 0.178 |

The values were not checked against the tables in the papers themselves (no PDFs available). Use "Custom (Cauchy)" to enter your own coefficients; B = C = 0 gives a constant Δn.

**Accuracy.** The fit really measures the retardation Γ = Δn·d, so d ∝ 1/Δn. The GUI refits the same data with every dataset of the selected LC to show the systematic spread. Δn also depends on temperature.

### References
- J. Li, C.-H. Wen, S. Gauza, R. Lu, S.-T. Wu (2005). Refractive indices of liquid crystals for display applications. *J. Display Technol.* 1, 51–61. doi:10.1109/JDT.2005.853357
- V. Tkachenko, G. Abbate, A. Marino, F. Vita, M. Giocondo, A. Mazzulla, F. Ciuchi, L. De Stefano (2006). Nematic liquid crystal optical dispersion in the visible-near infrared range. *Mol. Cryst. Liq. Cryst.* 454, 263–271. doi:10.1080/15421400600655816

## Input files

The loader accepts CSV/TXT with `,` `;` tab or whitespace delimiters and decimal commas. Non-numeric header lines are skipped. The λ and intensity columns can be chosen (1-based). Wavelengths in µm or m are converted to nm. The reference is interpolated onto the sample grid. Wavelengths where the reference falls below the "min. reference" fraction of its maximum are ignored.

*Save results…* writes `<name>.csv` (λ, T_meas, T_fit, with the fit report in the header) and `<name>.png`.

## Files
- `app.py`: GUI
- `model.py`: loading, normalization, scan + fit
- `dispersion.py`: Cauchy coefficients and references
